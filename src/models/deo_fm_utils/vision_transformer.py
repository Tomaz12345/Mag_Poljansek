# Implemented from: Wolf et al., "Brewing Stronger Features: Dual-Teacher Distillation for
#   Multispectral Earth Observation", CVPR 2026 (arXiv:2602.19863)
# Adapted from: https://github.com/wolfilip/DEO-FM/blob/main/DEO/vision_transformer.py
#   MIT License, Copyright (c) 2026 Filip Wolf; modified for this project.
#   That file is based on DINO, Apache License 2.0, Copyright (c) Facebook, Inc. and its affiliates.

from __future__ import annotations

import math
from collections.abc import Sequence
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .patch_embed import DEOPatchEmbed


def _drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _drop_path(x, self.drop_prob, self.training)


class _Mlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        out: torch.Tensor = self.drop(self.fc2(x))
        return out


class _Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool, attn_drop: float, proj_drop: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out: torch.Tensor = self.proj_drop(self.proj(x))
        return out


class _Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        drop: float,
        attn_drop: float,
        drop_path: float,
        norm_layer: type[nn.LayerNorm] | partial[nn.LayerNorm],
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = _Attention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = _Mlp(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DEOVisionTransformer(nn.Module):
    """ViT trunk with the same parameter layout as DEO's published checkpoint."""

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 8,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        # Kept for checkpoint compatibility; the wrapper attaches the
        # actually-used patch_embed_rgb / patch_embed_ms attributes.
        self.patch_embed = DEOPatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.blocks = nn.ModuleList(
            [
                _Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x: torch.Tensor, w: int, h: int) -> torch.Tensor:
        npatch = x.shape[1] - 1
        n = self.pos_embed.shape[1] - 1
        if npatch == n and w == h:
            return self.pos_embed
        class_pos = self.pos_embed[:, 0]
        patch_pos = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size + 0.1
        h0 = h // self.patch_size + 0.1
        side = int(math.sqrt(n))
        patch_pos = F.interpolate(
            patch_pos.reshape(1, side, side, dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / side, h0 / side),
            mode="bicubic",
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos.unsqueeze(0), patch_pos), dim=1)

    def get_intermediate_layers(self, x: torch.Tensor, n: Sequence[int]) -> list[torch.Tensor]:
        outputs: list[torch.Tensor] = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in n:
                outputs.append(x)
        return outputs


def vit_base(patch_size: int = 8, img_size: int = 256, **kwargs: object) -> DEOVisionTransformer:
    return DEOVisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        **kwargs,  # type: ignore[arg-type]
    )
