# Implemented from: Wang et al., "UNetFormer: A UNet-like transformer for efficient semantic
#   segmentation of remote sensing urban scene imagery", ISPRS J. Photogramm. Remote Sens. 190, 2022
# Adapted from: https://github.com/WangLibo1995/GeoSeg/blob/main/geoseg/models/UNetFormer.py
#   GNU GPL v3.0, Libo Wang; modified for this project.
#   Licence text: LICENSE

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_

from .conv_modules import Conv, ConvBN, SeparableConvBN


class GlobalLocalAttention(nn.Module):
    """Efficient Global-Local Attention.

    Global branch: window-based multi-head self-attention + cross-shaped
    window context interaction via AvgPool.
    Local branch: two parallel convolutions (1x1 + 3x3) with BatchNorm.

    Args:
        dim: Number of input channels.
        num_heads: Number of attention heads.
        qkv_bias: Add bias to QKV projection.
        window_size: Window size for local attention.
        relative_pos_embedding: Whether to use relative position bias.
    """

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 16,
        qkv_bias: bool = False,
        window_size: int = 8,
        relative_pos_embedding: bool = True,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim**-0.5
        self.ws = window_size

        # QKV projection via 1x1 conv
        self.qkv = Conv(dim, 3 * dim, kernel_size=1, bias=qkv_bias)

        # Local branch: two parallel convolutions
        self.local1 = ConvBN(dim, dim, kernel_size=3)
        self.local2 = ConvBN(dim, dim, kernel_size=1)

        self.proj = SeparableConvBN(dim, dim, kernel_size=window_size)

        # Cross-shaped window context interaction via AvgPool
        self.attn_x = nn.AvgPool2d(
            kernel_size=(window_size, 1),
            stride=1,
            padding=(window_size // 2 - 1, 0),
        )
        self.attn_y = nn.AvgPool2d(
            kernel_size=(1, window_size),
            stride=1,
            padding=(0, window_size // 2 - 1),
        )

        # Relative position bias
        self.relative_pos_embedding = relative_pos_embedding
        if self.relative_pos_embedding:
            self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))

            coords_h = torch.arange(self.ws)
            coords_w = torch.arange(self.ws)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.ws - 1
            relative_coords[:, :, 1] += self.ws - 1
            relative_coords[:, :, 0] *= 2 * self.ws - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index", relative_position_index)

            trunc_normal_(self.relative_position_bias_table, std=0.02)

    def _pad_feature(self, x: torch.Tensor, ps: int) -> torch.Tensor:
        """Pad feature map to be divisible by window size.

        Uses reflect padding when possible; falls back to constant padding
        when the pad size would exceed the input dimension (PyTorch constraint).
        """
        _, _, H, W = x.size()
        pad_w = (ps - W % ps) % ps
        pad_h = (ps - H % ps) % ps
        if pad_w > 0 or pad_h > 0:
            # Reflect padding requires pad < input dim
            mode = "reflect" if pad_w < W and pad_h < H else "constant"
            x = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)
        return x

    def _pad_out(self, x: torch.Tensor) -> torch.Tensor:
        """Pad output by 1 pixel on right and bottom using reflect padding."""
        return F.pad(x, pad=(0, 1, 0, 1), mode="reflect")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Output tensor of shape (B, C, H, W).
        """
        B, C, H, W = x.shape

        # Local branch: two parallel convolutions, summed
        local = self.local2(x) + self.local1(x)

        # Global branch: window-based self-attention
        x_padded = self._pad_feature(x, self.ws)
        _, _, Hp, Wp = x_padded.shape
        qkv = self.qkv(x_padded)

        # Window partition and reshape for multi-head attention
        # Reshape: (B, 3*C, Hp, Wp) -> (B, 3, num_heads, head_dim, num_h_windows, ws, num_w_windows, ws)
        # Then rearrange to: (3, B*num_h_windows*num_w_windows, num_heads, ws*ws, head_dim)
        q, k, v = self._window_partition_qkv(qkv, C, Hp, Wp)

        # Scaled dot-product attention
        dots = (q @ k.transpose(-2, -1)) * self.scale

        # Add relative position bias
        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[cast(torch.Tensor, self.relative_position_index).view(-1)].view(
                self.ws * self.ws, self.ws * self.ws, -1
            )
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            dots = dots + relative_position_bias.unsqueeze(0)

        attn = dots.softmax(dim=-1)
        attn_out = attn @ v

        # Reverse window partition
        attn_out = self._window_reverse(attn_out, C, Hp, Wp)

        # Remove padding
        attn_out = attn_out[:, :, :H, :W]

        # Cross-shaped window context interaction
        out = self.attn_x(F.pad(attn_out, pad=(0, 0, 0, 1), mode="reflect")) + self.attn_y(F.pad(attn_out, pad=(0, 1, 0, 0), mode="reflect"))

        # Fuse global and local branches (simple sum)
        out = out + local

        # Output projection via SeparableConvBN
        out = self._pad_out(out)
        out = self.proj(out)
        result: torch.Tensor = out[:, :, :H, :W]

        return result

    def _window_partition_qkv(
        self,
        qkv: torch.Tensor,
        C: int,
        Hp: int,
        Wp: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Partition QKV into windows for multi-head attention."""
        head_dim = C // self.num_heads
        num_h = Hp // self.ws
        num_w = Wp // self.ws

        # Reshape to (B, 3, num_heads, head_dim, num_h, ws, num_w, ws)
        qkv = qkv.reshape(-1, 3, self.num_heads, head_dim, num_h, self.ws, num_w, self.ws)
        # Permute to (3, B, num_h, num_w, num_heads, ws, ws, head_dim)
        qkv = qkv.permute(1, 0, 4, 6, 2, 5, 7, 3)
        # Reshape to (3, B*num_h*num_w, num_heads, ws*ws, head_dim)
        qkv = qkv.reshape(3, -1, self.num_heads, self.ws * self.ws, head_dim)

        q = qkv[0]
        k = qkv[1]
        v = qkv[2]
        return q, k, v

    def _window_reverse(
        self,
        attn_out: torch.Tensor,
        C: int,
        Hp: int,
        Wp: int,
    ) -> torch.Tensor:
        """Reverse window partition back to spatial feature map."""
        head_dim = C // self.num_heads
        num_h = Hp // self.ws
        num_w = Wp // self.ws
        B = attn_out.shape[0] // (num_h * num_w)

        # (B*num_h*num_w, num_heads, ws*ws, head_dim) -> (B, C, Hp, Wp)
        attn_out = attn_out.reshape(B, num_h, num_w, self.num_heads, self.ws, self.ws, head_dim)
        attn_out = attn_out.permute(0, 3, 6, 1, 4, 2, 5)  # (B, heads, head_dim, num_h, ws, num_w, ws)
        result: torch.Tensor = attn_out.reshape(B, C, Hp, Wp)
        return result


class Mlp(nn.Module):
    """MLP module using Conv2d 1x1 (operates on spatial feature maps)."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.ReLU6,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    """Global-Local Transformer Block (GLTB).

    Uses BatchNorm2d (not LayerNorm) and ReLU6 activation, matching the original
    UNetFormer paper (Fig. 2b).

    Structure:
        x -> BN -> GlobalLocalAttention -> + (residual) -> BN -> MLP -> + (residual) -> out

    Args:
        dim: Number of input channels.
        num_heads: Number of attention heads.
        mlp_ratio: Ratio of MLP hidden dim to embedding dim.
        qkv_bias: Add bias to QKV projections.
        drop: Dropout rate.
        attn_drop: Attention dropout rate (unused, kept for API compat).
        drop_path: Stochastic depth rate.
        act_layer: Activation function class.
        norm_layer: Normalization layer class.
        window_size: Window size for attention.
    """

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type[nn.Module] = nn.ReLU6,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
        window_size: int = 8,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = GlobalLocalAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            window_size=window_size,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.norm2 = norm_layer(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Output tensor of shape (B, C, H, W).
        """
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
