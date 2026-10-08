# Implemented from: Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision",
#   arXiv:2304.07193, 2023
# Reference implementation: https://github.com/facebookresearch/dinov2
# Module and parameter names follow the reference implementation so that the official checkpoints load.

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from .croma_utils.decoder import (
    FPNSegDecoder,
    LinearSegDecoder,
    UPerNetSegDecoder,
)
from .dinov2_utils import DINOv2Encoder, PatchEmbed
from .input_projection import InputProjection
from .pretrained_loading import log_pretrained_load_result

logger = logging.getLogger(__name__)


# Variant configurations matching official DINOv2 models
DINOV2_VARIANTS: dict[str, dict[str, Any]] = {
    "dinov2_small": {
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
        "patch_size": 14,
        "hub_name": "dinov2_vits14",
    },
    "dinov2_base": {
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "patch_size": 14,
        "hub_name": "dinov2_vitb14",
    },
    "dinov2_large": {
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "patch_size": 14,
        "hub_name": "dinov2_vitl14",
    },
    "dinov2_giant": {
        "embed_dim": 1536,
        "depth": 40,
        "num_heads": 24,
        "patch_size": 14,
        "hub_name": "dinov2_vitg14",
    },
}


class DINOv2(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        img_size: int,
        patch_size: int,
        input_resize: int | None,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        drop_rate: float,
        attn_drop_rate: float,
        drop_path_rate: float,
        decoder_type: str,
        decoder_channels: int,
        decoder_dropout: float,
        use_pretrained: bool,
        pretrained_path: str,
        freeze_encoder: bool,
        freeze_encoder_layers: int,
        freeze_encoder_layers_mode: str,
        variant: str | None,
        input_projection_type: str = "linear",
        input_projection_hidden: int = 32,
        input_projection_dims: str | list[int] | None = None,
    ) -> None:
        super().__init__()

        hub_name: str | None = None
        if variant is not None:
            variant_value = str(variant).lower()
            if variant_value in DINOV2_VARIANTS:
                config = DINOV2_VARIANTS[variant_value]
                embed_dim = config["embed_dim"]
                depth = config["depth"]
                num_heads = config["num_heads"]
                patch_size = config["patch_size"]
                hub_name = config["hub_name"]
                logger.info(f"Using DINOv2 variant '{variant_value}': {config}")

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.decoder_type = decoder_type

        if input_resize is None:
            self.input_resize = (img_size // patch_size) * patch_size
        else:
            self.input_resize = input_resize

        if self.input_resize != img_size:
            logger.info(
                f"Input will be resized from {img_size} to {self.input_resize} (patch_size={patch_size}, grid={self.input_resize // patch_size})"
            )

        self.grid_size = self.input_resize // patch_size
        self.num_patches = self.grid_size**2

        # Input projection: n_channels -> 3 (backbone always receives 3-channel input)
        self.input_proj = InputProjection(
            n_channels=n_channels,
            target_channels=3,
            projection_type=input_projection_type,
            hidden_channels=input_projection_hidden,
            dims=input_projection_dims,
        )

        self.patch_embed = PatchEmbed(
            img_size=self.input_resize,
            patch_size=patch_size,
            in_channels=3,
            embed_dim=embed_dim,
            norm_layer=nn.LayerNorm,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        out_indices = tuple(max(0, depth // 4 * (i + 1) - 1) for i in range(4))

        self.encoder = DINOv2Encoder(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            out_indices=out_indices,
        )

        in_channels = [embed_dim] * 4

        self.decoder: UPerNetSegDecoder | FPNSegDecoder | LinearSegDecoder
        if decoder_type == "upernet":
            self.decoder = UPerNetSegDecoder(
                in_channels=in_channels,
                channels=decoder_channels,
                num_classes=n_classes,
                dropout_rate=decoder_dropout,
            )
        elif decoder_type == "fpn":
            self.decoder = FPNSegDecoder(
                in_channels=in_channels,
                channels=decoder_channels,
                num_classes=n_classes,
                dropout_rate=decoder_dropout,
            )
        elif decoder_type == "linear":
            self.decoder = LinearSegDecoder(
                in_channels=embed_dim,
                num_classes=n_classes,
                dropout_rate=decoder_dropout,
            )
        else:
            raise ValueError(f"Unknown decoder type: {decoder_type}")

        self._init_weights()

        if use_pretrained:
            self._load_pretrained(pretrained_path, hub_name, variant)

        if freeze_encoder:
            self._freeze_encoder()
        elif freeze_encoder_layers > 0:
            self._freeze_encoder_layers(freeze_encoder_layers, mode=freeze_encoder_layers_mode)

        logger.info(
            f"DINOv2 initialized: n_channels={n_channels}, n_classes={n_classes}, "
            f"embed_dim={embed_dim}, depth={depth}, num_heads={num_heads}, "
            f"input_resize={self.input_resize}, decoder={decoder_type}"
        )

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _load_pretrained(
        self,
        pretrained_path: str,
        hub_name: str | None,
        variant: str | None,
    ) -> None:
        if hub_name is None and variant is not None:
            variant_value = str(variant).lower()
            if variant_value in DINOV2_VARIANTS:
                hub_name = DINOV2_VARIANTS[variant_value]["hub_name"]

        if hub_name is None:
            logger.warning("Cannot determine DINOv2 hub model name. Specify a variant or provide hub_name. Skipping pretrained weight loading.")
            return

        try:
            logger.info(f"Loading DINOv2 pretrained weights: torch.hub.load('{pretrained_path}', '{hub_name}')")
            hub_model = torch.hub.load(pretrained_path, hub_name, pretrained=True)
            state_dict = hub_model.state_dict()

            adapted = self._adapt_pretrained_weights(state_dict)

            missing, unexpected = self.load_state_dict(adapted, strict=False)

            log_pretrained_load_result(
                missing,
                unexpected,
                logger,
                model_name="DINOv2",
                expected_missing_prefixes=["decoder.", "teacher_"],
            )

            logger.info("Successfully loaded DINOv2 pretrained weights")

        except Exception as e:
            logger.warning(f"Failed to load pretrained weights: {e}")
            logger.info("Continuing with random initialization")

    def _adapt_pretrained_weights(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        adapted: dict[str, torch.Tensor] = {}

        skip_keys = {"mask_token"}
        skip_prefixes = ("head.",)

        for key, value in state_dict.items():
            if key in skip_keys or any(key.startswith(p) for p in skip_prefixes):
                continue

            new_key = key

            if key.startswith("blocks."):
                new_key = key.replace("blocks.", "encoder.blocks.", 1)
            elif key.startswith("norm."):
                new_key = key.replace("norm.", "encoder.norm.", 1)

            adapted[new_key] = value

        if "pos_embed" in adapted:
            pretrained_pos = adapted["pos_embed"]
            target_num_tokens = 1 + self.num_patches

            if pretrained_pos.shape[1] != target_num_tokens:
                adapted["pos_embed"] = self._interpolate_pos_embed(pretrained_pos)

        return adapted

    def _interpolate_pos_embed(self, pretrained_pos: torch.Tensor) -> torch.Tensor:
        cls_pos = pretrained_pos[:, :1, :]
        patch_pos = pretrained_pos[:, 1:, :]

        n_orig = patch_pos.shape[1]
        grid_orig = int(n_orig**0.5)
        embed_dim = patch_pos.shape[2]

        patch_pos = patch_pos.reshape(1, grid_orig, grid_orig, embed_dim)
        patch_pos = patch_pos.permute(0, 3, 1, 2)

        patch_pos = F.interpolate(
            patch_pos,
            size=(self.grid_size, self.grid_size),
            mode="bicubic",
            align_corners=False,
        )

        patch_pos = patch_pos.permute(0, 2, 3, 1)
        patch_pos = patch_pos.reshape(1, self.num_patches, embed_dim)

        result: torch.Tensor = torch.cat([cls_pos, patch_pos], dim=1)

        logger.info(f"Interpolated pos embed: {grid_orig}x{grid_orig} -> {self.grid_size}x{self.grid_size}")

        return result

    def _encode_sequence(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.input_proj(x)

        if x.shape[-2] != self.input_resize or x.shape[-1] != self.input_resize:
            x = F.interpolate(
                x,
                size=(self.input_resize, self.input_resize),
                mode="bilinear",
                align_corners=False,
            )

        bsz = x.shape[0]
        x_tokens = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(bsz, -1, -1)
        x_seq = torch.cat([cls_tokens, x_tokens], dim=1)
        x_seq = x_seq + self.pos_embed
        x_seq = self.pos_drop(x_seq)

        features = self.encoder(x_seq)
        return cast(list[torch.Tensor], features)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        bsz, _, height, width = x.shape

        features = self._encode_sequence(x)

        spatial_features = []
        for feat in features:
            feat = feat[:, 1:, :]
            feat = feat.transpose(1, 2)
            feat = feat.reshape(bsz, -1, self.grid_size, self.grid_size)
            spatial_features.append(feat)

        if self.decoder_type == "linear":
            output = self.decoder(
                [f[:, 1:, :] for f in features],
                output_size=(height, width),
            )
        else:
            output = self.decoder(spatial_features, output_size=(height, width))

        result: torch.Tensor = output
        return result

    def get_encoder_params(self) -> Iterator[nn.Parameter]:
        yield from self.patch_embed.parameters()
        yield self.cls_token
        yield self.pos_embed
        yield from self.encoder.parameters()

    def get_decoder_params(self) -> Iterator[nn.Parameter]:
        yield from self.decoder.parameters()

    def _freeze_encoder(self) -> None:
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        self.cls_token.requires_grad = False
        self.pos_embed.requires_grad = False
        for param in self.encoder.parameters():
            param.requires_grad = False
        logger.info("Encoder weights frozen")

    def _freeze_encoder_layers(self, n_layers: int, mode: str = "individual") -> None:
        if n_layers <= 0:
            return

        if mode == "individual":
            max_layers = 1 + len(self.encoder.blocks)
            if n_layers > max_layers:
                logger.warning(f"freeze_encoder_layers={n_layers} exceeds max layers {max_layers}, using {max_layers}")
                n_layers = max_layers

            frozen_count = 0
            if n_layers > 0:
                for param in self.patch_embed.parameters():
                    param.requires_grad = False
                self.cls_token.requires_grad = False
                self.pos_embed.requires_grad = False
                frozen_count = 1

            for i in range(min(n_layers - 1, len(self.encoder.blocks))):
                for param in self.encoder.blocks[i].parameters():
                    param.requires_grad = False
                frozen_count += 1

        elif mode == "grouped":
            num_blocks = len(self.encoder.blocks)
            group_size = num_blocks // 2

            max_layers = 3
            if n_layers > max_layers:
                logger.warning(f"freeze_encoder_layers={n_layers} exceeds max groups {max_layers}, using {max_layers}")
                n_layers = max_layers

            frozen_count = 0
            if n_layers > 0:
                for param in self.patch_embed.parameters():
                    param.requires_grad = False
                self.cls_token.requires_grad = False
                self.pos_embed.requires_grad = False
                frozen_count = 1

            for group_idx in range(1, n_layers):
                start_block = (group_idx - 1) * group_size
                end_block = min(group_idx * group_size, num_blocks)
                for block_idx in range(start_block, end_block):
                    for param in self.encoder.blocks[block_idx].parameters():
                        param.requires_grad = False
                frozen_count += 1

            logger.info(f"Froze {frozen_count} encoder groups ({n_layers} requested, {mode} mode)")
            return
        else:
            logger.warning(f"Unknown freeze_encoder_layers_mode '{mode}', using 'individual'")
            self._freeze_encoder_layers(n_layers, mode="individual")
            return

        logger.info(f"Froze first {frozen_count} encoder layers (out of {max_layers})")

    def get_frozen_layer_info(self) -> dict[str, int | list[str]]:
        layer_groups: list[tuple[str, list[nn.Module]]] = [
            ("patch_embed", [cast(nn.Module, self.patch_embed)]),
        ]
        for i, block in enumerate(self.encoder.blocks):
            layer_groups.append((f"block_{i}", [block]))

        frozen_layers = []
        for layer_name, modules in layer_groups:
            is_frozen = all(not param.requires_grad for module in modules for param in module.parameters())
            if is_frozen:
                frozen_layers.append(layer_name)

        return {
            "total_layers": len(layer_groups),
            "frozen_layers": len(frozen_layers),
            "frozen_layer_names": frozen_layers,
        }

    def get_param_groups(self, encoder_lr: float, decoder_lr: float) -> list[dict[str, Any]]:
        return [
            {"params": list(self.get_encoder_params()), "lr": encoder_lr},
            {"params": list(self.get_decoder_params()), "lr": decoder_lr},
        ]
