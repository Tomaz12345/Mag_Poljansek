# Implemented from: Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image
#   Recognition at Scale", ICLR 2021 (arXiv:2010.11929)
# Reference implementation: https://github.com/google-research/vision_transformer
# Pretrained weights are loaded through timm (https://github.com/huggingface/pytorch-image-models).

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.constants import (
    VIT_NECK_TYPES,
    VIT_PRETRAINED_TYPES,
    VIT_STEM_MODES,
    VIT_VARIANTS,
)

from .croma_utils.decoder import (
    FPNSegDecoder,
    LinearSegDecoder,
    UPerNetSegDecoder,
)
from .input_projection import InputProjection
from .vit_utils import ViTEncoder, ViTMultiScaleNeck, ViTPatchEmbed, load_vit_pretrained_weights

logger = logging.getLogger(__name__)

DECODER_TYPES: tuple[str, ...] = ("upernet", "fpn", "linear")

# Architecture table. ViT-Base / ViT-Large are Table 1 of Dosovitskiy et al.;
# ViT-Tiny / ViT-Small come from the AugReg study (Steiner et al., 2022).
# MLP size is 4 * embed_dim throughout, i.e. mlp_ratio 4.0.
VIT_VARIANT_CONFIGS: dict[str, dict[str, Any]] = {
    "vit_tiny": {"embed_dim": 192, "depth": 12, "num_heads": 3, "patch_size": 16},
    "vit_small": {"embed_dim": 384, "depth": 12, "num_heads": 6, "patch_size": 16},
    "vit_base": {"embed_dim": 768, "depth": 12, "num_heads": 12, "patch_size": 16},
    "vit_large": {"embed_dim": 1024, "depth": 24, "num_heads": 16, "patch_size": 16},
}

# Channels the ImageNet patch embedding was trained on.
_PRETRAINED_STEM_CHANNELS: int = 3


class ViT(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        img_size: int,
        patch_size: int,
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
        pretrained_type: str,
        pretrained_path: str,
        vit_stem_mode: str,
        vit_neck: str,
        freeze_encoder: bool,
        freeze_encoder_layers: int,
        freeze_encoder_layers_mode: str,
        variant: str | None,
        input_projection_type: str = "linear",
        input_projection_hidden: int = 32,
        input_projection_dims: str | list[int] | None = None,
    ) -> None:
        super().__init__()

        if variant is not None and variant not in VIT_VARIANTS:
            raise ValueError(f"Unknown ViT variant: {variant!r}. Choose from {VIT_VARIANTS}.")
        if pretrained_type not in VIT_PRETRAINED_TYPES:
            raise ValueError(f"Unknown pretrained_type: {pretrained_type!r}. Choose from {VIT_PRETRAINED_TYPES}.")
        if vit_stem_mode not in VIT_STEM_MODES:
            raise ValueError(f"Unknown vit_stem_mode: {vit_stem_mode!r}. Choose from {VIT_STEM_MODES}.")
        if vit_neck not in VIT_NECK_TYPES:
            raise ValueError(f"Unknown vit_neck: {vit_neck!r}. Choose from {VIT_NECK_TYPES}.")
        if decoder_type not in DECODER_TYPES:
            raise ValueError(f"Unknown decoder_type: {decoder_type!r}. Choose from {DECODER_TYPES}.")

        if variant is not None:
            config = VIT_VARIANT_CONFIGS[variant]
            embed_dim = config["embed_dim"]
            depth = config["depth"]
            num_heads = config["num_heads"]
            patch_size = config["patch_size"]
            logger.info(f"Using ViT variant '{variant}': {config}")

        if img_size % patch_size != 0:
            raise ValueError(f"img_size ({img_size}) must be divisible by patch_size ({patch_size}).")

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.variant = variant
        self.decoder_type = decoder_type
        self.vit_stem_mode = vit_stem_mode
        self.vit_neck_type = vit_neck

        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size**2

        # Stem: either project down to 3 channels, or widen the patch embedding.
        stem_channels = n_channels if vit_stem_mode == "inflate" else _PRETRAINED_STEM_CHANNELS
        self.input_proj = InputProjection(
            n_channels=n_channels,
            target_channels=stem_channels,
            projection_type=input_projection_type,
            hidden_channels=input_projection_hidden,
            dims=input_projection_dims,
        )
        self.stem_channels = stem_channels

        self.patch_embed = ViTPatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=stem_channels,
            embed_dim=embed_dim,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        out_indices = tuple(max(0, depth // 4 * (i + 1) - 1) for i in range(4))

        self.encoder = ViTEncoder(
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

        # The neck only makes sense for the pyramid decoders; the linear decoder
        # reads features[-1] alone, so building one would just waste parameters.
        self.neck: ViTMultiScaleNeck | None = None
        if vit_neck == "multiscale" and decoder_type != "linear":
            self.neck = ViTMultiScaleNeck(embed_dim=embed_dim)

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
        else:
            self.decoder = LinearSegDecoder(
                in_channels=embed_dim,
                num_classes=n_classes,
                dropout_rate=decoder_dropout,
            )

        self._init_weights()

        if use_pretrained:
            self._load_pretrained(pretrained_type, pretrained_path)

        if freeze_encoder:
            self._freeze_encoder()
        elif freeze_encoder_layers > 0:
            self._freeze_encoder_layers(freeze_encoder_layers, mode=freeze_encoder_layers_mode)

        logger.info(
            f"ViT initialized: variant={variant}, n_channels={n_channels}, n_classes={n_classes}, "
            f"embed_dim={embed_dim}, depth={depth}, num_heads={num_heads}, patch_size={patch_size}, "
            f"grid={self.grid_size}x{self.grid_size}, stem={vit_stem_mode}({stem_channels}ch), "
            f"neck={vit_neck if self.neck is not None else 'none'}, decoder={decoder_type}"
        )

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _load_pretrained(self, pretrained_type: str, pretrained_path: str) -> None:
        if self.variant is None:
            logger.warning("Cannot determine ViT pretrained weights without a variant. Skipping pretrained weight loading.")
            return

        load_vit_pretrained_weights(
            self,
            variant=self.variant,
            pretrained_type=pretrained_type,
            pretrained_path=pretrained_path,
            grid_size=self.grid_size,
            in_channels=self.stem_channels,
            expected_missing_prefixes=["decoder.", "neck.", "input_proj."],
        )

    def _encode_sequence(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.input_proj(x)

        if x.shape[-2] != self.img_size or x.shape[-1] != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)

        bsz = x.shape[0]
        x_tokens = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(bsz, -1, -1)
        x_seq = torch.cat([cls_tokens, x_tokens], dim=1)
        x_seq = x_seq + self.pos_embed
        x_seq = self.pos_drop(x_seq)

        features = self.encoder(x_seq)
        return cast(list[torch.Tensor], features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, _, height, width = x.shape

        features = self._encode_sequence(x)

        if self.decoder_type == "linear":
            output = self.decoder([f[:, 1:, :] for f in features], output_size=(height, width))
            return cast(torch.Tensor, output)

        spatial_features = []
        for feat in features:
            feat = feat[:, 1:, :].transpose(1, 2).reshape(bsz, self.embed_dim, self.grid_size, self.grid_size)
            spatial_features.append(feat)

        if self.neck is not None:
            spatial_features = self.neck(spatial_features)

        output = self.decoder(spatial_features, output_size=(height, width))
        return cast(torch.Tensor, output)

    def get_encoder_params(self) -> Iterator[nn.Parameter]:
        yield from self.patch_embed.parameters()
        yield self.cls_token
        yield self.pos_embed
        yield from self.encoder.parameters()

    def get_decoder_params(self) -> Iterator[nn.Parameter]:
        yield from self.input_proj.parameters()
        if self.neck is not None:
            yield from self.neck.parameters()
        yield from self.decoder.parameters()

    def _freeze_stem(self) -> None:
        """Freeze the patch embedding and the CLS/positional parameters."""
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        self.cls_token.requires_grad = False
        self.pos_embed.requires_grad = False

    def _freeze_encoder(self) -> None:
        self._freeze_stem()
        for param in self.encoder.parameters():
            param.requires_grad = False
        logger.info("Encoder weights frozen")

    def _freeze_encoder_layers(self, n_layers: int, mode: str = "individual") -> None:
        if n_layers <= 0:
            return

        if mode == "grouped":
            num_blocks = len(self.encoder.blocks)
            group_size = max(1, num_blocks // 2)

            max_layers = 3
            if n_layers > max_layers:
                logger.warning(f"freeze_encoder_layers={n_layers} exceeds max groups {max_layers}, using {max_layers}")
                n_layers = max_layers

            self._freeze_stem()
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

        if mode != "individual":
            logger.warning(f"Unknown freeze_encoder_layers_mode '{mode}', using 'individual'")

        max_layers = 1 + len(self.encoder.blocks)
        if n_layers > max_layers:
            logger.warning(f"freeze_encoder_layers={n_layers} exceeds max layers {max_layers}, using {max_layers}")
            n_layers = max_layers

        self._freeze_stem()
        frozen_count = 1

        for i in range(min(n_layers - 1, len(self.encoder.blocks))):
            for param in self.encoder.blocks[i].parameters():
                param.requires_grad = False
            frozen_count += 1

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
