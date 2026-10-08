# Implemented from: Wolf et al., "Brewing Stronger Features: Dual-Teacher Distillation for
#   Multispectral Earth Observation", CVPR 2026 (arXiv:2602.19863)
# Reference implementation: https://github.com/wolfilip/DEO-FM

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from typing import cast

import torch
import torch.nn as nn

from .croma_utils.decoder import (
    FPNSegDecoder,
    LinearSegDecoder,
    UPerNetSegDecoder,
)
from .deo_fm_utils import (
    DEOPatchEmbed,
    DEOSwinBackbone,
    DEOVisionTransformer,
    load_deo_pretrained_weights,
    vit_base,
)
from .input_projection import InputProjection

logger = logging.getLogger(__name__)


DEO_FM_VARIANTS: tuple[str, ...] = ("swin_b", "vit_base")

# Swin-B: 4 stage outputs at 1/4, 1/8, 1/16, 1/32 of input.
_SWIN_CHANNELS: tuple[int, int, int, int] = (128, 256, 512, 1024)

# ViT-B at patch_size=8, img_size=256: 4 evenly-spaced intermediate blocks.
_VIT_INTERMEDIATE_LAYERS: tuple[int, int, int, int] = (3, 5, 8, 11)

# HuggingFace location for the public Swin-B checkpoint. ViT-B is only
# distributed via OneDrive — users must supply a local path for that one.
_HF_REPO_ID = "SolaireTheSun/DEO"
_HF_FILENAMES: dict[str, str] = {"swin_b": "DEO_swin_b.pth"}


class DEOFM(nn.Module):
    """DEO-FM segmentation model: pretrained dual-stem trunk + decoder."""

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        img_size: int = 256,
        patch_size: int = 8,
        decoder_type: str = "upernet",
        decoder_channels: int = 512,
        decoder_dropout: float = 0.1,
        use_pretrained: bool = True,
        pretrained_path: str = _HF_REPO_ID,
        variant: str | None = "swin_b",
        deo_input_mode: str = "auto",
        freeze_encoder: bool = False,
        freeze_encoder_layers: int = 0,
        freeze_encoder_layers_mode: str = "first",
        input_projection_type: str = "linear",
        input_projection_hidden: int = 32,
        input_projection_dims: str | list[int] | None = None,
    ) -> None:
        super().__init__()

        if variant not in DEO_FM_VARIANTS:
            raise ValueError(f"Unknown DEO-FM variant: {variant!r}. Choose from {DEO_FM_VARIANTS}.")
        if deo_input_mode not in ("auto", "rgb", "ms"):
            raise ValueError(f"Unknown deo_input_mode: {deo_input_mode!r}. Choose 'auto', 'rgb', or 'ms'.")

        self.variant = variant
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.img_size = img_size
        self.decoder_type = decoder_type

        # Decide which stem to feed and whether an InputProjection is needed.
        self.use_ms, target_channels = self._decide_routing(n_channels, deo_input_mode)
        self.input_proj = InputProjection(
            n_channels=n_channels,
            target_channels=target_channels,
            projection_type=input_projection_type,
            hidden_channels=input_projection_hidden,
            dims=input_projection_dims,
        )

        # Build trunk.
        if variant == "swin_b":
            self.swin: DEOSwinBackbone | None = DEOSwinBackbone()
            self.vit: DEOVisionTransformer | None = None
            feature_dims: list[int] = list(_SWIN_CHANNELS)
        else:  # vit_base
            self.swin = None
            self.vit = vit_base(patch_size=patch_size, img_size=img_size)
            # Attach the two parallel patch embeddings under the ViT trunk so
            # the released DEO checkpoint's top-level keys
            # (``patch_embed_rgb.proj.weight`` / ``patch_embed_ms.proj.weight``)
            # line up when calling ``self.vit.load_state_dict``.
            self.vit.add_module(
                "patch_embed_rgb",
                DEOPatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=self.vit.embed_dim),
            )
            self.vit.add_module(
                "patch_embed_ms",
                DEOPatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=10, embed_dim=self.vit.embed_dim),
            )
            feature_dims = [self.vit.embed_dim] * 4

        # Build decoder.
        self.decoder: UPerNetSegDecoder | FPNSegDecoder | LinearSegDecoder
        if decoder_type == "upernet":
            self.decoder = UPerNetSegDecoder(
                in_channels=feature_dims,
                channels=decoder_channels,
                num_classes=n_classes,
                dropout_rate=decoder_dropout,
            )
        elif decoder_type == "fpn":
            self.decoder = FPNSegDecoder(
                in_channels=feature_dims,
                channels=decoder_channels,
                num_classes=n_classes,
                dropout_rate=decoder_dropout,
            )
        elif decoder_type == "linear":
            self.decoder = LinearSegDecoder(
                in_channels=feature_dims[-1],
                num_classes=n_classes,
                dropout_rate=decoder_dropout,
            )
        else:
            raise ValueError(f"Unknown decoder_type: {decoder_type!r}")

        if use_pretrained:
            self._load_pretrained(pretrained_path)

        if freeze_encoder:
            self._freeze_encoder()
        elif freeze_encoder_layers > 0:
            self._freeze_encoder_layers(freeze_encoder_layers, mode=freeze_encoder_layers_mode)

        logger.info(
            f"DEO-FM initialised: variant={variant}, n_channels={n_channels}, "
            f"n_classes={n_classes}, route={'ms' if self.use_ms else 'rgb'} "
            f"(target_channels={target_channels}), decoder={decoder_type}"
        )

    @staticmethod
    def _decide_routing(n_channels: int, deo_input_mode: str) -> tuple[bool, int]:
        if deo_input_mode == "rgb":
            return False, 3
        if deo_input_mode == "ms":
            return True, 10
        # auto
        if n_channels == 3:
            return False, 3
        if n_channels == 10:
            return True, 10
        logger.warning(
            f"DEO-FM: n_channels={n_channels} does not match either DEO stem (3 or 10). "
            "Projecting to 10 channels and routing through the multispectral stem; "
            "this deviates from the pretraining distribution."
        )
        return True, 10

    def _load_pretrained(self, pretrained_path: str) -> None:
        local_path = self._resolve_weights_path(pretrained_path)
        if local_path is None:
            return

        if self.variant == "swin_b":
            assert self.swin is not None
            load_deo_pretrained_weights(
                self.swin.backbone,
                local_path,
                model_name="DEO-FM (swin_b)",
                # `head.*` is the torchvision Swin's ImageNet classifier
                # head, which we replace with our own decoder.
                expected_missing_prefixes=["decoder.", "head."],
            )
        else:
            assert self.vit is not None
            load_deo_pretrained_weights(
                self.vit,
                local_path,
                model_name="DEO-FM (vit_base)",
                expected_missing_prefixes=["decoder."],
            )

    def _resolve_weights_path(self, pretrained_path: str) -> str | None:
        """Return a local .pth path or ``None`` if weights can't be obtained."""
        if os.path.isfile(pretrained_path):
            return pretrained_path

        if self.variant not in _HF_FILENAMES:
            logger.warning(
                f"DEO-FM: variant '{self.variant}' is not published on HuggingFace. "
                f"Download the checkpoint from "
                "https://unilj-my.sharepoint.com/:f:/g/personal/filip_wolf_fri1_uni-lj_si/IgCsJwuKs8-PQIp706uUiPpsAVkaNsVH5pPLAU_6mYkF22Q "
                f"and set pretrained_path to the local file."
            )
            return None

        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            logger.warning("huggingface_hub not installed; install with `pip install huggingface_hub`.")
            return None

        filename = _HF_FILENAMES[self.variant]
        logger.info(f"Fetching DEO-FM weights from HuggingFace: {pretrained_path}/{filename}")
        try:
            local: str = hf_hub_download(repo_id=pretrained_path, filename=filename)
            return local
        except Exception as exc:
            logger.warning(f"Failed to download DEO-FM weights from HuggingFace: {exc}")
            return None

    def _forward_trunk(self, x: torch.Tensor) -> list[torch.Tensor]:
        if self.variant == "swin_b":
            assert self.swin is not None
            features: list[torch.Tensor] = self.swin(x, use_ms=self.use_ms)
            return features

        # ViT path
        assert self.vit is not None
        b, _, w, h = x.shape
        patch_embed = cast(nn.Module, self.vit.patch_embed_ms if self.use_ms else self.vit.patch_embed_rgb)
        tokens = patch_embed(x)
        cls_tokens = self.vit.cls_token.expand(b, -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)
        tokens = tokens + self.vit.interpolate_pos_encoding(tokens, w, h)
        tokens = self.vit.pos_drop(tokens)

        sequences = self.vit.get_intermediate_layers(tokens, _VIT_INTERMEDIATE_LAYERS)

        # Convert token sequences (B, 1+N, D) to spatial maps (B, D, H', W').
        grid = w // self.vit.patch_size
        spatial: list[torch.Tensor] = []
        for seq in sequences:
            feat = seq[:, 1:, :].transpose(1, 2).reshape(b, self.vit.embed_dim, grid, grid)
            spatial.append(feat)
        return spatial

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        x = self.input_proj(x)
        features = self._forward_trunk(x)
        output = self.decoder(features, output_size=(h, w))
        return cast(torch.Tensor, output)

    def get_encoder_params(self) -> Iterator[nn.Parameter]:
        yield from self.input_proj.parameters()
        if self.swin is not None:
            yield from self.swin.parameters()
        if self.vit is not None:
            yield from self.vit.parameters()

    def get_decoder_params(self) -> Iterator[nn.Parameter]:
        yield from self.decoder.parameters()

    def _freeze_encoder(self) -> None:
        for p in self.get_encoder_params():
            p.requires_grad = False
        logger.info("DEO-FM: encoder fully frozen.")

    def _freeze_encoder_layers(self, num_layers: int, *, mode: str = "first") -> None:
        """Freeze the first/last ``num_layers`` transformer blocks (ViT) or
        Swin stages (Swin-B). The dual stems and positional/cls params are
        always frozen alongside the lower blocks when ``mode == "first"``.
        """
        if self.variant == "swin_b":
            assert self.swin is not None
            # torchvision swin features are: [stage1_blocks, merge, stage2_blocks,
            # merge, stage3_blocks, merge, stage4_blocks] after we removed the
            # original stem. Indices [0, 2, 4, 6] are the block groups.
            block_group_indices = [0, 2, 4, 6]
            target = block_group_indices[:num_layers] if mode == "first" else block_group_indices[-num_layers:]
            if mode == "first":
                for p in self.swin.backbone.conv_ms.parameters():
                    p.requires_grad = False
                for p in self.swin.backbone.conv_rgb.parameters():
                    p.requires_grad = False
            for i in target:
                for p in self.swin.backbone.features[i].parameters():
                    p.requires_grad = False
            logger.info(f"DEO-FM (swin_b): froze block groups {target} ({mode}).")
        else:
            assert self.vit is not None
            blocks = list(self.vit.blocks)
            target_blocks = blocks[:num_layers] if mode == "first" else blocks[-num_layers:]
            if mode == "first":
                for p in cast(nn.Module, self.vit.patch_embed_rgb).parameters():
                    p.requires_grad = False
                for p in cast(nn.Module, self.vit.patch_embed_ms).parameters():
                    p.requires_grad = False
                self.vit.cls_token.requires_grad = False
                self.vit.pos_embed.requires_grad = False
            for blk in target_blocks:
                for p in blk.parameters():
                    p.requires_grad = False
            logger.info(f"DEO-FM (vit_base): froze {len(target_blocks)} blocks ({mode}).")
