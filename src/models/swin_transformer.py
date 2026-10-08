# Implemented from: Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted
#   Windows", ICCV 2021 (arXiv:2103.14030)
# Reference implementation: https://github.com/microsoft/Swin-Transformer

from __future__ import annotations

import logging
from typing import Any, cast

import torch
import torch.nn as nn

from src.constants import (
    AUXILIARY_HEAD_DEFAULTS,
    DECODER_DEFAULTS,
    ENCODER_FREEZE_DEFAULTS,
    MODEL_USE_PRETRAIN_DEFAULTS,
    SwinTransformerArgDefaults,
)
from src.models.input_projection import InputProjection
from src.models.pretrained_loading import log_pretrained_load_result
from src.models.swin_utils import (
    AuxiliaryFCNHead,
    BasicLayer,
    FPNDecoder,
    PatchEmbed,
    PatchMerging,
    UPerNetDecoder,
)

logger = logging.getLogger(__name__)


# Model variant configurations
SWIN_VARIANTS: dict[str, dict[str, Any]] = {
    "swin_tiny": {
        "embed_dim": 96,
        "depths": (2, 2, 6, 2),
        "num_heads": (3, 6, 12, 24),
    },
    "swin_small": {
        "embed_dim": 96,
        "depths": (2, 2, 18, 2),
        "num_heads": (3, 6, 12, 24),
    },
    "swin_base": {
        "embed_dim": 128,
        "depths": (2, 2, 18, 2),
        "num_heads": (4, 8, 16, 32),
    },
    "swin_large": {
        "embed_dim": 192,
        "depths": (2, 2, 18, 2),
        "num_heads": (6, 12, 24, 48),
    },
}


class SwinTransformer(nn.Module):
    """Swin Transformer for semantic segmentation.

    A hierarchical Vision Transformer using shifted windows for efficient
    self-attention computation. Suitable for dense prediction tasks like
    semantic segmentation.

    Args:
        n_channels: Number of input channels (e.g., 13 for multi-spectral).
        n_classes: Number of output segmentation classes.
        img_size: Input image size (assumes square images).
        patch_size: Patch size for patch embedding.
        embed_dim: Patch embedding dimension (96 for Tiny, 128 for Base).
        depths: Number of Swin Transformer blocks in each stage.
        num_heads: Number of attention heads in each stage.
        window_size: Window size for local attention.
        mlp_ratio: Ratio of MLP hidden dimension.
        qkv_bias: If True, add learnable bias to Q, K, V projections.
        drop_rate: Dropout rate.
        attn_drop_rate: Attention dropout rate.
        drop_path_rate: Stochastic depth rate.
        norm_layer: Normalization layer.
        decoder_type: Decoder type ('upernet' or 'fpn').
        decoder_channels: Decoder channel dimension.
        decoder_dropout: Decoder dropout rate.
        decoder_align_corners: Whether to align corners in decoder interpolation.
        use_auxiliary_head: Whether to use an auxiliary FCN head for training.
        auxiliary_loss_weight: Loss weight for the auxiliary head output.
        auxiliary_head_channels: Number of intermediate channels in auxiliary head.
        auxiliary_head_index: Encoder stage index to attach auxiliary head to.
        use_pretrained: Whether to load pretrained weights.
        pretrained_type: Pretrained weights type ('imagenet_1k' or 'imagenet_22k').
        variant: Model variant shortcut (e.g., 'swin_tiny'). Overrides
                 embed_dim, depths, and num_heads if specified.

    Example:
        >>> # Swin-Tiny for 13-channel input
        >>> model = SwinTransformer(
        ...     n_channels=13, n_classes=10,
        ...     embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24)
        ... )
        >>> x = torch.randn(2, 13, 256, 256)
        >>> out = model(x)  # (2, 10, 256, 256)
        >>>
        >>> # Using variant shortcut
        >>> model = SwinTransformer(n_channels=13, n_classes=10, variant='swin_tiny')
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        img_size: int,
        patch_size: int,
        embed_dim: int,
        depths: tuple[int, ...] | list[int],
        num_heads: tuple[int, ...] | list[int],
        window_size: int,
        mlp_ratio: float,
        qkv_bias: bool,
        drop_rate: float,
        attn_drop_rate: float,
        drop_path_rate: float,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        decoder_type: str = DECODER_DEFAULTS["decoder_type"],
        decoder_channels: int = DECODER_DEFAULTS["decoder_channels"],
        decoder_dropout: float = DECODER_DEFAULTS["decoder_dropout"],
        decoder_align_corners: bool = DECODER_DEFAULTS["decoder_align_corners"],
        use_auxiliary_head: bool = AUXILIARY_HEAD_DEFAULTS["use_auxiliary_head"],
        auxiliary_loss_weight: float = AUXILIARY_HEAD_DEFAULTS["auxiliary_loss_weight"],
        auxiliary_head_channels: int = AUXILIARY_HEAD_DEFAULTS["auxiliary_head_channels"],
        auxiliary_head_index: int = AUXILIARY_HEAD_DEFAULTS["auxiliary_head_index"],
        use_pretrained: bool = MODEL_USE_PRETRAIN_DEFAULTS["use_pretrained"],
        pretrained_type: str = SwinTransformerArgDefaults["pretrained_type"],
        variant: str | None = None,
        freeze_encoder: bool = ENCODER_FREEZE_DEFAULTS["freeze_encoder"],
        freeze_encoder_layers: int = ENCODER_FREEZE_DEFAULTS["freeze_encoder_layers"],
        freeze_encoder_layers_mode: str = ENCODER_FREEZE_DEFAULTS["freeze_encoder_layers_mode"],
        input_projection_type: str = "linear",
        input_projection_hidden: int = 32,
        input_projection_dims: str | list[int] | None = None,
    ) -> None:
        super().__init__()

        # If variant specified, override architecture parameters
        if variant is not None:
            config = SWIN_VARIANTS[variant]
            embed_dim = config["embed_dim"]
            depths = config["depths"]
            num_heads = config["num_heads"]

        # Validate inputs
        self._validate_inputs(n_channels, n_classes, depths, num_heads, window_size)

        # Store configuration
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depths = list(depths)
        self.num_heads = list(num_heads)
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.num_stages = len(depths)

        # Input projection: n_channels -> 3 (backbone always receives 3-channel input)
        self.input_proj = InputProjection(
            n_channels=n_channels,
            target_channels=3,
            projection_type=input_projection_type,
            hidden_channels=input_projection_hidden,
            dims=input_projection_dims,
        )

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=3,
            embed_dim=embed_dim,
            norm_layer=norm_layer,
        )

        # Position dropout
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth decay rule
        total_depth = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]

        # Build encoder stages
        self.stages = nn.ModuleList()
        self.out_channels: list[int] = []

        patches_resolution = img_size // patch_size

        for i_stage in range(self.num_stages):
            dim = int(embed_dim * 2**i_stage)
            self.out_channels.append(dim)

            stage_resolution = patches_resolution // (2**i_stage)
            input_resolution = (stage_resolution, stage_resolution)

            # Get drop path rates for this stage
            stage_dpr = dpr[sum(depths[:i_stage]) : sum(depths[: i_stage + 1])]

            stage = BasicLayer(
                dim=dim,
                input_resolution=input_resolution,
                depth=depths[i_stage],
                num_heads=num_heads[i_stage],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=stage_dpr,
                norm_layer=norm_layer,
                downsample=PatchMerging if i_stage < self.num_stages - 1 else None,
            )
            self.stages.append(stage)

        # Layer norm for each stage output (for decoder)
        self.norm_layers = nn.ModuleList()
        for i in range(self.num_stages):
            dim = int(embed_dim * 2**i)
            self.norm_layers.append(norm_layer(dim))

        # Build decoder
        self.decoder: UPerNetDecoder | FPNDecoder
        if decoder_type == "upernet":
            self.decoder = UPerNetDecoder(
                in_channels=self.out_channels,
                channel=decoder_channels,
                num_classes=n_classes,
                dropout=decoder_dropout,
                align_corners=decoder_align_corners,
            )
        elif decoder_type == "fpn":
            self.decoder = FPNDecoder(
                in_channels=self.out_channels,
                out_channels=decoder_channels,
                num_classes=n_classes,
                dropout=decoder_dropout,
                align_corners=decoder_align_corners,
            )
        else:
            msg = f"Unknown decoder_type '{decoder_type}'. Must be 'upernet' or 'fpn'"
            raise ValueError(msg)

        self.decoder_type = decoder_type

        # Build auxiliary head (for intermediate supervision during training)
        self.use_auxiliary_head = use_auxiliary_head
        self.auxiliary_loss_weight = auxiliary_loss_weight
        self.auxiliary_head_index = auxiliary_head_index
        self.auxiliary_head: AuxiliaryFCNHead | None = None

        if use_auxiliary_head:
            if auxiliary_head_index >= self.num_stages:
                msg = f"auxiliary_head_index ({auxiliary_head_index}) must be < num_stages ({self.num_stages})"
                raise ValueError(msg)
            aux_in_channels = self.out_channels[auxiliary_head_index]
            self.auxiliary_head = AuxiliaryFCNHead(
                in_channels=aux_in_channels,
                channels=auxiliary_head_channels,
                num_classes=n_classes,
                dropout=decoder_dropout,
                align_corners=decoder_align_corners,
            )

        # Initialize weights
        self._init_weights()

        # Load pretrained weights if requested
        if use_pretrained:
            pretrained_variant = variant or "swin_tiny"
            self._load_pretrained(pretrained_variant, pretrained_type)

        # Freeze encoder if requested (freeze_encoder=True takes precedence)
        if freeze_encoder:
            self._freeze_encoder()
        elif freeze_encoder_layers > 0:
            self._freeze_encoder_layers(freeze_encoder_layers, mode=freeze_encoder_layers_mode)

        aux_info = ""
        if use_auxiliary_head:
            aux_info = f", auxiliary_head=stage_{auxiliary_head_index}(weight={auxiliary_loss_weight})"
        logger.info(
            f"SwinTransformer initialized: n_channels={n_channels}, n_classes={n_classes}, "
            f"embed_dim={embed_dim}, depths={depths}, num_heads={num_heads}, "
            f"window_size={window_size}, decoder={decoder_type}{aux_info}"
        )

    def _validate_inputs(
        self,
        n_channels: int,
        n_classes: int,
        depths: tuple[int, ...] | list[int],
        num_heads: tuple[int, ...] | list[int],
        window_size: int,
    ) -> None:
        """Validate constructor inputs."""
        if n_channels <= 0:
            raise ValueError(f"n_channels must be positive, got {n_channels}")

        if n_classes <= 0:
            raise ValueError(f"n_classes must be positive, got {n_classes}")

        if len(depths) != len(num_heads):
            raise ValueError(f"depths and num_heads must have same length, got {len(depths)} and {len(num_heads)}")

        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")

    def _init_weights(self) -> None:
        """Initialize weights using truncated normal distribution."""

        def _init_fn(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

        self.apply(_init_fn)

    def _load_pretrained(self, variant: str, pretrained_type: str) -> None:
        """Load pretrained weights.

        Args:
            variant: Model variant name.
            pretrained_type: Type of pretrained weights.
        """
        try:
            import timm

            # Map variant to timm model name
            timm_names = {
                "swin_tiny": "swin_tiny_patch4_window7_224",
                "swin_small": "swin_small_patch4_window7_224",
                "swin_base": "swin_base_patch4_window7_224",
                "swin_large": "swin_large_patch4_window7_224",
            }

            timm_name = timm_names.get(variant)
            if timm_name is None:
                logger.warning(f"No pretrained weights available for variant: {variant}")
                return

            if pretrained_type == "imagenet_22k":
                timm_name += "_in22k"

            # Load pretrained model
            pretrained_model = timm.create_model(timm_name, pretrained=True)
            state_dict = pretrained_model.state_dict()

            # Remap timm keys to our model's key names and filter incompatible keys.
            # Key differences vs timm:
            #   - timm 'layers.X' -> our 'stages.X'
            #   - timm places downsample at start of next stage: 'layers.X+1.downsample'
            #     maps to our 'stages.X.downsample' (end of current stage)
            remapped: dict[str, torch.Tensor] = {}
            import re

            for k, v in state_dict.items():
                if k.startswith("head.") or k.startswith("norm."):
                    # Skip classification head and final norm (not in our encoder)
                    continue
                elif k.startswith("layers."):
                    rest = k[len("layers.") :]  # e.g. "1.downsample.norm.weight"
                    m_idx = re.match(r"^(\d+)\.(downsample.*)", rest)
                    if m_idx:
                        stage_idx = int(m_idx.group(1)) - 1  # shift: layers.X+1 -> stages.X
                        if stage_idx >= 0:
                            remapped[f"stages.{stage_idx}.{m_idx.group(2)}"] = v
                    else:
                        remapped["stages." + rest] = v
                else:
                    remapped[k] = v
            state_dict = remapped

            # Load with strict=False to handle missing decoder/norm_layers/input_proj weights
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            log_pretrained_load_result(
                missing,
                unexpected,
                logger,
                model_name=f"SwinTransformer({variant})",
                expected_missing_prefixes=["decoder.", "auxiliary_head.", "input_proj."],
            )
            logger.info(f"Loaded pretrained {variant} ({pretrained_type}) weights.")

        except ImportError:
            logger.warning("timm library not installed. Skipping pretrained weight loading. Install with: pip install timm")
        except Exception as e:
            logger.warning(f"Failed to load pretrained weights: {e}")

    def forward(self, x: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            During eval or when auxiliary head is disabled:
                Segmentation logits of shape (B, n_classes, H, W).
            During training with auxiliary head enabled:
                Dict with keys 'main' and 'auxiliary', each containing
                segmentation logits of shape (B, n_classes, H, W).
        """
        input_size = (x.shape[2], x.shape[3])

        x = self.input_proj(x)

        # Patch embedding: (B, C, H, W) -> (B, embed_dim, H/patch_size, W/patch_size)
        x = self.patch_embed(x)

        # Convert to (B, H, W, C) format for Swin blocks
        x = x.permute(0, 2, 3, 1)
        x = self.pos_drop(x)

        # Collect multi-scale features from each stage
        features: list[torch.Tensor] = []

        for i, stage in enumerate(self.stages):
            # Get features before downsampling for FPN/UPerNet decoder
            result = stage(x, return_before_downsample=True)
            # Type assertion to help mypy understand tuple return
            assert isinstance(result, tuple)
            x, feat_before_ds = result

            # Apply layer norm and convert back to (B, C, H, W)
            feat = self.norm_layers[i](feat_before_ds)
            feat = feat.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
            features.append(feat)

        # Decode to segmentation map
        main_out: torch.Tensor = self.decoder(features)

        if self.training and self.auxiliary_head is not None:
            aux_out = self.auxiliary_head(features[self.auxiliary_head_index], target_size=input_size)
            return {"main": main_out, "auxiliary": aux_out}

        return main_out

    def get_encoder_params(self) -> list[nn.Parameter]:
        """Get encoder parameters for differential learning rates.

        Returns:
            List of encoder parameters (patch_embed + stages + norm_layers).
        """
        params: list[nn.Parameter] = []
        params.extend(self.patch_embed.parameters())
        for stage in self.stages:
            params.extend(stage.parameters())
        for norm in self.norm_layers:
            params.extend(norm.parameters())
        return params

    def get_decoder_params(self) -> list[nn.Parameter]:
        """Get decoder parameters for differential learning rates.

        Returns:
            List of decoder parameters (including auxiliary head if present).
        """
        params = list(self.decoder.parameters())
        if self.auxiliary_head is not None:
            params.extend(self.auxiliary_head.parameters())
        return params

    def _freeze_encoder(self) -> None:
        """Freeze encoder parameters so they are not updated during training."""
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        for stage in self.stages:
            for param in stage.parameters():
                param.requires_grad = False
        for norm in self.norm_layers:
            for param in norm.parameters():
                param.requires_grad = False
        logger.info("Encoder weights frozen")

    def _freeze_encoder_layers(self, n_layers: int, mode: str = "individual") -> None:
        """Freeze first N layers of the encoder.

        Supports two modes:
        - "individual": Treats each stage as a layer (5 layers: patch_embed + 4 stages)
        - "grouped": Groups transformer blocks within stages (useful for deep models)

        For Swin Transformer with depths=[2,2,6,2]:
        - Individual mode (5 layers):
          Layer 0: patch_embed
          Layer 1: stage 0 (2 blocks)
          Layer 2: stage 1 (2 blocks)
          Layer 3: stage 2 (6 blocks)
          Layer 4: stage 3 (2 blocks)

        Args:
            n_layers: Number of layers/groups to freeze
            mode: "individual" or "grouped"
        """
        if n_layers <= 0:
            return

        if mode == "individual":
            # 5 layers: patch_embed + 4 stages
            max_layers = 1 + len(self.stages)
            if n_layers > max_layers:
                logger.warning(f"freeze_encoder_layers={n_layers} exceeds max layers {max_layers}, using {max_layers}")
                n_layers = max_layers

            stages = self.stages
            layer_groups: list[tuple[str, list[nn.Module]]] = [
                ("patch_embed", [cast(nn.Module, self.patch_embed)]),
                *[(f"stage_{i}", [stage]) for i, stage in enumerate(stages)],
            ]

            frozen_count = 0
            for i, (layer_name, modules) in enumerate(layer_groups[:n_layers]):
                for module in modules:
                    for param in module.parameters():
                        param.requires_grad = False
                frozen_count = i + 1
                logger.debug(f"Froze encoder layer {i}: {layer_name}")

        elif mode == "grouped":
            # Group by stages but count total blocks
            # This allows freezing partial stages
            total_blocks = sum(self.depths)
            max_layers = total_blocks + 1  # +1 for patch_embed

            if n_layers > max_layers:
                logger.warning(f"freeze_encoder_layers={n_layers} exceeds max blocks {max_layers}, using {max_layers}")
                n_layers = max_layers

            # First freeze patch_embed
            if n_layers > 0:
                for param in self.patch_embed.parameters():
                    param.requires_grad = False
                logger.debug("Froze patch_embed")
                frozen_count = 1
            else:
                frozen_count = 0

            # Freeze individual blocks across stages
            blocks_to_freeze = n_layers - 1  # Remaining after patch_embed
            block_idx = 0
            for stage_idx, stage in enumerate(self.stages):
                for block in cast(nn.ModuleList, stage.blocks):
                    if blocks_to_freeze > 0:
                        for param in block.parameters():
                            param.requires_grad = False
                        blocks_to_freeze -= 1
                        block_idx += 1
                        frozen_count += 1
                        logger.debug(f"Froze block {block_idx} in stage {stage_idx}")

            logger.info(f"Froze {frozen_count} encoder components ({n_layers} requested, {mode} mode)")
            return
        else:
            logger.warning(f"Unknown freeze_encoder_layers_mode '{mode}', using 'individual'")
            self._freeze_encoder_layers(n_layers, mode="individual")
            return

        logger.info(f"Froze first {frozen_count} encoder layers (out of {max_layers})")

    def get_frozen_layer_info(self) -> dict[str, int | list[str]]:
        """Get information about frozen layers for logging.

        Returns:
            Dictionary with total_layers, frozen_layers count,
            and list of frozen layer names
        """
        stages = self.stages
        layer_groups: list[tuple[str, list[nn.Module]]] = [
            ("patch_embed", [cast(nn.Module, self.patch_embed)]),
            *[(f"stage_{i}", [stage]) for i, stage in enumerate(stages)],
        ]

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
