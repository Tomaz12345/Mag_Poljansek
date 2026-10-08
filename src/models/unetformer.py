# Implemented from: Wang et al., "UNetFormer: A UNet-like transformer for efficient semantic
#   segmentation of remote sensing urban scene imagery", ISPRS J. Photogramm. Remote Sens. 190, 2022
# Reference implementation: https://github.com/WangLibo1995/GeoSeg

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

import torch
import torch.nn as nn

from src.constants import AUXILIARY_HEAD_DEFAULTS

from .input_projection import InputProjection
from .unetformer_utils import (
    ResNet18Encoder,
    UNetFormerDecoder,
)

logger = logging.getLogger(__name__)


# Model variant configurations
UNETFORMER_VARIANTS: dict[str, dict[str, Any]] = {
    "unetformer_tiny": {
        "encoder_channels": (64, 128, 256, 512),
        "decoder_channels": 32,
        "num_heads": 4,
        "window_size": 8,
        "mlp_ratio": 2.0,
    },
    "unetformer_small": {
        "encoder_channels": (64, 128, 256, 512),
        "decoder_channels": 64,
        "num_heads": 8,
        "window_size": 8,
        "mlp_ratio": 4.0,
    },
    "unetformer_base": {
        "encoder_channels": (64, 128, 256, 512),
        "decoder_channels": 128,
        "num_heads": 8,
        "window_size": 8,
        "mlp_ratio": 4.0,
    },
    "unetformer_large": {
        "encoder_channels": (64, 128, 256, 512),
        "decoder_channels": 256,
        "num_heads": 16,
        "window_size": 8,
        "mlp_ratio": 4.0,
    },
}


class UNetFormer(nn.Module):
    """UNetFormer for semantic segmentation.

    A hybrid CNN-Transformer architecture that combines a lightweight ResNet18
    encoder with a transformer-based decoder using Global-Local Transformer
    Blocks (GLTB) for efficient semantic segmentation.

    During training, returns (main_output, aux_output) for deep supervision.
    During eval, returns main_output only.

    Args:
        n_channels: Number of input channels (e.g., 13 for S2+S1).
        n_classes: Number of output segmentation classes.
        encoder_channels: Tuple of encoder output channels per stage.
        decoder_channels: Decoder channel dimension.
        num_heads: Number of attention heads for GLTB blocks.
        window_size: Window size for local attention.
        mlp_ratio: MLP expansion ratio in GLTB.
        drop_rate: Dropout rate.
        attn_drop_rate: Attention dropout rate.
        drop_path_rate: Stochastic depth rate.
        use_pretrained: Whether to load pretrained ResNet18 encoder.
        variant: Model variant shortcut (e.g., 'unetformer_small').
                Overrides decoder_channels, num_heads, window_size, mlp_ratio.

    Example:
        >>> # UNetFormer-Small for 13-channel input
        >>> model = UNetFormer(n_channels=13, n_classes=10, variant='unetformer_small')
        >>> model.eval()
        >>> x = torch.randn(2, 13, 256, 256)
        >>> out = model(x)  # (2, 10, 256, 256)
        >>>
        >>> # During training, returns (main_output, aux_output)
        >>> model.train()
        >>> out, aux = model(x)  # both (2, 10, 256, 256)
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        encoder_channels: tuple[int, ...],
        decoder_channels: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float,
        drop_rate: float,
        attn_drop_rate: float,
        drop_path_rate: float,
        use_pretrained: bool,
        variant: str | None,
        freeze_encoder: bool,
        freeze_encoder_layers: int,
        input_projection_type: str = "linear",
        input_projection_hidden: int = 32,
        input_projection_dims: str | list[int] | None = None,
    ) -> None:
        super().__init__()

        # Apply variant configuration if specified
        if variant is not None and variant in UNETFORMER_VARIANTS:
            config = UNETFORMER_VARIANTS[variant]
            encoder_channels = config["encoder_channels"]
            decoder_channels = config["decoder_channels"]
            num_heads = config["num_heads"]
            window_size = config["window_size"]
            mlp_ratio = config["mlp_ratio"]
            logger.info(f"Using UNetFormer variant '{variant}': {config}")
        elif variant is not None:
            valid = ", ".join(UNETFORMER_VARIANTS.keys())
            msg = f"Unknown UNetFormer variant '{variant}'. Must be one of: {valid}"
            raise ValueError(msg)

        # Validate inputs
        self._validate_inputs(n_channels, n_classes, num_heads, window_size)

        # Store configuration
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.use_pretrained = use_pretrained

        # Auxiliary loss weight (α = 0.4 per paper Section 3.3)
        self.auxiliary_loss_weight = AUXILIARY_HEAD_DEFAULTS["auxiliary_loss_weight"]

        # Input projection: n_channels -> 3 (backbone always receives 3-channel input)
        self.input_proj = InputProjection(
            n_channels=n_channels,
            target_channels=3,
            projection_type=input_projection_type,
            hidden_channels=input_projection_hidden,
            dims=input_projection_dims,
        )

        # Build encoder
        self.encoder = ResNet18Encoder(
            in_channels=3,
            pretrained=use_pretrained,
            encoder_channels=encoder_channels,
        )

        # Build decoder
        self.decoder = UNetFormerDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            num_classes=n_classes,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
        )

        # Freeze encoder if requested (freeze_encoder=True takes precedence)
        if freeze_encoder:
            self._freeze_encoder()
        elif freeze_encoder_layers > 0:
            self._freeze_encoder_layers(freeze_encoder_layers)

        logger.info(
            f"UNetFormer initialized: n_channels={n_channels}, n_classes={n_classes}, "
            f"decoder_channels={decoder_channels}, num_heads={num_heads}, "
            f"window_size={window_size}, mlp_ratio={mlp_ratio}, "
            f"pretrained={use_pretrained}"
        )

    def _validate_inputs(
        self,
        n_channels: int,
        n_classes: int,
        num_heads: int,
        window_size: int,
    ) -> None:
        """Validate constructor inputs."""
        if n_channels <= 0:
            raise ValueError(f"n_channels must be positive, got {n_channels}")

        if n_classes <= 0:
            raise ValueError(f"n_classes must be positive, got {n_classes}")

        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")

        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            During training: tuple of (main_logits, aux_logits), both (B, n_classes, H, W).
            During eval: main_logits of shape (B, n_classes, H, W).
        """
        H, W = x.shape[2:]

        x = self.input_proj(x)

        # Encode
        features = self.encoder(x)

        # Decode (returns tuple during training, tensor during eval)
        result: torch.Tensor | tuple[torch.Tensor, torch.Tensor] = self.decoder(features, output_size=(H, W))
        return result

    def get_encoder_params(self) -> Iterator[nn.Parameter]:
        """Get encoder parameters (for differential learning rates)."""
        yield from self.encoder.parameters()

    def get_decoder_params(self) -> Iterator[nn.Parameter]:
        """Get decoder parameters (for differential learning rates)."""
        yield from self.decoder.parameters()

    def _freeze_encoder(self) -> None:
        """Freeze encoder parameters so they are not updated during training."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        logger.info("Encoder weights frozen")

    def _freeze_encoder_layers(self, n_layers: int) -> None:
        """Freeze first N layers of the encoder.

        For UNetFormer with ResNet18 encoder:
        - Layer 0: conv1 + bn1 (initial feature extraction)
        - Layer 1: layer1 (64 channels)
        - Layer 2: layer2 (128 channels)
        - Layer 3: layer3 (256 channels)
        - Layer 4: layer4 (512 channels, bottleneck)

        Args:
            n_layers: Number of layers to freeze (1-5)
        """
        if n_layers <= 0:
            return

        max_layers = 5
        if n_layers > max_layers:
            logger.warning(f"freeze_encoder_layers={n_layers} exceeds max layers {max_layers}, using {max_layers}")
            n_layers = max_layers

        # Get ResNet components in order
        layer_groups: list[tuple[str, list[nn.Module]]] = [
            ("conv1+bn1", [self.encoder.conv1, self.encoder.bn1]),
            ("layer1", [self.encoder.layer1]),
            ("layer2", [self.encoder.layer2]),
            ("layer3", [self.encoder.layer3]),
            ("layer4", [self.encoder.layer4]),
        ]

        frozen_count = 0
        for i in range(n_layers):
            layer_name, modules = layer_groups[i]
            for module in modules:
                for param in module.parameters():
                    param.requires_grad = False
            frozen_count += 1
            logger.debug(f"Froze encoder layer {i}: {layer_name}")

        logger.info(f"Froze first {frozen_count} encoder layers (out of {max_layers})")

    def get_frozen_layer_info(self) -> dict[str, int | list[str]]:
        """Get information about frozen layers for logging.

        Returns:
            Dictionary with total_layers, frozen_layers count,
            and list of frozen layer names
        """
        layer_groups: list[tuple[str, list[nn.Module]]] = [
            ("conv1+bn1", [self.encoder.conv1, self.encoder.bn1]),
            ("layer1", [self.encoder.layer1]),
            ("layer2", [self.encoder.layer2]),
            ("layer3", [self.encoder.layer3]),
            ("layer4", [self.encoder.layer4]),
        ]

        frozen_layers: list[str] = []
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
        """Get parameter groups with different learning rates.

        Args:
            encoder_lr: Learning rate for encoder.
            decoder_lr: Learning rate for decoder.

        Returns:
            List of parameter group dicts for optimizer.
        """
        return [
            {"params": list(self.get_encoder_params()), "lr": encoder_lr},
            {"params": list(self.get_decoder_params()), "lr": decoder_lr},
        ]
