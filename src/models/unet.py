# author: Alexandre Milesi (https://github.com/milesial/Pytorch-UNet)
# milesial/Pytorch-UNet is licensed under the GNU General Public License v3.0
# Extended with pretrained encoder support and flexible channel configuration

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet34_Weights

from src.messages import MODEL_INIT_ERRORS
from src.models.input_projection import InputProjection
from src.models.unet_utils.unet_parts import DoubleConv, Down, OutConv, Up
from src.utils import log_and_raise

logger = logging.getLogger(__name__)


class UNet(nn.Module):
    """U-Net model with optional pretrained encoder and flexible channel configuration.

    The U-Net architecture consists of:
    1. Encoder (contracting path): Captures context through downsampling
    2. Bottleneck: Bridge between encoder and decoder
    3. Decoder (expansive path): Enables precise localization through upsampling
    4. Skip connections: Concatenate encoder features with decoder features

    Args:
        n_channels: Number of input channels.
        n_classes: Number of output classes.
        bilinear: If True, use bilinear upsampling; else use transposed convolutions.
        channels: Tuple of channel sizes for each encoder level.
            Length determines network depth. Default: (64, 128, 256, 512, 1024).
        use_pretrained: If True, use pretrained encoder backbone.
        pretrained_encoder: Encoder architecture when use_pretrained=True.
            Options: 'resnet18', 'resnet34'. Default: 'resnet18'.

    Example:
        >>> # Default 5-level UNet
        >>> model = UNet(n_channels=13, n_classes=10)
        >>>
        >>> # Custom 4-level UNet with smaller channels
        >>> model = UNet(n_channels=13, n_classes=10, channels=(32, 64, 128, 256))
        >>>
        >>> # Pretrained ResNet18 encoder
        >>> model = UNet(n_channels=13, n_classes=10, use_pretrained=True)
        >>>
        >>> # Pretrained ResNet34 encoder
        >>> model = UNet(
        ...     n_channels=13, n_classes=10,
        ...     use_pretrained=True, pretrained_encoder='resnet34'
        ... )
    """

    # Supported pretrained encoders
    SUPPORTED_ENCODERS: tuple[str, ...] = ("resnet18", "resnet34")

    # ResNet channel configurations for pretrained encoders
    RESNET_CHANNELS: dict[str, tuple[int, ...]] = {
        "resnet18": (64, 64, 128, 256, 512),
        "resnet34": (64, 64, 128, 256, 512),
    }

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        bilinear: bool,
        channels: tuple[int, ...],
        use_pretrained: bool,
        pretrained_encoder: str,
        freeze_encoder: bool,
        freeze_encoder_layers: int,
        dropout: float,
        input_projection_type: str = "linear",
        input_projection_hidden: int = 32,
        input_projection_dims: str | list[int] | None = None,
        inference: bool = False,
    ) -> None:
        """Initialize UNet model."""
        super().__init__()

        # Validate inputs
        self._validate_inputs(n_channels, n_classes, channels, use_pretrained, pretrained_encoder)

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.use_pretrained = use_pretrained
        self.pretrained_encoder = pretrained_encoder
        self.dropout_rate = dropout

        # Input projection: n_channels -> 3 (backbone always receives 3-channel input)
        self.input_proj = InputProjection(
            n_channels=n_channels,
            target_channels=3,
            projection_type=input_projection_type,
            hidden_channels=input_projection_hidden,
            dims=input_projection_dims,
        )
        backbone_channels = 3

        # Determine channel configuration
        if use_pretrained:
            self.channels = self.RESNET_CHANNELS[pretrained_encoder]
            self._build_pretrained_encoder(backbone_channels, load_weights=not inference)
        else:
            self.channels = channels
            self._build_custom_encoder(backbone_channels)

        # Build decoder (same for both encoder types)
        self._build_decoder()

        # Bottleneck dropout (applied after encoder, before decoder)
        self.bottleneck_dropout = nn.Dropout2d(p=self.dropout_rate) if self.dropout_rate > 0 else None

        # Initialize weights for non-pretrained parts
        self._initialize_weights()

        # Freeze encoder if requested (freeze_encoder=True takes precedence)
        if freeze_encoder:
            self._freeze_encoder()
        elif freeze_encoder_layers > 0:
            self._freeze_encoder_layers(freeze_encoder_layers)

        # Log model configuration
        logger.info(f"UNet initialized: channels={self.channels}, depth={len(self.channels)}, bilinear={bilinear}, use_pretrained={use_pretrained}")

    def _validate_inputs(
        self,
        n_channels: int,
        n_classes: int,
        channels: tuple[int, ...],
        use_pretrained: bool,
        pretrained_encoder: str,
    ) -> None:
        """Validate constructor inputs."""
        if n_channels <= 0:
            msg = MODEL_INIT_ERRORS.NUM_CHANNELS_POSITIVE.format(n_channels=n_channels)
            log_and_raise(ValueError, msg, logger_instance=logger)

        if n_classes <= 0:
            msg = MODEL_INIT_ERRORS.NUM_CLASSES_POSITIVE.format(num_classes=n_classes)
            log_and_raise(ValueError, msg, logger_instance=logger)

        if len(channels) < 2:
            msg = f"channels must have at least 2 levels for encoder/decoder, got {len(channels)}"
            log_and_raise(ValueError, msg, logger_instance=logger)

        if not all(c > 0 for c in channels):
            msg = f"All channel values must be positive, got {channels}"
            log_and_raise(ValueError, msg, logger_instance=logger)

        if use_pretrained and pretrained_encoder not in self.SUPPORTED_ENCODERS:
            msg = f"pretrained_encoder must be one of {self.SUPPORTED_ENCODERS}, got '{pretrained_encoder}'"
            log_and_raise(ValueError, msg, logger_instance=logger)

    def _build_custom_encoder(self, backbone_channels: int) -> None:
        ch = self.channels
        depth = len(ch)

        # Initial double convolution
        self.inc = DoubleConv(backbone_channels, ch[0])

        # Encoder (downsampling) blocks
        self.encoders = nn.ModuleList()
        for i in range(depth - 1):
            self.encoders.append(Down(ch[i], ch[i + 1]))

    def _build_pretrained_encoder(self, backbone_channels: int, load_weights: bool = True) -> None:
        """Build encoder using pretrained ResNet backbone."""
        if self.pretrained_encoder == "resnet18":
            weights = ResNet18_Weights.IMAGENET1K_V1 if load_weights else None
            resnet = models.resnet18(weights=weights)
        elif self.pretrained_encoder == "resnet34":
            weights = ResNet34_Weights.IMAGENET1K_V1 if load_weights else None
            resnet = models.resnet34(weights=weights)
        else:
            # Should not reach here due to validation
            msg = f"Unsupported encoder: {self.pretrained_encoder}"
            log_and_raise(ValueError, msg, logger_instance=logger)

        # Backbone always receives backbone_channels (3) via input_proj; load weights as-is
        self.conv1 = resnet.conv1

        # Store ResNet components for encoder
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        # ResNet layers as encoder blocks
        self.layer1 = resnet.layer1  # 64 channels, stride 1 (after maxpool)
        self.layer2 = resnet.layer2  # 128 channels, stride 2
        self.layer3 = resnet.layer3  # 256 channels, stride 2
        self.layer4 = resnet.layer4  # 512 channels, stride 2

    def _build_decoder(self) -> None:
        """Build decoder with upsampling blocks."""
        ch = self.channels
        depth = len(ch)

        # Decoder (upsampling) blocks
        self.decoders = nn.ModuleList()

        if self.use_pretrained:
            # For pretrained encoder, decoder goes from deepest to shallowest
            # ResNet: 512 -> 256 -> 128 -> 64 -> 64
            decoder_levels = [
                (ch[4], ch[3], ch[3]),  # x1=512, skip=256, out=256
                (ch[3], ch[2], ch[2]),  # x1=256, skip=128, out=128
                (ch[2], ch[1], ch[1]),  # x1=128, skip=64, out=64
                (ch[1], ch[0], ch[0]),  # x1=64, skip=64, out=64
            ]
            for x1_ch, skip_ch, out_ch in decoder_levels:
                # Bilinear keeps channels; transposed-conv path halves x1 channels before concat.
                in_ch = x1_ch + skip_ch if self.bilinear else (x1_ch // 2) + skip_ch
                self.decoders.append(Up(in_ch, out_ch, self.bilinear, x1_channels=x1_ch))

            # 5th upsample: 64ch@128 -> 64ch@256 (no skip connection)
            self.final_up: nn.Module = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                DoubleConv(ch[0], ch[0]),
            )
        else:
            # For custom encoder, build decoder in reverse
            for i in range(depth - 1, 0, -1):
                x1_ch = ch[i]  # deeper feature map channels
                skip_ch = ch[i - 1]
                out_ch = skip_ch
                # Bilinear keeps channels; transposed-conv path halves x1 channels before concat.
                in_ch = x1_ch + skip_ch if self.bilinear else (x1_ch // 2) + skip_ch
                self.decoders.append(Up(in_ch, out_ch, self.bilinear, x1_channels=x1_ch))

        # Output convolution
        self.outc = OutConv(ch[0], self.n_classes)

    def _initialize_weights(self) -> None:
        """Initialize weights using He initialization for ReLU activations.

        Based on: He, K., et al. (2015). "Delving Deep into Rectifiers"
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Skip pretrained encoder weights
                if self.use_pretrained and self._is_pretrained_module(m):
                    continue
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _is_pretrained_module(self, module: nn.Module) -> bool:
        """Check if module is part of pretrained encoder."""
        if not self.use_pretrained:
            return False

        pretrained_modules = [
            self.conv1,
            self.bn1,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
        ]

        for pretrained_module in pretrained_modules:
            if module is pretrained_module:
                return True
            for child in pretrained_module.modules():
                if module is child:
                    return True
        return False

    def forward(self, x: torch.Tensor) -> Any:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, n_channels, height, width)

        Returns:
            Output tensor of shape (batch_size, n_classes, height, width)
        """
        x = self.input_proj(x)
        if self.use_pretrained:
            return self._forward_pretrained(x)
        else:
            return self._forward_custom(x)

    def _forward_custom(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for custom encoder."""
        # Store encoder outputs for skip connections
        encoder_outputs: list[torch.Tensor] = []

        # Initial convolution
        x = self.inc(x)
        encoder_outputs.append(x)

        # Encoder path
        for encoder in self.encoders:
            x = encoder(x)
            encoder_outputs.append(x)

        # Bottleneck dropout
        if self.bottleneck_dropout is not None:
            x = self.bottleneck_dropout(x)

        # Decoder path (process in reverse, skip the last encoder output)
        for i, decoder in enumerate(self.decoders):
            # Get corresponding skip connection (from encoder outputs in reverse)
            skip_idx = len(encoder_outputs) - 2 - i
            x = decoder(x, encoder_outputs[skip_idx])

        # Output
        result: torch.Tensor = self.outc(x)
        return result

    def _forward_pretrained(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for pretrained ResNet encoder."""
        # Store encoder outputs for skip connections
        encoder_outputs: list[torch.Tensor] = []

        # Initial conv + bn + relu
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        encoder_outputs.append(x)  # 64 channels, 1/2 resolution

        # Max pooling
        x = self.maxpool(x)

        # ResNet layers
        x = self.layer1(x)
        encoder_outputs.append(x)  # 64 channels, 1/4 resolution

        x = self.layer2(x)
        encoder_outputs.append(x)  # 128 channels, 1/8 resolution

        x = self.layer3(x)
        encoder_outputs.append(x)  # 256 channels, 1/16 resolution

        x = self.layer4(x)  # 512 channels, 1/32 resolution (bottleneck)

        # Bottleneck dropout
        if self.bottleneck_dropout is not None:
            x = self.bottleneck_dropout(x)

        # Decoder path
        for i, decoder in enumerate(self.decoders):
            skip_idx = len(encoder_outputs) - 1 - i
            x = decoder(x, encoder_outputs[skip_idx])

        # 5th upsample: 128x128 -> 256x256 (no skip)
        x = self.final_up(x)

        # Output
        result: torch.Tensor = self.outc(x)
        return result

    def get_encoder_params(self) -> list[nn.Parameter]:
        """Get encoder parameters (for differential learning rates)."""
        params: list[nn.Parameter] = []
        if self.use_pretrained:
            params.extend(self.conv1.parameters())
            params.extend(self.bn1.parameters())
            params.extend(self.layer1.parameters())
            params.extend(self.layer2.parameters())
            params.extend(self.layer3.parameters())
            params.extend(self.layer4.parameters())
        else:
            params.extend(self.inc.parameters())
            for encoder in self.encoders:
                params.extend(encoder.parameters())
        return params

    def get_decoder_params(self) -> list[nn.Parameter]:
        """Get decoder parameters (for differential learning rates)."""
        params: list[nn.Parameter] = []
        for decoder in self.decoders:
            params.extend(decoder.parameters())
        params.extend(self.outc.parameters())
        return list(params)

    def _freeze_encoder(self) -> None:
        """Freeze encoder parameters so they are not updated during training."""
        if self.use_pretrained:
            for param in self.conv1.parameters():
                param.requires_grad = False
            for param in self.bn1.parameters():
                param.requires_grad = False
            for param in self.layer1.parameters():
                param.requires_grad = False
            for param in self.layer2.parameters():
                param.requires_grad = False
            for param in self.layer3.parameters():
                param.requires_grad = False
            for param in self.layer4.parameters():
                param.requires_grad = False
        else:
            for param in self.inc.parameters():
                param.requires_grad = False
            for encoder in self.encoders:
                for param in encoder.parameters():
                    param.requires_grad = False
        logger.info("Encoder weights frozen")

    def _freeze_encoder_layers(self, n_layers: int) -> None:
        """Freeze first N layers of the encoder.

        For pretrained UNet (ResNet18/34):
        - Layer 0: conv1 + bn1 (initial feature extraction)
        - Layer 1: layer1 (64 channels)
        - Layer 2: layer2 (128 channels)
        - Layer 3: layer3 (256 channels)
        - Layer 4: layer4 (512 channels, bottleneck)

        For custom UNet:
        - Layer 0: inc (initial double conv)
        - Layer 1-4: encoders[0-3] (downsampling blocks)

        Args:
            n_layers: Number of layers to freeze (1-5)
        """
        if n_layers <= 0:
            return

        max_layers = 5
        if n_layers > max_layers:
            logger.warning(f"freeze_encoder_layers={n_layers} exceeds max layers {max_layers}, using {max_layers}")
            n_layers = max_layers

        if self.use_pretrained:
            # Pretrained ResNet encoder
            layer_groups = [
                ("conv1+bn1", [self.conv1, self.bn1]),
                ("layer1", [self.layer1]),
                ("layer2", [self.layer2]),
                ("layer3", [self.layer3]),
                ("layer4", [self.layer4]),
            ]
        else:
            # Custom encoder
            layer_groups = [
                ("inc", [self.inc]),
                *[(f"encoder{i}", [self.encoders[i]]) for i in range(len(self.encoders))],
            ]

        frozen_count = 0
        for i in range(min(n_layers, len(layer_groups))):
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
        if self.use_pretrained:
            layer_groups = [
                ("conv1+bn1", [self.conv1, self.bn1]),
                ("layer1", [self.layer1]),
                ("layer2", [self.layer2]),
                ("layer3", [self.layer3]),
                ("layer4", [self.layer4]),
            ]
        else:
            layer_groups = [
                ("inc", [self.inc]),
                *[(f"encoder{i}", [self.encoders[i]]) for i in range(len(self.encoders))],
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
