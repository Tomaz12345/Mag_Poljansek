# Implemented from: Wang et al., "UNetFormer: A UNet-like transformer for efficient semantic
#   segmentation of remote sensing urban scene imagery", ISPRS J. Photogramm. Remote Sens. 190, 2022
# Reference implementation: https://github.com/WangLibo1995/GeoSeg

from __future__ import annotations

import logging
from collections.abc import Iterator

import torch
import torch.nn as nn

from src.models.pretrained_loading import log_pretrained_load_result

logger = logging.getLogger(__name__)


class BasicBlock(nn.Module):
    """Basic residual block for ResNet18/34."""

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out: torch.Tensor = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity  # Non-inplace for SHAP compatibility
        out = self.relu(out)

        return out


class ResNet18Encoder(nn.Module):
    """ResNet18 encoder for UNetFormer.

    Outputs multi-scale features from 4 stages:
    - Stage 1: H/4, W/4, 64 channels
    - Stage 2: H/8, W/8, 128 channels
    - Stage 3: H/16, W/16, 256 channels
    - Stage 4: H/32, W/32, 512 channels

    Args:
        in_channels: Number of input channels.
        pretrained: Whether to load pretrained ImageNet weights.
        encoder_channels: Output channels for each stage.
    """

    def __init__(
        self,
        in_channels: int = 3,
        pretrained: bool = False,
        encoder_channels: tuple[int, ...] = (64, 128, 256, 512),
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.encoder_channels = encoder_channels

        # Initial convolution and pooling (stem)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Build encoder stages
        self.in_planes = 64
        self.layer1 = self._make_layer(encoder_channels[0], 2, stride=1)
        self.layer2 = self._make_layer(encoder_channels[1], 2, stride=2)
        self.layer3 = self._make_layer(encoder_channels[2], 2, stride=2)
        self.layer4 = self._make_layer(encoder_channels[3], 2, stride=2)

        # Initialize weights
        self._init_weights()

        # Load pretrained weights if requested
        if pretrained:
            self._load_pretrained()

    def _make_layer(
        self,
        out_channels: int,
        num_blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        """Build a ResNet stage."""
        downsample = None
        if stride != 1 or self.in_planes != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(BasicBlock(self.in_planes, out_channels, stride, downsample))
        self.in_planes = out_channels
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(self.in_planes, out_channels))

        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _load_pretrained(self) -> None:
        """Load pretrained ImageNet weights."""
        try:
            import torchvision.models as models

            pretrained_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            state_dict = pretrained_model.state_dict()

            # Filter out fc layer
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith("fc.")}

            # Load with strict=False to handle missing/unexpected keys
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            log_pretrained_load_result(
                missing,
                unexpected,
                logger,
                model_name="UNetFormer(ResNet18)",
            )
            logger.info("Loaded pretrained ResNet18 weights.")

        except ImportError:
            logger.warning("torchvision not available. Skipping pretrained weights.")
        except Exception as e:
            logger.warning(f"Failed to load pretrained weights: {e}")

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass returning multi-scale features.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            List of 4 feature maps from each stage:
            - [0]: (B, 64, H/4, W/4)
            - [1]: (B, 128, H/8, W/8)
            - [2]: (B, 256, H/16, W/16)
            - [3]: (B, 512, H/32, W/32)
        """
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Encoder stages
        features = []
        x = self.layer1(x)
        features.append(x)  # H/4, 64

        x = self.layer2(x)
        features.append(x)  # H/8, 128

        x = self.layer3(x)
        features.append(x)  # H/16, 256

        x = self.layer4(x)
        features.append(x)  # H/32, 512

        return features

    def get_params(self) -> Iterator[nn.Parameter]:
        yield from self.parameters()
