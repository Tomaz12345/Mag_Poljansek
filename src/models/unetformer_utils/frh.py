# Implemented from: Wang et al., "UNetFormer: A UNet-like transformer for efficient semantic
#   segmentation of remote sensing urban scene imagery", ISPRS J. Photogramm. Remote Sens. 190, 2022
# Adapted from: https://github.com/WangLibo1995/GeoSeg/blob/main/geoseg/models/UNetFormer.py
#   GNU GPL v3.0, Libo Wang; modified for this project.
#   Licence text: LICENSE

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv_modules import Conv, ConvBN, ConvBNReLU, SeparableConvBN


class WF(nn.Module):
    """Weighted Fusion module.

    Performs learnable weighted fusion of encoder skip features and
    upsampled decoder features.

    Formula: FF = α · RF + (1-α) · GLF  (Eq. 1 in paper)
    Weights are normalized via ReLU to ensure non-negativity.

    Args:
        in_channels: Number of encoder skip connection channels.
        decoder_channels: Number of decoder feature channels.
        eps: Epsilon for numerical stability in weight normalization.
    """

    def __init__(
        self,
        in_channels: int = 128,
        decoder_channels: int = 128,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.pre_conv = Conv(in_channels, decoder_channels, kernel_size=1)

        self.weights = nn.Parameter(
            torch.ones(2, dtype=torch.float32),
            requires_grad=True,
        )
        self.eps = eps
        self.post_conv = ConvBNReLU(decoder_channels, decoder_channels, kernel_size=3)

    def forward(self, x: torch.Tensor, res: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Decoder features to be upsampled.
            res: Encoder skip connection features (target spatial size).

        Returns:
            Fused feature of shape (B, decoder_channels, H, W).
        """
        x = F.interpolate(x, size=res.shape[2:], mode="bilinear", align_corners=False)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        return x


class FeatureRefinementHead(nn.Module):
    """Feature Refinement Head (FRH).

    Used at the shallowest decoder stage (1/4 resolution) to bridge the
    semantic gap between encoder and decoder features.

    Architecture (Fig. 5 in paper):
    1. Weighted fusion of upsampled decoder + encoder skip features
    2. Post-conv (ConvBNReLU 3x3)
    3. Dual attention paths:
       - Channel: GAP → Conv1x1 → ReLU6 → Conv1x1 → Sigmoid → multiply
       - Spatial: Depthwise Conv 3x3 → Sigmoid → multiply
    4. Sum of channel and spatial paths
    5. SeparableConvBN 3x3 + shortcut (ConvBN 1x1) → ReLU6

    Args:
        in_channels: Number of encoder skip connection channels.
        decoder_channels: Number of decoder feature channels.
    """

    def __init__(
        self,
        in_channels: int = 64,
        decoder_channels: int = 64,
    ) -> None:
        super().__init__()
        # Weighted fusion (same as WF)
        self.pre_conv = Conv(in_channels, decoder_channels, kernel_size=1)
        self.weights = nn.Parameter(
            torch.ones(2, dtype=torch.float32),
            requires_grad=True,
        )
        self.eps = 1e-8
        self.post_conv = ConvBNReLU(decoder_channels, decoder_channels, kernel_size=3)

        # Spatial attention path: depthwise conv → Sigmoid
        self.pa = nn.Sequential(
            nn.Conv2d(
                decoder_channels,
                decoder_channels,
                kernel_size=3,
                padding=1,
                groups=decoder_channels,
            ),
            nn.Sigmoid(),
        )

        # Channel attention path: GAP → reduce → ReLU6 → expand → Sigmoid
        reduction = decoder_channels // 16 if decoder_channels >= 16 else 1
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv(decoder_channels, reduction, kernel_size=1),
            nn.ReLU6(),
            Conv(reduction, decoder_channels, kernel_size=1),
            nn.Sigmoid(),
        )

        # Shortcut and output projection
        self.shortcut = ConvBN(decoder_channels, decoder_channels, kernel_size=1)
        self.proj = SeparableConvBN(decoder_channels, decoder_channels, kernel_size=3)
        self.act = nn.ReLU6()

    def forward(self, x: torch.Tensor, res: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Decoder features of shape (B, decoder_channels, H/2, W/2).
            res: Encoder skip connection of shape (B, in_channels, H, W).

        Returns:
            Refined feature of shape (B, decoder_channels, H, W).
        """
        # Upsample decoder features to match encoder skip spatial size
        x = F.interpolate(x, size=res.shape[2:], mode="bilinear", align_corners=False)

        # Weighted fusion
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)

        # Shortcut from fused feature
        shortcut = self.shortcut(x)

        # Dual attention paths
        pa = self.pa(x) * x  # Spatial attention (sigmoid-gated)
        ca = self.ca(x) * x  # Channel attention (sigmoid-gated)

        # Combine paths
        x = pa + ca

        # Output: SeparableConvBN + shortcut + ReLU6
        x = self.proj(x) + shortcut
        x = self.act(x)

        return x
