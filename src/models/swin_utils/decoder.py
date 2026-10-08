from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = self.relu(self.bn(self.conv(x)))
        return result


class PPM(nn.Module):
    """Pyramid Pooling Module.

    Multi-scale feature aggregation using pooling at different scales.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels per branch.
        pool_scales: Pooling output sizes for each branch.
        align_corners: Whether to align corners in bilinear interpolation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool_scales: tuple[int, ...] = (1, 2, 3, 6),
        align_corners: bool = False,
    ) -> None:
        super().__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.stages = nn.ModuleList()

        for scale in pool_scales:
            self.stages.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(output_size=scale),
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            List of upsampled pooled features, each of shape (B, out_channels, H, W).
        """
        h, w = x.shape[2:]
        outputs = []
        for stage in self.stages:
            out = stage(x)
            out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=self.align_corners)
            outputs.append(out)
        return outputs


class UPerNetDecoder(nn.Module):
    """UPerNet-style decoder for semantic segmentation.

    Combines multi-scale features from the encoder using Feature Pyramid
    Network (FPN) with Pyramid Pooling Module (PPM).

    Reference: Xiao et al. "Unified Perceptual Parsing for Scene Understanding"

    Args:
        in_channels: List of input channels from each encoder stage.
        channel: Decoder channel dimension.
        num_classes: Number of output classes.
        pool_scales: Pooling scales for PPM.
        dropout: Dropout rate before final classification.
        align_corners: Whether to align corners in bilinear interpolation.

    Shape:
        - Input: List of feature maps from encoder stages
        - Output: (B, num_classes, H_original, W_original)
    """

    def __init__(
        self,
        in_channels: list[int],
        channel: int = 512,
        num_classes: int = 10,
        pool_scales: tuple[int, ...] = (1, 2, 3, 6),
        dropout: float = 0.1,
        align_corners: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.channel = channel
        self.num_classes = num_classes
        self.align_corners = align_corners

        # PPM on the last (deepest) feature map
        self.ppm = PPM(
            in_channels=in_channels[-1],
            out_channels=channel,
            pool_scales=pool_scales,
            align_corners=align_corners,
        )

        # Bottleneck after PPM
        # PPM outputs: original + len(pool_scales) branches
        self.bottleneck = ConvBNReLU(
            in_channels[-1] + channel * len(pool_scales),
            channel,
            kernel_size=3,
            padding=1,
        )

        # Lateral connections for FPN (Conv + BN + ReLU, matching original UPerNet)
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for in_ch in in_channels[:-1]:  # All stages except the last
            self.lateral_convs.append(ConvBNReLU(in_ch, channel, kernel_size=1, padding=0))
            self.fpn_convs.append(ConvBNReLU(channel, channel, kernel_size=3, padding=1))

        # FPN output fusion
        self.fpn_bottleneck = ConvBNReLU(
            len(in_channels) * channel,
            channel,
            kernel_size=3,
            padding=1,
        )

        # Final classification head
        self.dropout = nn.Dropout2d(dropout)
        self.classifier = nn.Conv2d(channel, num_classes, kernel_size=1)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """Forward pass.

        Args:
            features: List of feature maps from encoder stages.
                      Each tensor has shape (B, C_i, H_i, W_i).
                      Features are expected in order from high to low resolution.

        Returns:
            Segmentation logits of shape (B, num_classes, H_original, W_original).
        """
        # Get original input size from the first (highest resolution) feature
        # The first feature is at 1/4 resolution after patch embedding
        target_size = (features[0].shape[2] * 4, features[0].shape[3] * 4)

        # Apply PPM to the deepest feature
        ppm_out = self.ppm(features[-1])
        ppm_out = torch.cat([features[-1]] + ppm_out, dim=1)
        ppm_out = self.bottleneck(ppm_out)

        # FPN: top-down pathway
        fpn_features = [ppm_out]

        for i in range(len(self.in_channels) - 2, -1, -1):
            lateral = self.lateral_convs[i](features[i])
            # Upsample previous FPN feature to match lateral size
            prev_fpn = F.interpolate(
                fpn_features[-1],
                size=lateral.shape[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
            fpn_feat = lateral + prev_fpn
            fpn_feat = self.fpn_convs[i](fpn_feat)
            fpn_features.append(fpn_feat)

        # Reverse to get features from high to low resolution
        fpn_features = fpn_features[::-1]

        # Upsample all FPN features to the same resolution (highest)
        output_size = fpn_features[0].shape[2:]
        fpn_upsampled = [fpn_features[0]]
        for feat in fpn_features[1:]:
            fpn_upsampled.append(F.interpolate(feat, size=output_size, mode="bilinear", align_corners=self.align_corners))

        # Concatenate and fuse
        x = torch.cat(fpn_upsampled, dim=1)
        x = self.fpn_bottleneck(x)

        # Final classification
        x = self.dropout(x)
        x = self.classifier(x)

        # Upsample to original input resolution
        out: torch.Tensor = F.interpolate(x, size=target_size, mode="bilinear", align_corners=self.align_corners)

        return out


class FPNDecoder(nn.Module):
    """Simple Feature Pyramid Network decoder.

    A lighter alternative to UPerNet, without PPM.

    Args:
        in_channels: List of input channels from each encoder stage.
        out_channels: Output channel dimension for FPN.
        num_classes: Number of output classes.
        dropout: Dropout rate before final classification.
        align_corners: Whether to align corners in bilinear interpolation.

    Shape:
        - Input: List of feature maps from encoder stages
        - Output: (B, num_classes, H_original, W_original)
    """

    def __init__(
        self,
        in_channels: list[int],
        out_channels: int = 256,
        num_classes: int = 10,
        dropout: float = 0.1,
        align_corners: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.align_corners = align_corners

        # Lateral connections (1x1 conv to reduce channels)
        self.lateral_convs = nn.ModuleList()
        for in_ch in in_channels:
            self.lateral_convs.append(nn.Conv2d(in_ch, out_channels, kernel_size=1, bias=False))

        # Output convolutions (refine features after fusion)
        self.output_convs = nn.ModuleList()
        for _ in in_channels:
            self.output_convs.append(ConvBNReLU(out_channels, out_channels, kernel_size=3, padding=1))

        # Fusion layer
        self.fusion = ConvBNReLU(
            out_channels * len(in_channels),
            out_channels,
            kernel_size=3,
            padding=1,
        )

        # Classification head
        self.dropout = nn.Dropout2d(dropout)
        self.classifier = nn.Conv2d(out_channels, num_classes, kernel_size=1)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """Forward pass.

        Args:
            features: List of feature maps from encoder stages.

        Returns:
            Segmentation logits of shape (B, num_classes, H_original, W_original).
        """
        # Get target size (original input resolution)
        target_size = (features[0].shape[2] * 4, features[0].shape[3] * 4)

        # Apply lateral convolutions
        laterals = [lateral_conv(feat) for lateral_conv, feat in zip(self.lateral_convs, features, strict=True)]

        # Top-down pathway with element-wise addition
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )

        # Apply output convolutions
        fpn_outs = [output_conv(lateral) for output_conv, lateral in zip(self.output_convs, laterals, strict=True)]

        # Upsample all to highest resolution
        output_size = fpn_outs[0].shape[2:]
        fpn_upsampled = [fpn_outs[0]]
        for feat in fpn_outs[1:]:
            fpn_upsampled.append(F.interpolate(feat, size=output_size, mode="bilinear", align_corners=self.align_corners))

        # Concatenate and fuse
        x = torch.cat(fpn_upsampled, dim=1)
        x = self.fusion(x)

        # Final classification
        x = self.dropout(x)
        x = self.classifier(x)

        # Upsample to original resolution
        out: torch.Tensor = F.interpolate(x, size=target_size, mode="bilinear", align_corners=self.align_corners)

        return out


class AuxiliaryFCNHead(nn.Module):
    """Auxiliary FCN head for intermediate supervision.

    A lightweight classification head applied to an intermediate encoder stage
    to provide auxiliary supervision during training. This follows the standard
    practice from PSPNet/UPerNet segmentation frameworks.

    In the official Swin + UPerNet config (mmsegmentation), this head is applied
    to stage 3 (index 2) features with auxiliary loss weight 0.4.

    Args:
        in_channels: Number of input channels from the encoder stage.
        channels: Number of intermediate channels.
        num_classes: Number of output segmentation classes.
        dropout: Dropout rate before final classification.
        align_corners: Whether to align corners in bilinear interpolation.

    Shape:
        - Input: (B, C, H, W) feature map from an encoder stage
        - Output: (B, num_classes, H_original, W_original)
    """

    def __init__(
        self,
        in_channels: int,
        channels: int = 256,
        num_classes: int = 10,
        dropout: float = 0.1,
        align_corners: bool = False,
    ) -> None:
        super().__init__()
        self.align_corners = align_corners

        self.conv = ConvBNReLU(in_channels, channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout2d(dropout)
        self.classifier = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Feature map from an encoder stage, shape (B, C, H, W).
            target_size: Target output size (H_original, W_original).

        Returns:
            Segmentation logits of shape (B, num_classes, H_original, W_original).
        """
        x = self.conv(x)
        x = self.dropout(x)
        x = self.classifier(x)
        out: torch.Tensor = F.interpolate(x, size=target_size, mode="bilinear", align_corners=self.align_corners)
        return out
