from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PPM(nn.Module):
    """Pyramid Pooling Module for multi-scale context.

    Args:
        in_channels: Input channel dimension.
        pool_scales: Pooling scales for pyramid.
        channels: Output channels per scale.
    """

    def __init__(
        self,
        in_channels: int,
        pool_scales: tuple[int, ...] = (1, 2, 3, 6),
        channels: int = 512,
    ) -> None:
        super().__init__()
        self.pool_scales = pool_scales

        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    nn.Conv2d(in_channels, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                )
                for scale in pool_scales
            ]
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                in_channels + len(pool_scales) * channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Output tensor of shape (B, channels, H, W).
        """
        h, w = x.shape[2:]
        ppm_outs_list: list[torch.Tensor] = [x]

        for stage in self.stages:
            ppm_out = stage(x)
            ppm_out = F.interpolate(ppm_out, size=(h, w), mode="bilinear", align_corners=True)
            ppm_outs_list.append(ppm_out)

        ppm_outs = torch.cat(ppm_outs_list, dim=1)
        output: torch.Tensor = self.bottleneck(ppm_outs)

        return output


class UPerNetDecoder(nn.Module):
    """UPerNet decoder for semantic segmentation.

    Combines multi-scale features using PPM and FPN-style lateral connections.

    Args:
        in_channels: List of input channel dimensions for each scale.
        channels: Decoder channel dimension.
        num_classes: Number of output classes.
        pool_scales: Pooling scales for PPM module.
        dropout_rate: Dropout rate before final classification.
    """

    def __init__(
        self,
        in_channels: list[int],
        channels: int = 512,
        num_classes: int = 10,
        pool_scales: tuple[int, ...] = (1, 2, 3, 6),
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.num_classes = num_classes

        # PPM module for deepest features
        self.ppm = PPM(
            in_channels=in_channels[-1],
            pool_scales=pool_scales,
            channels=channels,
        )

        # Lateral connections (1x1 conv to match channels)
        self.lateral_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_ch, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                )
                for in_ch in in_channels[:-1]
            ]
        )

        # FPN smoothing convolutions
        self.fpn_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                )
                for _ in in_channels[:-1]
            ]
        )

        # Bottleneck after concatenation
        self.fpn_bottleneck = nn.Sequential(
            nn.Conv2d(len(in_channels) * channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        # Final classifier
        self.dropout = nn.Dropout2d(dropout_rate)
        self.classifier = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(
        self,
        features: list[torch.Tensor],
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            features: List of feature tensors from encoder (low to high level).
            output_size: Target output size (H, W). If None, uses 4x the final feature size.

        Returns:
            Segmentation logits of shape (B, num_classes, H, W).
        """
        # PPM on deepest features
        ppm_out = self.ppm(features[-1])

        # FPN top-down pathway
        fpn_features = [ppm_out]
        for i in range(len(features) - 2, -1, -1):
            lateral = self.lateral_convs[i](features[i])
            h, w = lateral.shape[2:]
            upsampled = F.interpolate(fpn_features[-1], size=(h, w), mode="bilinear", align_corners=True)
            fpn_out = lateral + upsampled
            fpn_out = self.fpn_convs[i](fpn_out)
            fpn_features.append(fpn_out)

        fpn_features = fpn_features[::-1]  # Reverse to low-to-high order

        # Upsample all to largest feature map size
        target_h, target_w = fpn_features[0].shape[2:]
        upsampled_features = []
        for feat in fpn_features:
            if feat.shape[2:] != (target_h, target_w):
                feat = F.interpolate(feat, size=(target_h, target_w), mode="bilinear", align_corners=True)
            upsampled_features.append(feat)

        # Concatenate and fuse
        fused = torch.cat(upsampled_features, dim=1)
        fused = self.fpn_bottleneck(fused)

        # Classify
        output = self.dropout(fused)
        output = self.classifier(output)

        # Upsample to target size
        if output_size is not None:
            output = F.interpolate(output, size=output_size, mode="bilinear", align_corners=True)

        result: torch.Tensor = output
        return result


class FPNDecoder(nn.Module):
    """Feature Pyramid Network decoder.

    Simple FPN-style decoder that fuses multi-scale features.

    Args:
        in_channels: List of input channel dimensions for each scale.
        channels: Decoder channel dimension.
        num_classes: Number of output classes.
        dropout_rate: Dropout rate before final classification.
    """

    def __init__(
        self,
        in_channels: list[int],
        channels: int = 256,
        num_classes: int = 10,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.num_classes = num_classes

        # Lateral connections
        self.lateral_convs = nn.ModuleList([nn.Conv2d(in_ch, channels, kernel_size=1) for in_ch in in_channels])

        # Output convolutions
        self.output_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                )
                for _ in in_channels
            ]
        )

        # Fusion and classification
        self.fusion = nn.Sequential(
            nn.Conv2d(len(in_channels) * channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(dropout_rate)
        self.classifier = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(
        self,
        features: list[torch.Tensor],
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            features: List of feature tensors from encoder.
            output_size: Target output size (H, W).

        Returns:
            Segmentation logits of shape (B, num_classes, H, W).
        """
        # Apply lateral convolutions
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, features, strict=False)]

        # Top-down pathway with element-wise addition
        for i in range(len(laterals) - 1, 0, -1):
            h, w = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i], size=(h, w), mode="bilinear", align_corners=True)

        # Apply output convolutions
        outputs = [conv(lat) for conv, lat in zip(self.output_convs, laterals, strict=False)]

        # Upsample all to largest size
        target_h, target_w = outputs[0].shape[2:]
        upsampled = []
        for out in outputs:
            if out.shape[2:] != (target_h, target_w):
                out = F.interpolate(out, size=(target_h, target_w), mode="bilinear", align_corners=True)
            upsampled.append(out)

        # Concatenate and fuse
        fused = torch.cat(upsampled, dim=1)
        fused = self.fusion(fused)

        # Classify
        output = self.dropout(fused)
        output = self.classifier(output)

        # Upsample to target size
        if output_size is not None:
            output = F.interpolate(output, size=output_size, mode="bilinear", align_corners=True)

        result: torch.Tensor = output
        return result


class LinearDecoder(nn.Module):
    """Simple linear decoder for lightweight segmentation.

    Directly projects encoder features to class logits with minimal processing.

    Args:
        in_channels: Input channel dimension (from encoder).
        num_classes: Number of output classes.
        dropout_rate: Dropout rate.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 10,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.head = nn.Sequential(
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(in_channels, num_classes, kernel_size=1),
        )

    def forward(
        self,
        features: list[torch.Tensor] | torch.Tensor,
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            features: Feature tensor or list (uses last if list).
            output_size: Target output size (H, W).

        Returns:
            Segmentation logits of shape (B, num_classes, H, W).
        """
        x = features[-1] if isinstance(features, list) else features

        output = self.head(x)

        if output_size is not None:
            output = F.interpolate(output, size=output_size, mode="bilinear", align_corners=True)

        result: torch.Tensor = output
        return result
