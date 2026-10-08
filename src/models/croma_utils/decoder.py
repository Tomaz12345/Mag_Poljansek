from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_1d_sincos_pos_embed(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """Generate 1D sinusoidal position embedding from grid positions.

    Args:
        embed_dim: Output dimension (must be even).
        pos: Position array, flattened to 1D.

    Returns:
        Sinusoidal embedding of shape (num_positions, embed_dim).
    """
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos_flat = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos_flat, omega)  # (M, D/2)

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)
    emb: np.ndarray = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# Segmentation decoders, used for fine-tuning


class UPerNetSegDecoder(nn.Module):
    """UPerNet decoder for semantic segmentation.

    Unified Perceptual Parsing Network that fuses multi-scale features
    through a Pyramid Pooling Module (PPM) and Feature Pyramid Network.

    Args:
        in_channels: List of input channel dimensions from encoder.
        channels: Internal channel dimension.
        num_classes: Number of segmentation classes.
        dropout_rate: Dropout before classification head.
        pool_scales: Pooling scales for PPM module.
    """

    def __init__(
        self,
        in_channels: list[int],
        channels: int = 512,
        num_classes: int = 10,
        dropout_rate: float = 0.1,
        pool_scales: tuple[int, ...] = (1, 2, 3, 6),
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels

        # Lateral convolutions (project each scale to channels dim)
        self.lateral_convs = nn.ModuleList()
        for in_ch in in_channels:
            self.lateral_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, channels, 1),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                )
            )

        # Pyramid Pooling Module on the deepest features
        self.ppm = PyramidPoolingModule(in_channels[-1], channels, pool_scales)

        # FPN output convolutions
        self.fpn_convs = nn.ModuleList()
        for _ in range(len(in_channels)):
            self.fpn_convs.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels, 3, padding=1),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                )
            )

        # Bottleneck to merge all FPN outputs
        self.bottleneck = nn.Sequential(
            nn.Conv2d(channels * len(in_channels), channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.dropout = nn.Dropout2d(dropout_rate)
        self.cls_seg = nn.Conv2d(channels, num_classes, 1)

    def forward(
        self,
        features: list[torch.Tensor],
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            features: List of multi-scale feature maps from encoder.
            output_size: Target spatial size (H, W) for output.

        Returns:
            Segmentation logits of shape (B, num_classes, H, W).
        """
        # Apply lateral convolutions
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, features, strict=True)]

        # Apply PPM on deepest features
        laterals[-1] = self.ppm(features[-1])

        # Top-down path
        for i in range(len(laterals) - 1, 0, -1):
            target_size = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i], size=target_size, mode="bilinear", align_corners=False)

        # Apply FPN convolutions
        fpn_outs = [conv(lat) for conv, lat in zip(self.fpn_convs, laterals, strict=True)]

        # Upsample all to the largest scale
        target_size = fpn_outs[0].shape[2:]
        for i in range(1, len(fpn_outs)):
            fpn_outs[i] = F.interpolate(fpn_outs[i], size=target_size, mode="bilinear", align_corners=False)

        # Concatenate and classify
        x = torch.cat(fpn_outs, dim=1)
        x = self.bottleneck(x)
        x = self.dropout(x)
        x = self.cls_seg(x)

        if output_size is not None:
            x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        result: torch.Tensor = x
        return result


class PyramidPoolingModule(nn.Module):
    """Pyramid Pooling Module (PPM) for UPerNet.

    Args:
        in_channels: Input channel dimension.
        channels: Output channel dimension per scale.
        pool_scales: Pooling output sizes.
    """

    def __init__(
        self,
        in_channels: int,
        channels: int,
        pool_scales: tuple[int, ...] = (1, 2, 3, 6),
    ) -> None:
        super().__init__()
        self.pool_branches = nn.ModuleList()
        for scale in pool_scales:
            self.pool_branches.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    nn.Conv2d(in_channels, channels, 1),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                )
            )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + channels * len(pool_scales), channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_size = x.shape[2:]
        pool_outs = [x]
        for branch in self.pool_branches:
            pooled = branch(x)
            pool_outs.append(F.interpolate(pooled, size=target_size, mode="bilinear", align_corners=False))
        result: torch.Tensor = self.bottleneck(torch.cat(pool_outs, dim=1))
        return result


class FPNSegDecoder(nn.Module):
    """Feature Pyramid Network decoder for semantic segmentation.

    Simpler than UPerNet - uses lateral connections and top-down merging.

    Args:
        in_channels: List of input channel dimensions from encoder.
        channels: Internal channel dimension.
        num_classes: Number of segmentation classes.
        dropout_rate: Dropout before classification head.
    """

    def __init__(
        self,
        in_channels: list[int],
        channels: int = 256,
        num_classes: int = 10,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()

        # Lateral convolutions
        self.lateral_convs = nn.ModuleList()
        for in_ch in in_channels:
            self.lateral_convs.append(nn.Conv2d(in_ch, channels, 1))

        # Output convolutions
        self.output_convs = nn.ModuleList()
        for _ in range(len(in_channels)):
            self.output_convs.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels, 3, padding=1),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                )
            )

        self.merge = nn.Sequential(
            nn.Conv2d(channels * len(in_channels), channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(dropout_rate)
        self.cls_seg = nn.Conv2d(channels, num_classes, 1)

    def forward(
        self,
        features: list[torch.Tensor],
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, features, strict=True)]

        # Top-down merging
        for i in range(len(laterals) - 1, 0, -1):
            target_size = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i], size=target_size, mode="bilinear", align_corners=False)

        # Output convolutions
        fpn_outs = [conv(lat) for conv, lat in zip(self.output_convs, laterals, strict=True)]

        # Upsample all to largest scale and merge
        target_size = fpn_outs[0].shape[2:]
        for i in range(1, len(fpn_outs)):
            fpn_outs[i] = F.interpolate(fpn_outs[i], size=target_size, mode="bilinear", align_corners=False)

        x = torch.cat(fpn_outs, dim=1)
        x = self.merge(x)
        x = self.dropout(x)
        x = self.cls_seg(x)

        if output_size is not None:
            x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        result: torch.Tensor = x
        return result


class LinearSegDecoder(nn.Module):
    """Simple linear decoder for semantic segmentation.

    Projects encoder features directly to class logits.

    Args:
        in_channels: Input channel dimension.
        num_classes: Number of segmentation classes.
        dropout_rate: Dropout before classification head.
    """

    def __init__(
        self,
        in_channels: int = 768,
        num_classes: int = 10,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_channels, num_classes),
        )

    def forward(
        self,
        features: list[torch.Tensor],
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Uses only the last feature map. Accepts either spatial (B, D, H, W)
        or token (B, N, D) format.

        Args:
            features: List of feature maps (uses last one).
            output_size: Target spatial size (H, W).

        Returns:
            Segmentation logits of shape (B, num_classes, H, W).
        """
        x = features[-1]

        if x.ndim == 4:
            # Spatial format (B, D, H, W) -> (B, H*W, D)
            B, D, h, w = x.shape
            x = x.flatten(2).transpose(1, 2)  # (B, H*W, D)
        else:
            # Token format (B, N, D)
            B = x.shape[0]
            N = x.shape[1]
            h = w = int(N**0.5)

        x = self.head(x)  # (B, N, num_classes)

        # Reshape to spatial
        C = x.shape[-1]
        x = x.transpose(1, 2).reshape(B, C, h, w)

        if output_size is not None:
            x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        result: torch.Tensor = x
        return result
