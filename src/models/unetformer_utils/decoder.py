# Implemented from: Wang et al., "UNetFormer: A UNet-like transformer for efficient semantic
#   segmentation of remote sensing urban scene imagery", ISPRS J. Photogramm. Remote Sens. 190, 2022
# Adapted from: https://github.com/WangLibo1995/GeoSeg/blob/main/geoseg/models/UNetFormer.py
#   GNU GPL v3.0, Libo Wang; modified for this project.
#   Licence text: LICENSE

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv_modules import Conv, ConvBN, ConvBNReLU
from .frh import WF, FeatureRefinementHead
from .gltb import Block


class AuxHead(nn.Module):
    """Auxiliary segmentation head for deep supervision.

    Takes fused features from GLTB stages and produces auxiliary predictions.
    Architecture: ConvBNReLU 3x3 → Dropout → Conv 1x1 → bilinear upsample.

    Args:
        in_channels: Number of input channels.
        num_classes: Number of output classes.
    """

    def __init__(self, in_channels: int = 64, num_classes: int = 8) -> None:
        super().__init__()
        self.conv = ConvBNReLU(in_channels, in_channels)
        self.drop = nn.Dropout(0.1)
        self.conv_out = Conv(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input feature of shape (B, C, H', W').
            h: Target output height.
            w: Target output width.

        Returns:
            Auxiliary predictions of shape (B, num_classes, h, w).
        """
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        result: torch.Tensor = F.interpolate(feat, size=(h, w), mode="bilinear", align_corners=False)
        return result


class UNetFormerDecoder(nn.Module):
    """Full UNetFormer decoder matching the original paper architecture.

    4-stage decoder with progressive upsampling:
    - Stage 4: ConvBN 1x1 projection → Block (GLTB at bottleneck)
    - Stage 3: WF (weighted fusion with encoder skip) → Block
    - Stage 2: WF → Block
    - Stage 1: FeatureRefinementHead (FRH, no Block)

    During training, returns (main_output, aux_output) for deep supervision.
    During eval, returns main_output only.

    Args:
        encoder_channels: Tuple of encoder output channels per stage [c1, c2, c3, c4].
        decoder_channels: Unified decoder channel dimension.
        num_classes: Number of output segmentation classes.
        num_heads: Number of attention heads for GLTB blocks.
        window_size: Window size for attention.
        mlp_ratio: MLP expansion ratio in GLTB.
        drop_rate: Dropout rate.
        attn_drop_rate: Attention dropout rate.
        drop_path_rate: Stochastic depth rate.
    """

    def __init__(
        self,
        encoder_channels: tuple[int, ...] = (64, 128, 256, 512),
        decoder_channels: int = 64,
        num_classes: int = 10,
        num_heads: int = 8,
        window_size: int = 8,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()

        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels
        self.num_classes = num_classes

        # Stage 4 (bottleneck): project encoder output → Block
        self.pre_conv = ConvBN(encoder_channels[3], decoder_channels, kernel_size=1)
        self.b4 = Block(
            dim=decoder_channels,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=drop_path_rate,
        )

        # Stage 3: WF + Block (skip from encoder stage 3)
        self.p3 = WF(encoder_channels[2], decoder_channels)
        self.b3 = Block(
            dim=decoder_channels,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=drop_path_rate,
        )

        # Stage 2: WF + Block (skip from encoder stage 2)
        self.p2 = WF(encoder_channels[1], decoder_channels)
        self.b2 = Block(
            dim=decoder_channels,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=drop_path_rate,
        )

        # Stage 1: FRH (skip from encoder stage 1, no Block)
        self.p1 = FeatureRefinementHead(encoder_channels[0], decoder_channels)

        # Auxiliary head for deep supervision (training only)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.aux_head = AuxHead(decoder_channels, num_classes)

        # Segmentation head: ConvBNReLU 3x3 → Dropout2d → Conv 1x1
        self.segmentation_head = nn.Sequential(
            ConvBNReLU(decoder_channels, decoder_channels),
            nn.Dropout2d(p=drop_rate, inplace=True),
            Conv(decoder_channels, num_classes, kernel_size=1),
        )

        self._init_weight()

    def _init_weight(self) -> None:
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        encoder_features: list[torch.Tensor],
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            encoder_features: List of encoder features [f1, f2, f3, f4] from
                              stages 1-4 with decreasing spatial resolution.
            output_size: Desired output size (H, W). If None, uses input size
                        inferred from encoder features.

        Returns:
            During training: tuple of (main_output, aux_output).
            During eval: main_output only.
            Both of shape (B, num_classes, H, W).
        """
        f1, f2, f3, f4 = encoder_features

        # Determine target output size
        if output_size is None:
            h, w = f1.shape[2] * 4, f1.shape[3] * 4
        else:
            h, w = output_size

        if self.training:
            # Stage 4: bottleneck
            x = self.b4(self.pre_conv(f4))
            h4 = x  # Save for aux

            # Stage 3: WF fusion + Block
            x = self.p3(x, f3)
            x = self.b3(x)
            h3 = x  # Save for aux

            # Stage 2: WF fusion + Block
            x = self.p2(x, f2)
            x = self.b2(x)
            h2 = x  # Save for aux (target size for fusion)

            # Stage 1: FRH
            x = self.p1(x, f1)

            # Main segmentation head
            x = self.segmentation_head(x)
            main_out: torch.Tensor = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)

            # Auxiliary head: upsample stage features to h2's resolution and fuse
            h4_up = F.interpolate(h4, size=h2.shape[2:], mode="bilinear", align_corners=False)
            h3_up = F.interpolate(h3, size=h2.shape[2:], mode="bilinear", align_corners=False)
            ah = h4_up + h3_up + h2
            aux_out = self.aux_head(ah, h, w)

            return main_out, aux_out
        else:
            # Stage 4: bottleneck
            x = self.b4(self.pre_conv(f4))

            # Stage 3: WF fusion + Block
            x = self.p3(x, f3)
            x = self.b3(x)

            # Stage 2: WF fusion + Block
            x = self.p2(x, f2)
            x = self.b2(x)

            # Stage 1: FRH
            x = self.p1(x, f1)

            # Segmentation head
            x = self.segmentation_head(x)
            out: torch.Tensor = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)

            return out
