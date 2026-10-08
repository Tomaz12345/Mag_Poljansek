# Implemented from: Wang et al., "UNetFormer: A UNet-like transformer for efficient semantic
#   segmentation of remote sensing urban scene imagery", ISPRS J. Photogramm. Remote Sens. 190, 2022
# Adapted from: https://github.com/WangLibo1995/GeoSeg/blob/main/geoseg/models/UNetFormer.py
#   GNU GPL v3.0, Libo Wang; modified for this project.
#   Licence text: LICENSE

from __future__ import annotations

import torch.nn as nn


class Conv(nn.Sequential):
    """Convolution with automatic padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        stride: int = 1,
        bias: bool = False,
    ) -> None:
        padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                bias=bias,
                dilation=dilation,
                stride=stride,
                padding=padding,
            ),
        )


class ConvBN(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        stride: int = 1,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
        bias: bool = False,
    ) -> None:
        padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                bias=bias,
                dilation=dilation,
                stride=stride,
                padding=padding,
            ),
            norm_layer(out_channels),
        )


class ConvBNReLU(nn.Sequential):
    """Convolution + BatchNorm + ReLU6."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        stride: int = 1,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
        bias: bool = False,
    ) -> None:
        padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                bias=bias,
                dilation=dilation,
                stride=stride,
                padding=padding,
            ),
            norm_layer(out_channels),
            nn.ReLU6(),
        )


class SeparableConvBN(nn.Sequential):
    """Depthwise separable convolution + BatchNorm.

    Depthwise conv (groups=in_channels) followed by BatchNorm and pointwise 1x1 conv.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
    ) -> None:
        padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                groups=in_channels,
                bias=False,
            ),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        )
