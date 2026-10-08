# author: Alexandre Milesi (https://github.com/milesial/Pytorch-UNet)
# milesial/Pytorch-UNet is licensed under the GNU General Public License v3.0

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Double convolution block: (Conv3x3 -> BN -> ReLU) x 2.

    This is the basic building block of U-Net, used in both encoder and decoder.
    Uses padded convolutions to maintain spatial dimensions.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        mid_channels: Number of channels after first conv. If None, uses out_channels.

    Shape:
        - Input: (N, in_channels, H, W)
        - Output: (N, out_channels, H, W)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int | None = None,
    ) -> None:
        super().__init__()

        if mid_channels is None:
            mid_channels = out_channels

        self.double_conv: nn.Module = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return cast(torch.Tensor, self.double_conv(x))


class Down(nn.Module):
    """Downsampling block: MaxPool2d -> DoubleConv.

    Reduces spatial dimensions by factor of 2 while increasing channels.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.

    Shape:
        - Input: (N, in_channels, H, W)
        - Output: (N, out_channels, H/2, W/2)
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.maxpool_conv: nn.Module = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return cast(torch.Tensor, self.maxpool_conv(x))


class Up(nn.Module):
    """Upsampling block: Upsample -> Concatenate -> DoubleConv.

    Increases spatial dimensions by factor of 2 while reducing channels.
    Concatenates with skip connection from encoder.

    Args:
        in_channels: Number of input channels (upsampled + skip connection).
        out_channels: Number of output channels.
        bilinear: If True, use bilinear upsampling. If False, use transposed conv.

    Shape:
        - Input x1: (N, C1, H, W) - feature map to upsample
        - Input x2: (N, C2, 2H, 2W) - skip connection from encoder
        - Output: (N, out_channels, 2H, 2W)

    Note:
        in_channels should equal C1 + C2 (after upsampling x1).
    """

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = False, x1_channels: int | None = None) -> None:
        super().__init__()

        # Store for potential debugging
        self.bilinear = bilinear

        # x1_channels is the channel count of the deeper feature map (before concat)
        # If not provided, assume symmetric split (legacy behavior)
        if x1_channels is None:
            x1_channels = in_channels // 2

        if bilinear:
            # Bilinear upsampling doesn't change channel count
            self.up: nn.Module = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            # DoubleConv reduces channels
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            # Original U-Net style up-conv: upsample and halve channels before concat.
            if x1_channels % 2 != 0:
                raise ValueError(f"x1_channels must be even for transposed up-conv, got {x1_channels}")
            self.up = nn.ConvTranspose2d(x1_channels, x1_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Forward pass with skip connection.

        Args:
            x1: Feature map from deeper layer (to be upsampled).
            x2: Skip connection from corresponding encoder layer.

        Returns:
            Concatenated and convolved feature map.
        """
        x1 = cast(torch.Tensor, self.up(x1))

        # Handle potential size mismatch due to pooling/upsampling
        # Input is CHW format
        diff_y: int = x2.size(2) - x1.size(2)
        diff_x: int = x2.size(3) - x1.size(3)

        # Pad x1 to match x2 dimensions
        x1 = F.pad(
            x1,
            [
                diff_x // 2,
                diff_x - diff_x // 2,
                diff_y // 2,
                diff_y - diff_y // 2,
            ],
        )

        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return cast(torch.Tensor, self.conv(x))


class OutConv(nn.Module):
    """Output convolution: 1x1 conv for final classification.

    Maps features to class predictions.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels (number of classes).

    Shape:
        - Input: (N, in_channels, H, W)
        - Output: (N, out_channels, H, W)
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv: nn.Conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return cast(torch.Tensor, self.conv(x))
