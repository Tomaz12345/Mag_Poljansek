# Implemented from: Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted
#   Windows", ICCV 2021 (arXiv:2103.14030)
# Adapted from: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
#   MIT License, Copyright (c) Microsoft Corporation; modified for this project.

from __future__ import annotations

import torch
import torch.nn as nn


class PatchMerging(nn.Module):
    """Patch Merging Layer.

    Merges 2x2 neighboring patches into a single patch, reducing spatial
    resolution by 2x while expanding the channel dimension by 2x.

    This is the downsampling operation used between Swin Transformer stages.

    Args:
        input_resolution: Resolution of input feature map (H, W).
        dim: Number of input channels.
        norm_layer: Normalization layer.

    Shape:
        - Input: (B, H, W, C)
        - Output: (B, H/2, W/2, 2*C)

    Example:
        >>> merge = PatchMerging(input_resolution=(64, 64), dim=96)
        >>> x = torch.randn(1, 64, 64, 96)
        >>> out = merge(x)  # (1, 32, 32, 192)
    """

    def __init__(
        self,
        input_resolution: tuple[int, int],
        dim: int,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim

        # Linear projection: 4*C -> 2*C
        # We concatenate 4 neighboring patches (2x2), then project
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, H, W, C).

        Returns:
            Output tensor of shape (B, H/2, W/2, 2*C).
        """
        h, w = self.input_resolution
        b, h_, w_, c = x.shape

        # Ensure dimensions are even for clean merging
        # Pad if necessary
        pad_h = (2 - h_ % 2) % 2
        pad_w = (2 - w_ % 2) % 2
        if pad_h > 0 or pad_w > 0:
            x = nn.functional.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            h_ = h_ + pad_h
            w_ = w_ + pad_w

        # Extract 2x2 patches
        x0 = x[:, 0::2, 0::2, :]  # Top-left
        x1 = x[:, 1::2, 0::2, :]  # Bottom-left
        x2 = x[:, 0::2, 1::2, :]  # Top-right
        x3 = x[:, 1::2, 1::2, :]  # Bottom-right

        # Concatenate along channel dimension
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # (B, H/2, W/2, 4*C)

        # Normalize and project
        x = self.norm(x)
        x = self.reduction(x)  # (B, H/2, W/2, 2*C)

        return x
