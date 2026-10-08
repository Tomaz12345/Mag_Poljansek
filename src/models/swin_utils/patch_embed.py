# Implemented from: Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted
#   Windows", ICCV 2021 (arXiv:2103.14030)
# Adapted from: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
#   MIT License, Copyright (c) Microsoft Corporation; modified for this project.

from __future__ import annotations

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """Image to Patch Embedding using Conv2d.

    Splits the input image into non-overlapping patches and projects them
    to the embedding dimension.

    Args:
        img_size: Input image size (assumes square images).
        patch_size: Size of each patch (assumes square patches).
        in_chans: Number of input channels.
        embed_dim: Embedding dimension for each patch.
        norm_layer: Normalization layer to apply after projection.
            If None, no normalization is applied.

    Shape:
        - Input: (B, C, H, W)
        - Output: (B, embed_dim, H/patch_size, W/patch_size)

    Example:
        >>> patch_embed = PatchEmbed(img_size=256, patch_size=4, in_chans=13, embed_dim=96)
        >>> x = torch.randn(2, 13, 256, 256)
        >>> out = patch_embed(x)  # (2, 96, 64, 64)
    """

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
        norm_layer: type[nn.Module] | None = None,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        # Number of patches in each dimension
        self.patches_resolution = img_size // patch_size
        self.num_patches = self.patches_resolution**2

        # Convolutional projection
        # kernel_size and stride equal to patch_size creates non-overlapping patches
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # Optional normalization
        self.norm: nn.Module | None
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Patch embeddings of shape (B, embed_dim, H/patch_size, W/patch_size).
        """
        # Project patches: (B, C, H, W) -> (B, embed_dim, H/P, W/P)
        x = self.proj(x)

        # Apply normalization if specified
        if self.norm is not None:
            # Reshape for LayerNorm: (B, C, H, W) -> (B, H, W, C) -> norm -> (B, C, H, W)
            b, c, h, w = x.shape
            x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
            x = self.norm(x)
            x = x.permute(0, 3, 1, 2)  # (B, C, H, W)

        return x
