# Implemented from: Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image
#   Recognition at Scale", ICLR 2021 (arXiv:2010.11929)
# Reference implementation: https://github.com/google-research/vision_transformer

from __future__ import annotations

import torch
import torch.nn as nn


class ViTPatchEmbed(nn.Module):
    """Split an image into patches and linearly embed each one.

    Args:
        img_size: Input image size (assumes square).
        patch_size: Patch size.
        in_channels: Number of input channels.
        embed_dim: Embedding dimension.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size**2

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Patch tokens of shape (B, num_patches, embed_dim).
        """
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)  # (B, embed_dim, N)
        x = x.transpose(1, 2)  # (B, N, embed_dim)
        return x
