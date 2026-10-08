# Implemented from: Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision",
#   arXiv:2304.07193, 2023
# Reference implementation: https://github.com/facebookresearch/dinov2

from __future__ import annotations

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """Convert image patches to token embeddings.

    Args:
        img_size: Input image size (assumes square).
        patch_size: Patch size.
        in_channels: Number of input channels.
        embed_dim: Embedding dimension.
        norm_layer: Optional normalization layer.
    """

    def __init__(
        self,
        img_size: int = 252,
        patch_size: int = 14,
        in_channels: int = 13,
        embed_dim: int = 384,
        norm_layer: type[nn.Module] | None = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size**2

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

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
        x = self.norm(x)
        return x
