# Implemented from: Fuller et al., "CROMA: Remote Sensing Representations with Contrastive
#   Radar-Optical Masked Autoencoders", NeurIPS 2023 (arXiv:2311.00566)
# Reference implementation: https://github.com/antofuller/CROMA

from __future__ import annotations

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """Convert image into patch embeddings via linear projection.

    Splits the input image into non-overlapping patches by rearranging pixels,
    then projects each flattened patch into an embedding vector using a linear
    layer. This matches the original CROMA implementation which uses
    ``nn.Linear(patch_size * patch_size * in_channels, embed_dim)``.

    Args:
        in_channels: Number of input image channels.
        embed_dim: Embedding dimension for each patch token.
        patch_size: Size of each square patch (pixels).
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: int = 8,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        pixels_per_patch = patch_size * patch_size * in_channels
        self.proj = nn.Linear(pixels_per_patch, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input image tensor of shape (B, C, H, W).

        Returns:
            Patch tokens of shape (B, num_patches, embed_dim) where
            num_patches = (H // patch_size) * (W // patch_size).
        """
        B, C, H, W = x.shape
        p = self.patch_size
        # (B, C, H, W) -> (B, num_patches, C * patch_size * patch_size)
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5)  # (B, H//p, W//p, C, p, p)
        x = x.reshape(B, (H // p) * (W // p), C * p * p)
        # Linear projection to embed_dim
        x = self.proj(x)
        return x
