# Implemented from: Wolf et al., "Brewing Stronger Features: Dual-Teacher Distillation for
#   Multispectral Earth Observation", CVPR 2026 (arXiv:2602.19863)
# Reference implementation: https://github.com/wolfilip/DEO-FM

from __future__ import annotations

import torch
import torch.nn as nn


class DEOPatchEmbed(nn.Module):
    """Image-to-patch embedding used by the DEO ViT trunk."""

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 8,
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.proj(x).flatten(2).transpose(1, 2)
        return out
