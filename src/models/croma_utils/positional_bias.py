# Implemented from: Fuller et al., "CROMA: Remote Sensing Representations with Contrastive
#   Radar-Optical Masked Autoencoders", NeurIPS 2023 (arXiv:2311.00566)
# Reference implementation: https://github.com/antofuller/CROMA

from __future__ import annotations

import math

import torch


def get_2d_alibi(
    num_heads: int,
    num_patches_h: int,
    num_patches_w: int,
) -> torch.Tensor:
    """Generate 2D ALiBi position bias matrix.

    Computes L2 (Euclidean) distances between all pairs of 2D patch positions,
    then scales by per-head slopes following the ALiBi geometric progression.

    Args:
        num_heads: Number of attention heads.
        num_patches_h: Number of patches along height.
        num_patches_w: Number of patches along width.

    Returns:
        Bias tensor of shape (num_heads, num_patches, num_patches) where
        num_patches = num_patches_h * num_patches_w. Values are non-positive
        (distances are negated so closer patches have less penalty).
    """
    # Create 2D coordinates for each patch
    coords_h = torch.arange(num_patches_h)
    coords_w = torch.arange(num_patches_w)
    grid_h, grid_w = torch.meshgrid(coords_h, coords_w, indexing="ij")
    coords = torch.stack([grid_h.flatten(), grid_w.flatten()], dim=-1).float()

    # Compute pairwise L2 (Euclidean) distances: (num_patches, num_patches)
    distances = torch.cdist(coords, coords, p=2)

    # Generate per-head slopes using geometric progression
    # Slopes decrease geometrically: 2^(-8/num_heads), 2^(-16/num_heads), ...
    slopes = _get_slopes(num_heads)

    # Shape: (num_heads, 1, 1) * (1, num_patches, num_patches) -> (num_heads, N, N)
    slopes_tensor = torch.tensor(slopes, dtype=torch.float32).view(num_heads, 1, 1)
    alibi = -slopes_tensor * distances.unsqueeze(0)

    return alibi


def _get_slopes(num_heads: int) -> list[float]:
    """Get per-head slopes for ALiBi.

    Uses geometric progression: if num_heads is a power of 2, slopes are
    2^(-8/n), 2^(-16/n), ..., 2^(-8). For non-power-of-2, uses recursive
    interleaving matching the original CROMA/ALiBi implementation.

    Args:
        num_heads: Number of attention heads.

    Returns:
        List of slope values, one per head.
    """

    def _get_slopes_power_of_2(n: int) -> list[float]:
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * (ratio**i) for i in range(n)]

    if math.log2(num_heads).is_integer():
        return _get_slopes_power_of_2(num_heads)

    # For non-power-of-2, use recursive interleaving
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    return _get_slopes_power_of_2(closest_power_of_2) + _get_slopes(2 * closest_power_of_2)[0::2][: num_heads - closest_power_of_2]
