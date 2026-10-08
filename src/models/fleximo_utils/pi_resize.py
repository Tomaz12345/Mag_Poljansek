# Implemented from: Beyer et al., "FlexiViT: One Model for All Patch Sizes", CVPR 2023 (PI-resize),
#   as used by Li et al., "FlexiMo: A Flexible Remote Sensing Foundation Model", IEEE TGRS 2026

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_pi_resize_matrix(base_size: int, target_size: int) -> torch.Tensor:
    """Build the PI-Resize projection matrix P = (M^T)^+.

    Constructs the bilinear interpolation matrix M that maps a base-size
    kernel to a target-size kernel, then computes P = pinv(M^T) which is
    the optimal (least-squares) resize operator that preserves token norms.

    Args:
        base_size: Spatial size of the base kernel (e.g., 16 from DOFA).
        target_size: Desired spatial size of the output kernel.

    Returns:
        Projection matrix P of shape (target_size^2, base_size^2).
    """
    # Build M column-by-column: M maps base-size flat vector to target-size
    # flat vector via bilinear interpolation.
    # M has shape (target_size^2, base_size^2).
    n_base = base_size * base_size
    n_target = target_size * target_size

    M = torch.zeros(n_target, n_base)

    for j in range(n_base):
        # Create unit basis vector at position j in base-size space
        e_j = torch.zeros(1, 1, base_size, base_size)
        row, col = j // base_size, j % base_size
        e_j[0, 0, row, col] = 1.0

        # Apply bilinear interpolation to target size
        interp = F.interpolate(
            e_j,
            size=(target_size, target_size),
            mode="bilinear",
            align_corners=False,
        )
        M[:, j] = interp.flatten()

    # P = (M^T)^+ = pinv(M^T), shape (target_size^2, base_size^2)
    # This is the projection matrix from Eq. 11 of FlexiMo paper.
    P: torch.Tensor = torch.linalg.pinv(M.T)

    return P


class PIResize(nn.Module):
    """Parameter-free kernel resizing via pseudo-inverse bilinear interpolation.

    Resizes dynamically generated convolution kernels from a pretrained base
    size to an arbitrary target patch size. The resize matrix is precomputed
    at initialization and stored as a buffer.

    When base_size == target_size, acts as identity (no-op).

    Args:
        base_size: Spatial size of kernels from the weight generator (e.g., 16).
        target_size: Desired spatial size for the output kernels.
    """

    def __init__(self, base_size: int = 16, target_size: int = 16) -> None:
        super().__init__()
        self.base_size = base_size
        self.target_size = target_size
        self.is_identity = base_size == target_size

        if not self.is_identity:
            P = build_pi_resize_matrix(base_size, target_size)
            self.register_buffer("resize_matrix", P)
        else:
            self.register_buffer(
                "resize_matrix",
                torch.eye(base_size * base_size),
            )

    def forward(self, kernel: torch.Tensor) -> torch.Tensor:
        """Resize convolution kernel from base size to target size.

        Args:
            kernel: Dynamic kernel of shape (embed_dim, C, base_size, base_size).

        Returns:
            Resized kernel of shape (embed_dim, C, target_size, target_size).
        """
        if self.is_identity:
            return kernel

        E, C, H, W = kernel.shape

        # Flatten spatial dims: (E, C, base_size^2)
        flat = kernel.reshape(E * C, H * W)

        # Apply PI-Resize: (E*C, base_size^2) @ (base_size^2, target_size^2)
        # -> (E*C, target_size^2)
        resized = flat @ cast(torch.Tensor, self.resize_matrix).T

        # Reshape back to spatial: (E, C, target_size, target_size)
        result: torch.Tensor = resized.reshape(E, C, self.target_size, self.target_size)
        return result
