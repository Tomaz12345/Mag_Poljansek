# Implemented from: Wang et al., "Towards a Unified Copernicus Foundation Model for Earth Vision",
#   arXiv:2503.11849, 2025
# Adapted from: https://github.com/zhu-xlab/Copernicus-FM/blob/main/Copernicus-FM/src/dynamic_hypernetwork.py
#   Apache License 2.0 (Copernicus-FM); modified for this project.

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dynamic_hypernetwork import SpectralPatchEmbedHypernetwork, VariablePatchEmbedHypernetwork


class DynamicSpectralPatchEmbed(nn.Module):
    """Dynamic patch embedding with spectral/variable hypernetworks.

    By default, weights are generated from wavelength + bandwidth metadata.
    If variable embeddings are provided, a variable hypernetwork is used.
    """

    def __init__(
        self,
        wv_planes: int = 128,
        kernel_size: int = 16,
        embed_dim: int = 768,
        use_variable_hypernetwork: bool = False,
        variable_embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.wv_planes = wv_planes
        self.embed_dim = embed_dim
        self.patch_size = (kernel_size, kernel_size)
        self.use_variable_hypernetwork = use_variable_hypernetwork

        self.spectral_hypernetwork = SpectralPatchEmbedHypernetwork(
            wv_planes=wv_planes,
            kernel_size=kernel_size,
            embed_dim=embed_dim,
        )

        self.variable_hypernetwork: VariablePatchEmbedHypernetwork | None = None
        if use_variable_hypernetwork:
            self.variable_hypernetwork = VariablePatchEmbedHypernetwork(
                input_dim=variable_embed_dim,
                wv_planes=wv_planes,
                kernel_size=kernel_size,
                embed_dim=embed_dim,
            )

    def forward(
        self,
        x: torch.Tensor,
        wavelengths: torch.Tensor,
        bandwidths: torch.Tensor,
        variable_embeddings: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply dynamic patch embedding.

        Args:
            x: Input image of shape (B, C, H, W).
            wavelengths: Central wavelengths in nm, shape (C,).
            bandwidths: Bandwidths in nm, shape (C,).
            variable_embeddings: Optional variable embeddings (C, D_var).

        Returns:
            Tuple of (patch_tokens, conditioning_embeddings).
        """
        weights: torch.Tensor
        bias: torch.Tensor
        conditioning: torch.Tensor

        if self.variable_hypernetwork is not None and variable_embeddings is not None:
            weights, bias, conditioning = self.variable_hypernetwork(variable_embeddings)
        else:
            weights, bias, conditioning = self.spectral_hypernetwork(wavelengths, bandwidths)

        dynamic_out = F.conv2d(
            x,
            weights,
            bias=bias,
            stride=self.kernel_size,
            padding=1,
        )

        patch_tokens = dynamic_out.flatten(2).transpose(1, 2)
        return patch_tokens, conditioning

    def get_output_shape(self, img_size: int) -> tuple[int, int]:
        """Get output grid shape for a given input size."""
        grid = (img_size + 2 - self.kernel_size) // self.kernel_size + 1
        return (grid, grid)


class StaticPatchEmbed(nn.Module):
    """Standard fixed Conv2d patch embedding (fallback)."""

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_channels: int = 13,
        embed_dim: int = 768,
        norm_layer: type[nn.Module] | None = None,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.num_patches = (img_size // patch_size) ** 2
        self.grid_size = img_size // patch_size

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

    def get_output_shape(self, img_size: int | None = None) -> tuple[int, int]:
        """Get output grid shape."""
        size = img_size if img_size is not None else self.img_size
        grid = size // self.patch_size
        return (grid, grid)
