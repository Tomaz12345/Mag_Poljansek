# Implemented from: Wang et al., "Towards a Unified Copernicus Foundation Model for Earth Vision",
#   arXiv:2503.11849, 2025
# Adapted from: https://github.com/zhu-xlab/Copernicus-FM/blob/main/Copernicus-FM/src/dynamic_hypernetwork.py
#   Apache License 2.0 (Copernicus-FM; contains code from DOFA, MIT, Copyright (c) 2024 Zhitong Xiong); modified for this project.

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.init as init

from ..fleximo_utils.dynamic_patch_embed import FCResLayer, TransformerWeightGenerator
from .fourier import FourierExpansion


class SpectralPatchEmbedHypernetwork(nn.Module):
    """Spectral hypernetwork for dynamic patch embedding weight generation.

    Generates Conv2d kernel weights conditioned on per-channel wavelengths
    and bandwidths via Fourier feature expansion.
    """

    def __init__(
        self,
        wv_planes: int = 128,
        kernel_size: int = 16,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.wv_planes = wv_planes
        self.embed_dim = embed_dim
        self._num_kernel = kernel_size * kernel_size * embed_dim

        # Fourier expansions for wavelengths and bandwidths (nm)
        self.spectrum_central_expansion = FourierExpansion(100, 1e9)
        self.spectrum_bandwidth_expansion = FourierExpansion(1, 1e9)

        # Residual refinement layer
        self.fclayer = FCResLayer(wv_planes)

        # Transformer-based weight generator
        self.weight_generator = TransformerWeightGenerator(
            input_dim=wv_planes,
            output_dim=self._num_kernel,
            embed_dim=embed_dim,
        )

        self.scaler = 0.01

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Xavier uniform, bias=0.01."""

        def _weight_init(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)

        self.weight_generator.apply(_weight_init)
        self.fclayer.apply(_weight_init)

    def forward(
        self,
        wavelengths: torch.Tensor,
        bandwidths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate dynamic convolution weights from spectral metadata.

        Args:
            wavelengths: Central wavelengths in nm, shape (C,).
            bandwidths: Bandwidths in nm, shape (C,).

        Returns:
            Tuple of (weights, bias, wave_embeddings).
        """
        inplanes = wavelengths.size(0)

        # Fourier-encode wavelengths and bandwidths, then sum
        emb_central = self.spectrum_central_expansion(wavelengths, self.wv_planes)
        emb_bandwidth = self.spectrum_bandwidth_expansion(bandwidths, self.wv_planes)
        waves = emb_central + emb_bandwidth

        # Refine through residual FC layer
        waves = self.fclayer(waves)

        # Generate conv kernel weights and bias
        weight, bias = self.weight_generator(waves)

        # Reshape to conv kernel: (C, K*K*E) -> (E, C, K, K)
        dynamic_weight = weight.view(inplanes, self.kernel_size, self.kernel_size, self.embed_dim)
        dynamic_weight = dynamic_weight.permute(3, 0, 1, 2)

        # Apply scaling
        weights_scaled = dynamic_weight * self.scaler
        bias_scaled = bias.view(self.embed_dim) * self.scaler

        return weights_scaled, bias_scaled, waves


class VariablePatchEmbedHypernetwork(nn.Module):
    """Variable-name-guided hypernetwork for non-spectral modalities.

    Accepts precomputed variable embeddings (e.g., from frozen language models)
    and generates patch-embedding convolution weights.
    """

    def __init__(
        self,
        input_dim: int = 768,
        wv_planes: int = 128,
        kernel_size: int = 16,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.wv_planes = wv_planes
        self.kernel_size = kernel_size
        self.embed_dim = embed_dim
        self._num_kernel = kernel_size * kernel_size * embed_dim

        self.input_proj = nn.Linear(input_dim, wv_planes)
        self.fclayer = FCResLayer(wv_planes)
        self.weight_generator = TransformerWeightGenerator(
            input_dim=wv_planes,
            output_dim=self._num_kernel,
            embed_dim=embed_dim,
        )
        self.scaler = 0.01

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize MLP/attention components with Xavier uniform."""

        def _weight_init(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)

        self.input_proj.apply(_weight_init)
        self.fclayer.apply(_weight_init)
        self.weight_generator.apply(_weight_init)

    def forward(self, variable_embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate dynamic convolution weights from variable embeddings.

        Args:
            variable_embeddings: Tensor of shape (C, input_dim).

        Returns:
            Tuple of (weights, bias, refined_embeddings).
        """
        inplanes = variable_embeddings.size(0)

        refined = self.input_proj(variable_embeddings)
        refined = self.fclayer(refined)

        weight, bias = self.weight_generator(refined)
        dynamic_weight = weight.view(inplanes, self.kernel_size, self.kernel_size, self.embed_dim)
        dynamic_weight = dynamic_weight.permute(3, 0, 1, 2)

        weights_scaled = dynamic_weight * self.scaler
        bias_scaled = bias.view(self.embed_dim) * self.scaler
        return weights_scaled, bias_scaled, refined


class AdapterLayer(nn.Module):
    """Low-rank adapter layer for efficient fine-tuning.

    Applies a low-rank transformation: x + scale * (x @ W_down @ W_up).

    Note: This is a project extension not present in the original
    Copernicus-FM paper. It can be optionally enabled for fine-tuning.

    Args:
        embed_dim: Input/output dimension.
        rank: Rank of the adapter.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        rank: int = 8,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.rank = rank

        self.down = nn.Linear(embed_dim, rank, bias=False)
        self.up = nn.Linear(rank, embed_dim, bias=False)
        self.scale = nn.Parameter(torch.ones(1) * 0.1)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize with near-zero output."""
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply adapter transformation.

        Args:
            x: Input tensor of shape (..., embed_dim).

        Returns:
            Output tensor of shape (..., embed_dim).
        """
        adapted = self.up(self.down(x))
        result: torch.Tensor = x + self.scale * adapted
        return result
