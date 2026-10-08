# Implemented from: Xiong et al., "Neural Plasticity-Inspired Foundation Model for Observing the
#   Earth Crossing Modalities", arXiv:2403.15356, 2024
# Adapted from: https://github.com/zhu-xlab/DOFA/blob/master/wave_dynamic_layer.py
#   MIT License, Copyright (c) 2024 Zhitong Xiong; modified for this project.
#   Used here as the wavelength-conditioned patch embedding of FlexiMo (Li et al., IEEE TGRS 2026).

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .pi_resize import PIResize

logger = logging.getLogger(__name__)


def get_1d_sincos_pos_embed(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
    """Generate 1D sinusoidal positional embedding from positions.

    Args:
        embed_dim: Output dimension for each position (must be even).
        pos: Position values tensor of shape (M,).

    Returns:
        Positional embedding of shape (M, embed_dim).
    """
    assert embed_dim % 2 == 0, f"embed_dim must be even, got {embed_dim}"

    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega = omega / (embed_dim / 2.0)
    omega = 1.0 / (10000.0**omega)  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2)

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    """Generate 2D sinusoidal positional embedding for a grid.

    Args:
        embed_dim: Total embedding dimension (split between H and W).
        grid_size: Height/width of the grid.

    Returns:
        Positional embedding of shape (grid_size*grid_size, embed_dim).
    """
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_h, grid_w, indexing="ij")
    grid_h_flat = grid[0].reshape(-1)
    grid_w_flat = grid[1].reshape(-1)

    emb_h = get_1d_sincos_pos_embed(embed_dim // 2, grid_h_flat)
    emb_w = get_1d_sincos_pos_embed(embed_dim // 2, grid_w_flat)

    return torch.cat([emb_h, emb_w], dim=1)


class FCResLayer(nn.Module):
    """Fully-connected residual layer for wavelength embedding refinement.

    Applies a two-layer MLP with residual connection to refine
    wavelength embeddings before feeding them to the weight generator.

    Args:
        linear_size: Dimension of the FC layers.
    """

    def __init__(self, linear_size: int = 128) -> None:
        super().__init__()
        self.l_size = linear_size
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection.

        Args:
            x: Input tensor of shape (C, D).

        Returns:
            Refined tensor of shape (C, D).
        """
        y = self.w1(x)
        y = self.nonlin1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        result: torch.Tensor = x + y
        return result


class TransformerWeightGenerator(nn.Module):
    """Generates dynamic convolution weights from wavelength embeddings.

    Uses a small transformer encoder to process learnable weight tokens
    concatenated with wavelength embeddings, then projects the output
    to convolution kernel weights and bias.

    Args:
        input_dim: Dimension of wavelength embeddings.
        output_dim: Total number of weight parameters per channel
            (kernel_size * kernel_size * embed_dim).
        embed_dim: Output embedding dimension (for bias).
        num_heads: Number of attention heads in transformer.
        num_layers: Number of transformer encoder layers.
        num_weight_tokens: Number of learnable weight query tokens.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        embed_dim: int,
        num_heads: int = 4,
        num_layers: int = 1,
        num_weight_tokens: int = 128,
    ) -> None:
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            activation="gelu",
            norm_first=False,
            batch_first=False,
            dropout=0.0,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)

        self.fc_weight = nn.Linear(input_dim, output_dim)
        self.fc_bias = nn.Linear(input_dim, embed_dim)

        self.wt_num = num_weight_tokens
        self.weight_tokens = nn.Parameter(torch.empty(self.wt_num, input_dim))
        self.bias_token = nn.Parameter(torch.empty(1, input_dim))

        nn.init.normal_(self.weight_tokens, std=0.02)
        nn.init.normal_(self.bias_token, std=0.02)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate convolution weights and bias from wavelength embeddings.

        Args:
            x: Wavelength embeddings of shape (C, D) where C is number of
                input channels and D is the embedding dimension.

        Returns:
            Tuple of (weights, bias):
                - weights: Shape (C, output_dim) - per-channel kernel weights
                - bias: Shape (embed_dim,) - shared bias vector
        """
        pos_wave = x
        # Concatenate: [weight_tokens; wavelength_embeddings; bias_token]
        x = torch.cat([self.weight_tokens, pos_wave], dim=0)
        x = torch.cat([x, self.bias_token], dim=0)

        transformer_output = self.transformer_encoder(x)

        # Weight tokens are at positions [wt_num : -1] (aligned with wavelength positions)
        # Bias token is at position [-1]
        weights = self.fc_weight(transformer_output[self.wt_num : -1] + pos_wave)
        bias = self.fc_bias(transformer_output[-1])

        return weights, bias


class DynamicPatchEmbed(nn.Module):
    """Dynamic wavelength-conditioned patch embedding.

    Replaces the standard fixed Conv2d patch embedding with a
    wavelength-conditioned dynamic convolution. Generates the convolution
    kernel on-the-fly based on the central wavelengths of input bands.

    When target_patch_size differs from base_kernel_size, PI-Resize
    (pseudo-inverse bilinear interpolation) adapts the kernel from the
    pretrained base size to the target size without adding parameters.

    Args:
        wv_planes: Dimension of wavelength embeddings.
        base_kernel_size: Base kernel size from pretraining (always generates
            kernels at this size). Defaults to 16 (DOFA pretrained size).
        target_patch_size: Actual patch size for convolution stride. If None,
            uses base_kernel_size (no PI-Resize). Set to {4, 8, 16, 32, 64}
            for flexible patch sizes.
        embed_dim: Output embedding dimension.
        num_weight_tokens: Number of learnable weight query tokens.
    """

    def __init__(
        self,
        wv_planes: int = 128,
        base_kernel_size: int = 16,
        target_patch_size: int | None = None,
        embed_dim: int = 768,
        num_weight_tokens: int = 128,
    ) -> None:
        super().__init__()
        self.base_kernel_size = base_kernel_size
        self.target_patch_size = target_patch_size if target_patch_size is not None else base_kernel_size
        self.wv_planes = wv_planes
        self.embed_dim = embed_dim
        # Weight generator always produces kernels at base size
        self._num_kernel = base_kernel_size * base_kernel_size * embed_dim
        self.patch_size = (self.target_patch_size, self.target_patch_size)

        self.weight_generator = TransformerWeightGenerator(
            input_dim=wv_planes,
            output_dim=self._num_kernel,
            embed_dim=embed_dim,
        )
        self.scaler = 0.01
        self.fclayer = FCResLayer(wv_planes)

        # PI-Resize for flexible patch sizes (parameter-free)
        self.pi_resize = PIResize(
            base_size=base_kernel_size,
            target_size=self.target_patch_size,
        )
        if not self.pi_resize.is_identity:
            logger.info(f"PI-Resize enabled: {base_kernel_size}x{base_kernel_size} -> {self.target_patch_size}x{self.target_patch_size}")

        self._init_weights()

    def _init_weights(self) -> None:
        def _weight_init(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)

        self.weight_generator.apply(_weight_init)
        self.fclayer.apply(_weight_init)

    def set_target_patch_size(self, target_patch_size: int) -> None:
        """Update the target patch size at runtime.

        Recomputes the PI-Resize matrix for the new target size.
        The weight generator continues to produce kernels at base_kernel_size.
        """
        self.target_patch_size = target_patch_size
        self.patch_size = (target_patch_size, target_patch_size)
        self.pi_resize = PIResize(
            base_size=self.base_kernel_size,
            target_size=target_patch_size,
        ).to(next(self.parameters()).device)
        if not self.pi_resize.is_identity:
            logger.info(f"PI-Resize updated: {self.base_kernel_size}x{self.base_kernel_size} -> {target_patch_size}x{target_patch_size}")

    def forward(self, img: torch.Tensor, wavelengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply dynamic patch embedding.

        The weight generator produces kernels at base_kernel_size (e.g., 16x16).
        If target_patch_size differs, PI-Resize adapts the kernel before
        convolution.

        Args:
            img: Input image of shape (B, C, H, W).
            wavelengths: Central wavelengths in micrometers, shape (C,).

        Returns:
            Tuple of (patch_tokens, wave_embeddings):
                - patch_tokens: Shape (B, N, embed_dim) where
                  N = (H/target_patch_size) * (W/target_patch_size)
                - wave_embeddings: Shape (C, wv_planes) refined wavelength embeddings
        """
        inplanes = wavelengths.size(0)

        # Encode wavelengths: scale by 1000 and apply sincos encoding
        waves = get_1d_sincos_pos_embed(self.wv_planes, wavelengths * 1000)
        waves = self.fclayer(waves)

        # Generate dynamic convolution weights at base kernel size
        weight, bias = self.weight_generator(waves)

        # Reshape to convolution kernel: (embed_dim, C, base_K, base_K)
        dynamic_weight = weight.view(inplanes, self.base_kernel_size, self.base_kernel_size, self.embed_dim)
        dynamic_weight = dynamic_weight.permute(3, 0, 1, 2)  # (E, C, base_K, base_K)

        if bias is not None:
            bias = bias.view(self.embed_dim) * self.scaler

        weights = dynamic_weight * self.scaler

        # PI-Resize: adapt kernel from base size to target patch size
        weights = self.pi_resize(weights)

        # Apply convolution with stride=target_patch_size (non-overlapping patches)
        dynamic_out = F.conv2d(img, weights, bias=bias, stride=self.target_patch_size, padding=0)

        # Reshape to sequence: (B, E, H', W') -> (B, N, E)
        x = dynamic_out.flatten(2).transpose(1, 2)

        return x, waves
