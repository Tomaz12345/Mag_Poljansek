# Implemented from: Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted
#   Windows", ICCV 2021 (arXiv:2103.14030)
# Adapted from: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
#   MIT License, Copyright (c) Microsoft Corporation; modified for this project.

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn


class WindowAttention(nn.Module):
    """Window-based multi-head self-attention with relative position bias.

    This module computes attention within non-overlapping local windows,
    significantly reducing computational complexity from O(n^2) to O(n*w^2)
    where n is the sequence length and w is the window size.

    Args:
        dim: Number of input channels.
        window_size: Window size (H, W). Attention is computed within this window.
        num_heads: Number of attention heads.
        qkv_bias: If True, add learnable bias to Q, K, V projections.
        attn_drop: Dropout rate for attention weights.
        proj_drop: Dropout rate for output projection.

    Shape:
        - Input: (num_windows*B, window_size*window_size, C)
        - mask: (num_windows, window_size*window_size, window_size*window_size) or None
        - Output: (num_windows*B, window_size*window_size, C)

    Example:
        >>> attn = WindowAttention(dim=96, window_size=(7, 7), num_heads=3)
        >>> x = torch.randn(4, 49, 96)  # 4 windows, 7*7=49 tokens, 96 channels
        >>> out = attn(x)  # (4, 49, 96)
    """

    def __init__(
        self,
        dim: int,
        window_size: tuple[int, int],
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wh, Ww)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        # Shape: (2*Wh-1) * (2*Ww-1), num_heads
        # The relative position along each axis ranges from -(window_size-1) to (window_size-1)
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        # Compute relative position index for each token pair in the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # (2, Wh, Ww)
        coords_flatten = torch.flatten(coords, 1)  # (2, Wh*Ww)

        # Relative coordinates: (2, Wh*Ww, Wh*Ww)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (Wh*Ww, Wh*Ww, 2)

        # Shift to start from 0
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1

        # Convert to 1D index
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # (Wh*Ww, Wh*Ww)

        self.register_buffer("relative_position_index", relative_position_index)

        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features of shape (num_windows*B, N, C) where N = Wh*Ww.
            mask: Attention mask of shape (num_windows, N, N) or None.
                  Used for shifted window attention to prevent attention
                  across different regions.

        Returns:
            Output features of shape (num_windows*B, N, C).
        """
        b_, n, c = x.shape

        # QKV projection: (B_, N, C) -> (B_, N, 3*C) -> (B_, N, 3, num_heads, head_dim)
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B_, num_heads, N, head_dim)
        q, k, v = qkv.unbind(0)  # Each: (B_, num_heads, N, head_dim)

        # Scaled dot-product attention
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # (B_, num_heads, N, N)

        # Add relative position bias
        rpi = cast(torch.Tensor, self.relative_position_index)
        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )  # (Wh*Ww, Wh*Ww, num_heads)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # (num_heads, Wh*Ww, Wh*Ww)
        attn = attn + relative_position_bias.unsqueeze(0)

        # Apply attention mask for shifted windows
        if mask is not None:
            num_windows = mask.shape[0]
            attn = attn.view(b_ // num_windows, num_windows, self.num_heads, n, n)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Partition input feature map into non-overlapping windows.

    Args:
        x: Input feature map of shape (B, H, W, C).
        window_size: Window size.

    Returns:
        Windows of shape (num_windows*B, window_size, window_size, C).
    """
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, c)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, h: int, w: int) -> torch.Tensor:
    """Reverse window partition operation.

    Args:
        windows: Windows of shape (num_windows*B, window_size, window_size, C).
        window_size: Window size.
        h: Height of the feature map.
        w: Width of the feature map.

    Returns:
        Feature map of shape (B, H, W, C).
    """
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(b, h, w, -1)
    return x
