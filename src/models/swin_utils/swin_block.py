# Implemented from: Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted
#   Windows", ICCV 2021 (arXiv:2103.14030)
# Adapted from: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
#   MIT License, Copyright (c) Microsoft Corporation; modified for this project.

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
import torch.nn as nn

from .window_attention import WindowAttention, window_partition, window_reverse


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample.

    Randomly drops entire paths during training for regularization.

    Args:
        drop_prob: Probability of dropping a path.
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        # Work with arbitrary dimensions
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # Binarize
        output = x.div(keep_prob) * random_tensor
        return output


class Mlp(nn.Module):
    """MLP module with GELU activation.

    Args:
        in_features: Number of input features.
        hidden_features: Number of hidden features. Defaults to in_features.
        out_features: Number of output features. Defaults to in_features.
        drop: Dropout rate.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block.

    Consists of window-based multi-head self-attention (W-MSA or SW-MSA),
    followed by a feed-forward network (MLP), with residual connections
    and layer normalization.

    For shifted window attention (SW-MSA), set shift_size > 0.

    Args:
        dim: Number of input channels.
        input_resolution: Input resolution (H, W).
        num_heads: Number of attention heads.
        window_size: Window size for local attention.
        shift_size: Shift size for SW-MSA. Use 0 for W-MSA.
        mlp_ratio: Ratio of MLP hidden dim to embedding dim.
        qkv_bias: If True, add learnable bias to Q, K, V.
        drop: Dropout rate.
        attn_drop: Attention dropout rate.
        drop_path: Stochastic depth rate.
        norm_layer: Normalization layer.

    Shape:
        - Input: (B, H, W, C)
        - Output: (B, H, W, C)
    """

    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        # Layer normalization
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        # Window attention
        self.attn = WindowAttention(
            dim=dim,
            window_size=(self.window_size, self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        # Stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            drop=drop,
        )

        # Create attention mask for shifted window attention
        if self.shift_size > 0:
            self._create_attention_mask()
        else:
            self.attn_mask = None

    def _create_attention_mask(self) -> None:
        """Create attention mask for shifted window partitioning."""
        h, w = self.input_resolution

        # Pad to multiples of window size
        hp = int(((h - 1) // self.window_size + 1) * self.window_size)
        wp = int(((w - 1) // self.window_size + 1) * self.window_size)

        img_mask = torch.zeros((1, hp, wp, 1))

        # Define slices for different regions
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )

        # Assign region indices
        cnt = 0
        for h_slice in h_slices:
            for w_slice in w_slices:
                img_mask[:, h_slice, w_slice, :] = cnt
                cnt += 1

        # Partition into windows
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)

        # Create attention mask
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0)
        attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0)

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, H, W, C).

        Returns:
            Output tensor of shape (B, H, W, C).
        """
        b, h, w, c = x.shape

        shortcut = x
        x = self.norm1(x)

        # Pad feature maps to multiples of window size if needed
        pad_l = pad_t = 0
        pad_r = (self.window_size - w % self.window_size) % self.window_size
        pad_b = (self.window_size - h % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
            x = torch.nn.functional.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, hp, wp, _ = x.shape

        # Cyclic shift for SW-MSA
        attn_mask: torch.Tensor | None = None
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            # Recompute attention mask for the actual padded size if different from expected
            attn_mask = self._compute_attention_mask(hp, wp) if (hp != self.input_resolution[0] or wp != self.input_resolution[1]) else self.attn_mask
        else:
            shifted_x = x

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c)

        # W-MSA or SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, hp, wp)

        # Reverse cyclic shift
        x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)) if self.shift_size > 0 else shifted_x

        # Remove padding if applied
        if pad_r > 0 or pad_b > 0:
            x = x[:, :h, :w, :].contiguous()

        # First residual connection
        x = shortcut + self.drop_path(x)

        # MLP with second residual connection
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def _compute_attention_mask(self, h: int, w: int) -> torch.Tensor:
        """Compute attention mask for given height and width.

        Args:
            h: Feature map height.
            w: Feature map width.

        Returns:
            Attention mask tensor.
        """
        img_mask = torch.zeros((1, h, w, 1), device=cast(torch.Tensor, self.norm1.weight).device)

        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )

        cnt = 0
        for h_slice in h_slices:
            for w_slice in w_slices:
                img_mask[:, h_slice, w_slice, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)

        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0)
        attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0)

        return attn_mask


class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage.

    Contains multiple Swin Transformer blocks, alternating between
    W-MSA (shift_size=0) and SW-MSA (shift_size=window_size//2).

    Args:
        dim: Number of input channels.
        input_resolution: Input resolution (H, W).
        depth: Number of Swin Transformer blocks.
        num_heads: Number of attention heads.
        window_size: Local window size.
        mlp_ratio: Ratio of MLP hidden dim.
        qkv_bias: If True, add learnable bias to Q, K, V.
        drop: Dropout rate.
        attn_drop: Attention dropout rate.
        drop_path: Stochastic depth rate (can be a list for each block).
        norm_layer: Normalization layer.
        downsample: Downsample layer at the end of the stage.

    Shape:
        - Input: (B, H, W, C)
        - Output: (B, H', W', C') where H', W', C' depend on downsample
    """

    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float | list[float] = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        downsample: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # Build blocks - alternate between W-MSA and SW-MSA
        self.blocks = nn.ModuleList()
        for i in range(depth):
            drop_path_rate = drop_path[i] if isinstance(drop_path, list) else drop_path
            block = SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path_rate,
                norm_layer=norm_layer,
            )
            self.blocks.append(block)

        # Downsample layer
        self.downsample: nn.Module | None
        if downsample is not None:
            self.downsample = downsample(
                input_resolution=input_resolution,
                dim=dim,
                norm_layer=norm_layer,
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor, return_before_downsample: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, H, W, C).
            return_before_downsample: If True, also return features before
                downsampling (for FPN/UPerNet decoder).

        Returns:
            If return_before_downsample is False:
                Output tensor. Shape depends on whether downsample is used.
            If return_before_downsample is True:
                Tuple of (output_after_downsample, features_before_downsample).
                If no downsample, both are the same tensor.
        """
        for block in self.blocks:
            x = block(x)

        features_before_downsample = x

        if self.downsample is not None:
            x = self.downsample(x)

        if return_before_downsample:
            return x, features_before_downsample
        return x
