# Implemented from: Fuller et al., "CROMA: Remote Sensing Representations with Contrastive
#   Radar-Optical Masked Autoencoders", NeurIPS 2023 (arXiv:2311.00566)
# Reference implementation: https://github.com/antofuller/CROMA

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import CrossAttentionBlock, SelfAttentionBlock


class CROMAEncoder(nn.Module):
    """Vision Transformer encoder for a single modality.

    Consists of a stack of self-attention blocks with optional ALiBi bias.
    Can optionally output intermediate features for multi-scale decoding.

    Args:
        embed_dim: Embedding dimension.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        mlp_ratio: Ratio of FFN hidden dim to embed_dim.
        qkv_bias: If True, add bias to Q/K/V projections.
        drop_rate: Dropout probability.
        attn_drop_rate: Attention dropout probability.
        out_indices: Optional tuple of block indices to output features from
            (for multi-scale decoding). If None, only final output is stored.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        out_indices: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.out_indices = out_indices

        self.blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        alibi_bias: torch.Tensor | None = None,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Forward pass through transformer blocks.

        Args:
            x: Input tokens of shape (B, N, D).
            alibi_bias: Optional 2D-ALiBi bias of shape (num_heads, N, N).

        Returns:
            If out_indices is None: final features of shape (B, N, D).
            If out_indices is set: list of intermediate features.
        """
        intermediate: dict[int, torch.Tensor] = {}

        for i, block in enumerate(self.blocks):
            x = block(x, alibi_bias)
            if self.out_indices is not None and i in self.out_indices:
                intermediate[i] = self.norm(x)

        if self.out_indices is not None:
            # Preserve order and handle repeated indices
            features = [intermediate[i] for i in self.out_indices]
            return features

        x = self.norm(x)
        return x


class CrossEncoder(nn.Module):
    """Cross-modal encoder for fusing SAR and optical modalities.

    Uses cross-attention blocks where SAR attends to optical and vice versa,
    producing joint multimodal representations. Can optionally output
    intermediate features for multi-scale decoding.

    Args:
        embed_dim: Embedding dimension.
        depth: Number of cross-attention blocks.
        num_heads: Number of attention heads.
        mlp_ratio: Ratio of FFN hidden dim to embed_dim.
        qkv_bias: If True, add bias to projections.
        drop_rate: Dropout probability.
        attn_drop_rate: Attention dropout probability.
        out_indices: Optional tuple of block indices to output features from
            (for multi-scale decoding). If None, only final output is returned.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        depth: int = 6,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        out_indices: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.out_indices = out_indices

        self.blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        sar_tokens: torch.Tensor,
        optical_tokens: torch.Tensor,
        self_alibi_bias: torch.Tensor | None = None,
        cross_alibi_bias: torch.Tensor | None = None,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Forward pass through cross-attention blocks.

        SAR tokens attend to optical tokens through cross-attention,
        building a fused representation.

        Args:
            sar_tokens: SAR encoder output of shape (B, N, D).
            optical_tokens: Optical encoder output of shape (B, N, D).
            self_alibi_bias: ALiBi bias for self-attention.
            cross_alibi_bias: X-ALiBi bias for cross-attention.

        Returns:
            If out_indices is None: fused multimodal features of shape (B, N, D).
            If out_indices is set: list of intermediate features.
        """
        intermediate: dict[int, torch.Tensor] = {}

        x = sar_tokens
        for i, block in enumerate(self.blocks):
            x = block(x, optical_tokens, self_alibi_bias, cross_alibi_bias)
            if self.out_indices is not None and i in self.out_indices:
                intermediate[i] = x  # store raw, un-normed

        # Apply norm once at the end (matching original CROMA norm_out)
        x = self.norm(x)

        if self.out_indices is not None:
            # Reuse the same norm module for intermediates; replace last with
            # already-normed final output to avoid a redundant norm call.
            out_idx_list = list(self.out_indices)
            features = [self.norm(intermediate[i]) for i in out_idx_list[:-1]]
            features.append(x)
            return features

        result: torch.Tensor = x
        return result
