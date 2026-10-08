# Implemented from: Fuller et al., "CROMA: Remote Sensing Representations with Contrastive
#   Radar-Optical Masked Autoencoders", NeurIPS 2023 (arXiv:2311.00566)
# Reference implementation: https://github.com/antofuller/CROMA

from __future__ import annotations

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation.

    Args:
        dim: Input/output dimension.
        hidden_dim: Hidden layer dimension.
        drop_rate: Dropout probability.
    """

    def __init__(self, dim: int, hidden_dim: int, drop_rate: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SelfAttention(nn.Module):
    """Multi-head self-attention with optional ALiBi bias.

    Args:
        dim: Embedding dimension.
        num_heads: Number of attention heads.
        qkv_bias: If True, add bias to Q/K/V projections.
        attn_drop: Attention dropout probability.
        proj_drop: Output projection dropout probability.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 16,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, alibi_bias: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, N, D).
            alibi_bias: Optional ALiBi bias of shape (num_heads, N, N).

        Returns:
            Output tensor of shape (B, N, D).
        """
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)

        # Add ALiBi positional bias
        if alibi_bias is not None:
            attn = attn + alibi_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    """Multi-head cross-attention with optional X-ALiBi bias.

    Query comes from one modality, Key/Value from another.

    Args:
        dim: Embedding dimension.
        num_heads: Number of attention heads.
        qkv_bias: If True, add bias to projections.
        attn_drop: Attention dropout probability.
        proj_drop: Output projection dropout probability.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 16,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        alibi_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            query: Query tensor of shape (B, N_q, D).
            context: Context tensor (Key/Value source) of shape (B, N_kv, D).
            alibi_bias: Optional X-ALiBi bias of shape (num_heads, N_q, N_kv).

        Returns:
            Output tensor of shape (B, N_q, D).
        """
        B, N_q, D = query.shape
        N_kv = context.shape[1]

        q = self.q_proj(query).reshape(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context).reshape(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).reshape(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if alibi_bias is not None:
            attn = attn + alibi_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N_q, D)
        x = self.proj(x)
        result: torch.Tensor = self.proj_drop(x)
        return result


class SelfAttentionBlock(nn.Module):
    """Transformer block with self-attention, LayerNorm, and FFN.

    Uses pre-norm architecture (LayerNorm before attention and FFN).

    Args:
        dim: Embedding dimension.
        num_heads: Number of attention heads.
        mlp_ratio: Ratio of FFN hidden dimension to embedding dimension.
        qkv_bias: If True, add bias to Q/K/V projections.
        drop_rate: Dropout probability.
        attn_drop_rate: Attention dropout probability.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(
            dim=dim,
            hidden_dim=int(dim * mlp_ratio),
            drop_rate=drop_rate,
        )

    def forward(self, x: torch.Tensor, alibi_bias: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), alibi_bias)
        x = x + self.ffn(self.norm2(x))
        return x


class CrossAttentionBlock(nn.Module):
    """Transformer block with self-attention, cross-attention, and FFN.

    Used in the cross-encoder to fuse SAR and optical modalities.
    Architecture: self-attn -> cross-attn -> FFN (all with pre-norm and residual).

    Uses a shared LayerNorm for both query and context in cross-attention,
    matching the original CROMA implementation.

    Args:
        dim: Embedding dimension.
        num_heads: Number of attention heads.
        mlp_ratio: Ratio of FFN hidden dimension to embedding dimension.
        qkv_bias: If True, add bias to projections.
        drop_rate: Dropout probability.
        attn_drop_rate: Attention dropout probability.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        # Self-attention on query modality
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = SelfAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
        )
        # Cross-attention: query attends to context modality
        # Shared norm for both query and context (matches original CROMA)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
        )
        # Feed-forward
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = FeedForward(
            dim=dim,
            hidden_dim=int(dim * mlp_ratio),
            drop_rate=drop_rate,
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        self_alibi_bias: torch.Tensor | None = None,
        cross_alibi_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Query tensor of shape (B, N, D).
            context: Context tensor from other modality of shape (B, N, D).
            self_alibi_bias: ALiBi bias for self-attention.
            cross_alibi_bias: X-ALiBi bias for cross-attention.

        Returns:
            Output tensor of shape (B, N, D).
        """
        # Self-attention
        x = x + self.self_attn(self.norm1(x), self_alibi_bias)
        # Cross-attention (shared norm for both query and context)
        x = x + self.cross_attn(self.norm2(x), self.norm2(context), cross_alibi_bias)
        # FFN
        x = x + self.ffn(self.norm3(x))
        return x
