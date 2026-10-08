# Implemented from: Li et al., "FlexiMo: A Flexible Remote Sensing Foundation Model",
#   IEEE Transactions on Geoscience and Remote Sensing, 2026
# Reference implementation: https://github.com/danfenghong/IEEE_TGRS_Fleximo
# Pretrained weights and layout follow DOFA: https://github.com/zhu-xlab/DOFA

from __future__ import annotations

import torch
import torch.nn as nn


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample.

    Args:
        drop_prob: Probability of dropping a path.
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with stochastic depth."""
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class Attention(nn.Module):
    """Multi-head self-attention.

    Args:
        dim: Input/output dimension.
        num_heads: Number of attention heads.
        qkv_bias: Whether to add bias to QKV projections.
        attn_drop: Attention dropout rate.
        proj_drop: Output projection dropout rate.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 12,
        qkv_bias: bool = True,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, N, C).

        Returns:
            Output tensor of shape (B, N, C).
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """MLP block with GELU activation.

    Args:
        in_features: Input dimension.
        hidden_features: Hidden dimension.
        out_features: Output dimension (defaults to in_features).
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
        """Forward pass."""
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Standard ViT transformer block.

    Pre-norm architecture with multi-head self-attention and MLP.

    Args:
        dim: Input/output dimension.
        num_heads: Number of attention heads.
        mlp_ratio: Ratio of MLP hidden dim to input dim.
        qkv_bias: Whether to add bias to QKV projections.
        drop: Dropout rate.
        attn_drop: Attention dropout rate.
        drop_path: Stochastic depth rate.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            drop=drop,
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, N, C).

        Returns:
            Output tensor of shape (B, N, C).
        """
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class FlexiMoEncoder(nn.Module):
    """ViT encoder with multi-scale feature extraction.

    Processes patch token sequences through a stack of transformer blocks,
    extracting intermediate features at specified indices for multi-scale
    dense prediction.

    Args:
        embed_dim: Embedding dimension.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        mlp_ratio: MLP expansion ratio.
        qkv_bias: Whether to use QKV bias.
        drop_rate: Dropout rate.
        attn_drop_rate: Attention dropout rate.
        drop_path_rate: Stochastic depth rate (linearly increasing).
        out_indices: Block indices at which to extract features.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        out_indices: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth

        if out_indices is None:
            self.out_indices = tuple(depth // 4 * (i + 1) - 1 for i in range(4))
        else:
            self.out_indices = out_indices

        # Linearly increasing stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                )
                for i in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        return_final_tokens: bool = False,
    ) -> list[torch.Tensor] | tuple[list[torch.Tensor], torch.Tensor]:
        """Forward pass returning multi-scale features.

        Args:
            x: Input patch embeddings of shape (B, N, C).

        Returns:
            List of feature tensors at specified output indices,
            each of shape (B, N, C).
        """
        features: list[torch.Tensor] = []

        for i, block in enumerate(self.blocks):
            x = block(x)

            if i in self.out_indices:
                features.append(self.norm(x))

        final_tokens = self.norm(x)
        if return_final_tokens:
            return features, final_tokens
        return features

    def get_num_layers(self) -> int:
        """Return number of transformer blocks."""
        return self.depth
