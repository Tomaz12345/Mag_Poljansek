# Implemented from: Fuller et al., "CROMA: Remote Sensing Representations with Contrastive
#   Radar-Optical Masked Autoencoders", NeurIPS 2023 (arXiv:2311.00566)
# Reference implementation: https://github.com/antofuller/CROMA
# Module and parameter names follow the reference implementation so that the official checkpoints load.

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

import torch
import torch.nn as nn

from .croma_utils import (
    CROMAEncoder,
    CrossEncoder,
    FPNSegDecoder,
    LinearSegDecoder,
    PatchEmbed,
    UPerNetSegDecoder,
    get_2d_alibi,
)
from .pretrained_loading import log_pretrained_load_result

logger = logging.getLogger(__name__)


# Model variant configurations
CROMA_VARIANTS: dict[str, dict[str, Any]] = {
    "base": {
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 16,
        "patch_size": 8,
    },
    "large": {
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "patch_size": 8,
    },
}


class CROMA(nn.Module):
    """CROMA model for semantic segmentation.

    Dual-encoder architecture with cross-modal fusion:
    1. SAR encoder processes Sentinel-1 data (depth // 2 layers)
    2. Optical encoder processes Sentinel-2 data (full depth layers)
    3. Cross encoder fuses both modalities (depth // 2 layers)
    4. Segmentation decoder produces per-pixel class predictions

    For fine-tuning on segmentation tasks, the model concatenates input
    channels and uses a single forward path through the optical encoder
    (when only optical data) or the full dual-encoder + cross-encoder
    pipeline (when both modalities are available).

    Args:
        n_channels: Total number of input channels.
        n_channels_sar: Number of SAR channels (typically 2 for S1 VV/VH).
        n_channels_optical: Number of optical channels (typically 4-13 for S2).
        n_classes: Number of output segmentation classes.
        img_size: Input image size (assumes square images).
        patch_size: Patch size for patch embedding.
        embed_dim: Embedding dimension.
        depth: Total encoder depth. SAR uses depth//2, optical uses depth,
            cross-encoder uses depth//2.
        num_heads: Number of attention heads.
        mlp_ratio: Ratio of FFN hidden dim to embed_dim.
        qkv_bias: If True, add learnable bias to Q/K/V projections.
        drop_rate: Dropout rate.
        attn_drop_rate: Attention dropout rate.
        use_alibi: If True, use 2D-ALiBi and X-ALiBi positional biases.
        decoder_type: Segmentation decoder type ('upernet', 'fpn', 'linear').
        decoder_channels: Decoder channel dimension.
        decoder_dropout: Decoder dropout rate.
        use_pretrained: Whether to load pretrained weights.
        pretrained_path: Path to pretrained checkpoint file.
        freeze_encoder: Whether to freeze all encoder weights.
        freeze_encoder_layers: Number of individual encoder layers to freeze.
        freeze_encoder_layers_mode: Freezing mode ('individual' or 'grouped').
        variant: Model variant shortcut ('base' or 'large').

    Example:
        >>> # CROMA-Base with both S1 and S2
        >>> model = CROMA(
        ...     n_channels=15, n_channels_sar=2, n_channels_optical=13,
        ...     n_classes=10, variant='base'
        ... )
        >>> x = torch.randn(2, 15, 120, 120)
        >>> out = model(x)  # (2, 10, 120, 120)
    """

    def __init__(
        self,
        n_channels: int,
        n_channels_sar: int,
        n_channels_optical: int,
        n_classes: int,
        img_size: int,
        patch_size: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        drop_rate: float,
        attn_drop_rate: float,
        use_alibi: bool,
        decoder_type: str,
        decoder_channels: int,
        decoder_dropout: float,
        use_pretrained: bool,
        pretrained_path: str | None,
        freeze_encoder: bool,
        freeze_encoder_layers: int,
        freeze_encoder_layers_mode: str,
        croma_drop_optical_cirrus: bool,
        variant: str | None,
    ) -> None:
        super().__init__()

        # Apply variant configuration
        if variant is not None and variant in CROMA_VARIANTS:
            config = CROMA_VARIANTS[variant]
            embed_dim = config["embed_dim"]
            depth = config["depth"]
            num_heads = config["num_heads"]
            patch_size = config["patch_size"]
            logger.info(f"Using CROMA variant '{variant}': {config}")

        self.n_channels = n_channels
        self.n_channels_sar = n_channels_sar
        self.n_channels_optical = n_channels_optical
        self.n_classes = n_classes
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.use_alibi = use_alibi
        self.decoder_type = decoder_type
        self.croma_drop_optical_cirrus = croma_drop_optical_cirrus
        self.has_sar = n_channels_sar > 0
        self.has_optical = n_channels_optical > 0
        self.has_both = self.has_sar and self.has_optical

        # Calculate grid dimensions
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        # Patch embeddings (separate for each modality)
        if self.has_sar:
            self.sar_patch_embed = PatchEmbed(
                in_channels=n_channels_sar,
                embed_dim=embed_dim,
                patch_size=patch_size,
            )

        if self.has_optical:
            self.optical_patch_embed = PatchEmbed(
                in_channels=n_channels_optical,
                embed_dim=embed_dim,
                patch_size=patch_size,
            )

        # Compute out_indices for multi-scale features (4 evenly spaced)
        sar_depth = depth // 2
        optical_depth = depth
        cross_depth = depth // 2

        optical_out_indices = self._compute_out_indices(optical_depth, 4)

        # SAR encoder (half depth)
        if self.has_sar:
            self.sar_encoder = CROMAEncoder(
                embed_dim=embed_dim,
                depth=sar_depth,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
            )

        # Optical encoder (full depth)
        if self.has_optical:
            self.optical_encoder = CROMAEncoder(
                embed_dim=embed_dim,
                depth=optical_depth,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                out_indices=optical_out_indices if not self.has_both else None,
            )

        # Cross encoder (only when both modalities available)
        # out_indices enables multi-scale feature extraction for segmentation.
        if self.has_both:
            cross_out_indices = self._compute_out_indices(cross_depth, 4)
            self.cross_encoder = CrossEncoder(
                embed_dim=embed_dim,
                depth=cross_depth,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                out_indices=cross_out_indices,
            )

        # ALiBi bias (computed once and cached as buffer)
        if use_alibi:
            alibi = get_2d_alibi(num_heads, self.grid_size, self.grid_size)
            self.register_buffer("alibi_bias", alibi)
        else:
            self.alibi_bias: torch.Tensor | None = None

        # Segmentation decoder
        in_channels = [embed_dim] * 4

        self.seg_decoder: UPerNetSegDecoder | FPNSegDecoder | LinearSegDecoder
        if decoder_type == "upernet":
            self.seg_decoder = UPerNetSegDecoder(
                in_channels=in_channels,
                channels=decoder_channels,
                num_classes=n_classes,
                dropout_rate=decoder_dropout,
            )
        elif decoder_type == "fpn":
            self.seg_decoder = FPNSegDecoder(
                in_channels=in_channels,
                channels=decoder_channels,
                num_classes=n_classes,
                dropout_rate=decoder_dropout,
            )
        elif decoder_type == "linear":
            self.seg_decoder = LinearSegDecoder(
                in_channels=embed_dim,
                num_classes=n_classes,
                dropout_rate=decoder_dropout,
            )
        else:
            raise ValueError(f"Unknown decoder type: {decoder_type}")

        # Initialize weights
        self._init_weights()

        # Load pretrained weights if requested
        if use_pretrained and pretrained_path:
            self._load_pretrained(pretrained_path)

        # Freeze encoder if requested (freeze_encoder=True takes precedence)
        if freeze_encoder:
            self._freeze_encoder()
        elif freeze_encoder_layers > 0:
            self._freeze_encoder_layers(freeze_encoder_layers, mode=freeze_encoder_layers_mode)

        logger.info(
            f"CROMA initialized: embed_dim={embed_dim}, depth={depth}, "
            f"heads={num_heads}, patch={patch_size}, SAR={self.has_sar}, "
            f"optical={self.has_optical}, decoder={decoder_type}"
        )

    @staticmethod
    def _compute_out_indices(depth: int, num_scales: int) -> tuple[int, ...]:
        """Compute evenly spaced output indices for multi-scale features.

        Handles small depths gracefully by distributing indices across
        available layers (with repetition if depth < num_scales).

        Args:
            depth: Total number of transformer blocks.
            num_scales: Number of feature scales to extract.

        Returns:
            Tuple of block indices to extract features from.
        """
        if depth >= num_scales:
            # Evenly spaced: e.g., depth=12, scales=4 -> (2, 5, 8, 11)
            return tuple(depth // num_scales * (i + 1) - 1 for i in range(num_scales))
        else:
            # depth < num_scales: use all layers, repeating last layer
            indices = list(range(depth))
            while len(indices) < num_scales:
                indices.append(depth - 1)
            return tuple(indices)

    def _init_weights(self) -> None:
        """Initialize model weights using truncated normal."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _load_pretrained(self, pretrained_path: str) -> None:
        """Load pretrained CROMA weights.

        Supports loading from:
        - Local checkpoint file (.pt)
        - HuggingFace Hub (antofuller/CROMA)
        """
        try:
            import os

            if os.path.isfile(pretrained_path):
                logger.info(f"Loading pretrained weights from local file: {pretrained_path}")
                state_dict = torch.load(pretrained_path, map_location="cpu", weights_only=True)
            else:
                # Try loading from HuggingFace Hub
                try:
                    from huggingface_hub import hf_hub_download

                    logger.info(f"Loading pretrained weights from HuggingFace: {pretrained_path}")
                    # Determine file name from variant
                    filename = "CROMA_base.pt" if self.depth <= 12 else "CROMA_large.pt"
                    weights_file = hf_hub_download(
                        repo_id=pretrained_path,
                        filename=filename,
                    )
                    state_dict = torch.load(weights_file, map_location="cpu", weights_only=True)
                except ImportError:
                    logger.warning("huggingface_hub not installed. Install with: pip install huggingface_hub")
                    return

            adapted = self._adapt_pretrained_weights(state_dict)
            missing, unexpected = self.load_state_dict(adapted, strict=False)

            log_pretrained_load_result(
                missing,
                unexpected,
                logger,
                model_name="CROMA",
                expected_missing_prefixes=["seg_decoder."],
            )

            logger.info("Successfully loaded pretrained CROMA weights")

        except Exception as e:
            logger.warning(f"Failed to load pretrained weights: {e}")
            logger.info("Continuing with random initialization")

    def _adapt_pretrained_weights(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Adapt pretrained CROMA weights to our architecture.

        The official CROMA checkpoint (antofuller/CROMA) stores weights as a
        dict of 5 OrderedDicts, each containing a sub-module's state dict:
            s1_encoder, s1_GAP_FFN, s2_encoder, s2_GAP_FFN, joint_encoder

        The internal key names also differ from ours:
        - Patch embed is inside the encoder as ``linear_input``
        - Self-attention blocks: ``transformer.layers.i.[0|1].*``
        - Cross-attention blocks: ``layers.i.[0|1|2].*``
        - Norm names: ``input_norm`` (per sub-module), ``norm_out`` (final)
        - FFN weights: ``net.0`` / ``net.3`` (our: ``fc1`` / ``fc2``)

        Args:
            state_dict: Original checkpoint state dict (may be nested or flat).

        Returns:
            Adapted flat state dict compatible with our model.
        """
        # Flatten nested OrderedDicts produced by the official checkpoint into
        # a single flat dict with dotted keys, e.g.:
        #   s1_encoder -> {linear_input.weight: ...}
        # becomes: s1_encoder.linear_input.weight -> tensor
        flat: dict[str, torch.Tensor] = {}
        for top_key, value in state_dict.items():
            if isinstance(value, dict):
                for sub_key, tensor in value.items():
                    flat[f"{top_key}.{sub_key}"] = tensor
            else:
                flat[top_key] = value

        adapted: dict[str, torch.Tensor] = {}

        for key, value in flat.items():
            new_key = self._remap_pretrained_key(key)
            if new_key is None:
                continue
            adapted[new_key] = value

        return adapted

    def _remap_pretrained_key(self, key: str) -> str | None:
        """Map a single official CROMA checkpoint key to our naming scheme.

        Returns the remapped key, or None if the key should be skipped.

        Official checkpoint structure vs ours:

        Unimodal encoder (s1/s2_encoder):
            linear_input.*                -> patch_embed.proj.*
            transformer.layers.i.0.*     -> encoder.blocks.i  (self-attn block)
              .to_qkv.*                    -> .attn.qkv.*
              .to_out.*                    -> .attn.proj.*
              .input_norm.*               -> .norm1.*
            transformer.layers.i.1.*     -> encoder.blocks.i  (FFN block)
              .net.0.*                     -> .ffn.fc1.*
              .net.3.*                     -> .ffn.fc2.*
              .input_norm.*               -> .norm2.*
            transformer.norm_out.*       -> encoder.norm.*

        Cross encoder (joint_encoder):
            layers.i.0.*                 -> cross_encoder.blocks.i  (self-attn)
              .to_qkv.*                    -> .self_attn.qkv.*
              .to_out.*                    -> .self_attn.proj.*
              .input_norm.*               -> .norm1.*
            layers.i.1.*                 -> cross_encoder.blocks.i  (cross-attn)
              .to_q.*                      -> .cross_attn.q_proj.*
              .to_k.*                      -> .cross_attn.k_proj.*
              .to_v.*                      -> .cross_attn.v_proj.*
              .to_out.*                    -> .cross_attn.proj.*
              .input_norm.*               -> .norm2.*
            layers.i.2.*                 -> cross_encoder.blocks.i  (FFN)
              .net.0.*                     -> .ffn.fc1.*
              .net.3.*                     -> .ffn.fc2.*
              .input_norm.*               -> .norm3.*
            norm_out.*                   -> cross_encoder.norm.*
        """
        import re

        # --- s1_encoder (SAR unimodal) ---
        if key.startswith("s1_encoder."):
            rest = key[len("s1_encoder.") :]
            # Patch embed (linear_input inside encoder)
            if rest.startswith("linear_input."):
                suffix = rest[len("linear_input.") :]
                return f"sar_patch_embed.proj.{suffix}"
            # Final norm
            if rest.startswith("transformer.norm_out."):
                suffix = rest[len("transformer.norm_out.") :]
                return f"sar_encoder.norm.{suffix}"
            # Transformer blocks: transformer.layers.i.{0|1}.*
            m = re.match(r"transformer\.layers\.(\d+)\.(0|1)\.(.*)", rest)
            if m:
                i, slot, tail = m.group(1), m.group(2), m.group(3)
                return self._remap_unimodal_block(i, slot, tail, "sar_encoder")
            return None

        # --- s2_encoder (optical unimodal) ---
        if key.startswith("s2_encoder."):
            rest = key[len("s2_encoder.") :]
            if rest.startswith("linear_input."):
                suffix = rest[len("linear_input.") :]
                return f"optical_patch_embed.proj.{suffix}"
            if rest.startswith("transformer.norm_out."):
                suffix = rest[len("transformer.norm_out.") :]
                return f"optical_encoder.norm.{suffix}"
            m = re.match(r"transformer\.layers\.(\d+)\.(0|1)\.(.*)", rest)
            if m:
                i, slot, tail = m.group(1), m.group(2), m.group(3)
                return self._remap_unimodal_block(i, slot, tail, "optical_encoder")
            return None

        # --- joint_encoder (cross-modal encoder) ---
        if key.startswith("joint_encoder."):
            rest = key[len("joint_encoder.") :]
            # Final norm
            if rest.startswith("norm_out."):
                suffix = rest[len("norm_out.") :]
                return f"cross_encoder.norm.{suffix}"
            # Cross blocks: layers.i.{0|1|2}.*
            m = re.match(r"layers\.(\d+)\.(0|1|2)\.(.*)", rest)
            if m:
                i, slot, tail = m.group(1), m.group(2), m.group(3)
                return self._remap_cross_block(i, slot, tail)
            return None

        # Unknown top-level key — skip
        return None

    @staticmethod
    def _remap_unimodal_block(i: str, slot: str, tail: str, encoder_prefix: str) -> str | None:
        """Remap a single unimodal encoder block key.

        Slot 0 = self-attention (Attention module).
        Slot 1 = feed-forward (FFN module).
        """
        if slot == "0":
            # Self-attention block
            if tail.startswith("to_qkv."):
                return f"{encoder_prefix}.blocks.{i}.attn.qkv.{tail[len('to_qkv.') :]}"
            if tail.startswith("to_out."):
                return f"{encoder_prefix}.blocks.{i}.attn.proj.{tail[len('to_out.') :]}"
            if tail.startswith("input_norm."):
                return f"{encoder_prefix}.blocks.{i}.norm1.{tail[len('input_norm.') :]}"
        elif slot == "1":
            # FFN block
            if tail.startswith("net.0."):
                return f"{encoder_prefix}.blocks.{i}.ffn.fc1.{tail[len('net.0.') :]}"
            if tail.startswith("net.3."):
                return f"{encoder_prefix}.blocks.{i}.ffn.fc2.{tail[len('net.3.') :]}"
            if tail.startswith("input_norm."):
                return f"{encoder_prefix}.blocks.{i}.norm2.{tail[len('input_norm.') :]}"
        return None

    @staticmethod
    def _remap_cross_block(i: str, slot: str, tail: str) -> str | None:
        """Remap a single cross-encoder block key.

        Slot 0 = self-attention (Attention module).
        Slot 1 = cross-attention (CrossAttention module).
        Slot 2 = feed-forward (FFN module).
        """
        prefix = f"cross_encoder.blocks.{i}"
        if slot == "0":
            # Self-attention
            if tail.startswith("to_qkv."):
                return f"{prefix}.self_attn.qkv.{tail[len('to_qkv.') :]}"
            if tail.startswith("to_out."):
                return f"{prefix}.self_attn.proj.{tail[len('to_out.') :]}"
            if tail.startswith("input_norm."):
                return f"{prefix}.norm1.{tail[len('input_norm.') :]}"
        elif slot == "1":
            # Cross-attention
            if tail.startswith("to_q."):
                return f"{prefix}.cross_attn.q_proj.{tail[len('to_q.') :]}"
            if tail.startswith("to_k."):
                return f"{prefix}.cross_attn.k_proj.{tail[len('to_k.') :]}"
            if tail.startswith("to_v."):
                return f"{prefix}.cross_attn.v_proj.{tail[len('to_v.') :]}"
            if tail.startswith("to_out."):
                return f"{prefix}.cross_attn.proj.{tail[len('to_out.') :]}"
            if tail.startswith("input_norm."):
                return f"{prefix}.norm2.{tail[len('input_norm.') :]}"
        elif slot == "2":
            # FFN
            if tail.startswith("net.0."):
                return f"{prefix}.ffn.fc1.{tail[len('net.0.') :]}"
            if tail.startswith("net.3."):
                return f"{prefix}.ffn.fc2.{tail[len('net.3.') :]}"
            if tail.startswith("input_norm."):
                return f"{prefix}.norm3.{tail[len('input_norm.') :]}"
        return None

    def _freeze_encoder(self) -> None:
        encoder_modules = self._get_encoder_modules()
        for module in encoder_modules:
            for param in module.parameters():
                param.requires_grad = False
        logger.info("All encoder weights frozen")

    def _freeze_encoder_layers(self, n_layers: int, mode: str = "individual") -> None:
        """Freeze first N layers of the encoders.

        Supports two modes:
        - "individual": Treats patch_embed + each transformer block as a layer
        - "grouped": Groups blocks into logical stages

        For CROMA with depth=12 (base), individual mode layers are:
        - Layer 0: SAR patch_embed (or optical patch_embed if SAR not available)
        - Layer 1: optical patch_embed (if both modalities)
        - Layer 2+: encoder blocks in order (SAR blocks, then optical blocks)

        Args:
            n_layers: Number of layers to freeze.
            mode: "individual" or "grouped".
        """
        if n_layers <= 0:
            return

        layer_groups = self._get_layer_groups(mode)
        max_layers = len(layer_groups)

        if n_layers > max_layers:
            logger.warning(f"freeze_encoder_layers={n_layers} exceeds max layers {max_layers}, using {max_layers}")
            n_layers = max_layers

        frozen_count = 0
        for i in range(n_layers):
            layer_name, modules = layer_groups[i]
            for module in modules:
                for param in module.parameters():
                    param.requires_grad = False
            frozen_count += 1
            logger.debug(f"Froze encoder layer {i}: {layer_name}")

        logger.info(f"Froze first {frozen_count} encoder layers (out of {max_layers})")

    def _get_layer_groups(self, mode: str = "individual") -> list[tuple[str, list[nn.Module]]]:
        """Get encoder layer groups for freezing/unfreezing.

        Args:
            mode: "individual" or "grouped".

        Returns:
            List of (name, modules) tuples.
        """
        groups: list[tuple[str, list[nn.Module]]] = []

        if mode == "individual":
            # Patch embeddings first
            if self.has_sar:
                groups.append(("sar_patch_embed", [self.sar_patch_embed]))
            if self.has_optical:
                groups.append(("optical_patch_embed", [self.optical_patch_embed]))

            # SAR encoder blocks
            if self.has_sar:
                for i, block in enumerate(self.sar_encoder.blocks):
                    groups.append((f"sar_block_{i}", [block]))

            # Optical encoder blocks
            if self.has_optical:
                for i, block in enumerate(self.optical_encoder.blocks):
                    groups.append((f"optical_block_{i}", [block]))

            # Cross encoder blocks
            if self.has_both:
                for i, block in enumerate(self.cross_encoder.blocks):
                    groups.append((f"cross_block_{i}", [block]))

        elif mode == "grouped":
            # Group into: patch_embeds, SAR encoder, optical encoder, cross encoder
            if self.has_sar:
                groups.append(("sar_embed", [self.sar_patch_embed]))
                groups.append(("sar_encoder", [self.sar_encoder]))
            if self.has_optical:
                groups.append(("optical_embed", [self.optical_patch_embed]))
                groups.append(("optical_encoder", [self.optical_encoder]))
            if self.has_both:
                groups.append(("cross_encoder", [self.cross_encoder]))

        return groups

    def get_frozen_layer_info(self) -> dict[str, int | list[str]]:
        """Get information about frozen layers for logging.

        Returns:
            Dictionary with total_layers, frozen_layers count,
            and list of frozen layer names.
        """
        layer_groups = self._get_layer_groups("individual")

        frozen_layers = []
        for layer_name, modules in layer_groups:
            is_frozen = all(not param.requires_grad for module in modules for param in module.parameters())
            if is_frozen:
                frozen_layers.append(layer_name)

        return {
            "total_layers": len(layer_groups),
            "frozen_layers": len(frozen_layers),
            "frozen_layer_names": frozen_layers,
        }

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for segmentation.

        Splits input into SAR and optical channels, processes through
        respective encoders, fuses with cross-encoder (if both modalities),
        and decodes to segmentation logits.

        Args:
            x: Input tensor of shape (B, C, H, W) where C = n_channels_sar + n_channels_optical.

        Returns:
            Segmentation logits of shape (B, n_classes, H, W).
        """
        B, C, H, W = x.shape
        output_size = (H, W)

        # Get ALiBi bias (may need recomputing if image size changed)
        alibi_bias = self._get_alibi_bias(H, W, x.device)

        if self.has_both:
            # Split input into optical and SAR (dataset stores S2 first, then S1).
            optical_input, sar_input = self._split_both_modalities(x)

            # Embed patches
            sar_tokens = self.sar_patch_embed(sar_input)
            optical_tokens = self.optical_patch_embed(optical_input)

            # Encode each modality
            sar_features = self.sar_encoder(sar_tokens, alibi_bias)
            optical_features = self.optical_encoder(optical_tokens, alibi_bias)

            # Cross-encoder fusion with multi-scale output
            # We use the segmentation-specific cross encoder that outputs at multiple indices
            # Feed the optical features as context
            features = self._cross_encode_for_seg(sar_features, optical_features, alibi_bias)

        elif self.has_optical:
            # Optical only: use optical encoder with multi-scale output
            optical_tokens = self.optical_patch_embed(x)
            features = self.optical_encoder(optical_tokens, alibi_bias)
            if not isinstance(features, list):
                features = [features] * 4

        elif self.has_sar:
            # SAR only: use SAR encoder
            sar_tokens = self.sar_patch_embed(x)
            features = self.sar_encoder(sar_tokens, alibi_bias)
            if not isinstance(features, list):
                features = [features] * 4
        else:
            raise ValueError("At least one of SAR or optical channels must be > 0")

        # Reshape features from (B, N, D) to (B, D, H/P, W/P) for decoder
        grid_h = H // self.patch_size
        grid_w = W // self.patch_size
        spatial_features = []
        for feat in features:
            feat_spatial = feat.transpose(1, 2).reshape(B, -1, grid_h, grid_w)
            spatial_features.append(feat_spatial)

        # Decode
        result: torch.Tensor = self.seg_decoder(spatial_features, output_size=output_size)

        return result

    def _cross_encode_for_seg(
        self,
        sar_features: torch.Tensor,
        optical_features: torch.Tensor,
        alibi_bias: torch.Tensor | None,
    ) -> list[torch.Tensor]:
        """Run cross-encoder for segmentation with multi-scale output.

        Uses proper cross-attention: SAR tokens self-attend, then cross-attend
        to optical tokens, matching the original CROMA architecture.

        Args:
            sar_features: SAR encoder output (B, N, D).
            optical_features: Optical encoder output (B, N, D).
            alibi_bias: ALiBi positional bias (used for both self and cross attention).

        Returns:
            List of 4 feature tensors for multi-scale decoding.
        """
        # SAR and optical are spatially aligned, so ALiBi bias is the same
        # for both self-attention and cross-attention (2D-ALiBi = X-ALiBi).
        features = self.cross_encoder(
            sar_tokens=sar_features,
            optical_tokens=optical_features,
            self_alibi_bias=alibi_bias,
            cross_alibi_bias=alibi_bias,
        )
        if not isinstance(features, list):
            features = [features] * 4
        return features

    def _split_both_modalities(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Split an input batch into optical and SAR tensors."""
        total_expected = self.n_channels_optical + self.n_channels_sar
        if x.shape[1] == total_expected:
            optical_input = x[:, : self.n_channels_optical, :, :]
            sar_input = x[:, self.n_channels_optical :, :, :]
            return optical_input, sar_input

        # Optional compatibility path when data includes all 13 S2 bands and
        # model expects 12 optical channels (cirrus removed).
        if self.croma_drop_optical_cirrus and x.shape[1] == total_expected + 1:
            optical_full = x[:, : self.n_channels_optical + 1, :, :]
            optical_input = torch.cat([optical_full[:, :10, :, :], optical_full[:, 11:, :, :]], dim=1)
            sar_input = x[:, self.n_channels_optical + 1 :, :, :]
            return optical_input, sar_input

        raise ValueError(f"Unexpected channel count for CROMA dual-modality input: {x.shape[1]} (expected {total_expected} or {total_expected + 1}).")

    def _get_alibi_bias(self, h: int, w: int, device: torch.device) -> torch.Tensor | None:
        """Get ALiBi bias, recomputing if image size changed.

        Args:
            h: Image height.
            w: Image width.
            device: Target device.

        Returns:
            ALiBi bias tensor or None if ALiBi is disabled.
        """
        if not self.use_alibi:
            return None

        grid_h = h // self.patch_size
        grid_w = w // self.patch_size
        current_num_patches = grid_h * grid_w

        # Check if cached bias has correct size
        if self.alibi_bias is not None and self.alibi_bias.shape[-1] == current_num_patches:
            return self.alibi_bias

        # Recompute for different image size (enables extrapolation)
        logger.debug(f"Recomputing ALiBi bias for grid {grid_h}x{grid_w} (was {self.grid_size}x{self.grid_size})")
        alibi = get_2d_alibi(self.num_heads, grid_h, grid_w).to(device)
        return alibi

    def _get_encoder_modules(self) -> list[nn.Module]:
        """Get all encoder modules (for freezing/parameter grouping)."""
        modules: list[nn.Module] = []
        if self.has_sar:
            modules.extend([self.sar_patch_embed, self.sar_encoder])
        if self.has_optical:
            modules.extend([self.optical_patch_embed, self.optical_encoder])
        if self.has_both:
            modules.append(self.cross_encoder)
        return modules

    def get_encoder_params(self) -> Iterator[nn.Parameter]:
        """Get encoder parameters (for differential learning rates)."""
        for module in self._get_encoder_modules():
            yield from module.parameters()

    def get_decoder_params(self) -> Iterator[nn.Parameter]:
        """Get decoder parameters (for differential learning rates)."""
        yield from self.seg_decoder.parameters()

    def get_param_groups(self, encoder_lr: float, decoder_lr: float) -> list[dict[str, Any]]:
        """Get parameter groups with different learning rates.

        Args:
            encoder_lr: Learning rate for encoder.
            decoder_lr: Learning rate for decoder.

        Returns:
            List of parameter group dicts for optimizer.
        """
        return [
            {"params": list(self.get_encoder_params()), "lr": encoder_lr},
            {"params": list(self.get_decoder_params()), "lr": decoder_lr},
        ]
