# Implemented from: Li et al., "FlexiMo: A Flexible Remote Sensing Foundation Model",
#   IEEE Transactions on Geoscience and Remote Sensing, 2026
# Reference implementation: https://github.com/danfenghong/IEEE_TGRS_Fleximo
# Pretrained weights and layout follow DOFA: https://github.com/zhu-xlab/DOFA

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.constants import S2_BANDS_HR, S2_BANDS_LR, S2_BANDS_MR

from .copernicus_fm_utils.decoder import FPNDecoder, LinearDecoder, UPerNetDecoder
from .fleximo_utils import DynamicPatchEmbed, FlexiMoEncoder, get_2d_sincos_pos_embed
from .pretrained_loading import log_pretrained_load_result

logger = logging.getLogger(__name__)


# Central wavelengths (in micrometers) keyed by the 1-based rasterio band index in the
# SEN12MS/DFC2020 Sentinel-2 GeoTIFFs, where band 9 = B8A, 10 = B09 and 11 = B10.
# Values follow the DOFA convention for wavelength-based encoding, as recorded in DOFA's
# pretraining table `pretraining/datasets/waves.json` (https://github.com/zhu-xlab/DOFA).
S2_WAVELENGTHS_UM: dict[int, float] = {
    1: 0.443,  # B01 coastal aerosol
    2: 0.49,  # B02 blue
    3: 0.56,  # B03 green
    4: 0.665,  # B04 red
    5: 0.705,  # B05 red edge 1
    6: 0.74,  # B06 red edge 2
    7: 0.783,  # B07 red edge 3
    8: 0.842,  # B08 NIR
    9: 0.865,  # B8A narrow NIR
    10: 0.945,  # B09 water vapour
    11: 1.375,  # B10 cirrus
    12: 1.61,  # B11 SWIR 1
    13: 2.19,  # B12 SWIR 2
}
# S1 SAR bands: VV, VH. DOFA's waves.json uses 3.75 for its 2-channel (Sentinel-1) key,
# so this is the value the released `earthflow/DOFA` weights were pretrained with. Note
# DOFA's README example instead shows 5.405 (the C-band centre *frequency* in GHz); the
# pretraining table is authoritative here.
S1_WAVELENGTHS = [3.75, 3.75]


def build_wavelength_list(
    use_s2hr: bool = False,
    use_s2mr: bool = False,
    use_s2lr: bool = False,
    use_s1: bool = False,
) -> list[float]:
    """Build wavelength list from sensor configuration.

    The order must match the channel stacking order the dataset produces: ``load_s2``
    concatenates the selected HR/MR/LR band numbers and then **sorts** them, and
    ``load_sample`` appends S1 **after** S2. So the channels arrive in ascending S2 band
    order followed by VV/VH, not grouped by resolution.

    This ordering matters because the wavelength list is positional: FlexiMo's
    ``DynamicPatchEmbed`` generates the patch-embedding conv weights per channel from
    these wavelengths, so a mismatch silently trains every channel against the wrong
    spectral filter rather than raising a shape error.

    Args:
        use_s2hr: Whether S2 high-resolution bands are used.
        use_s2mr: Whether S2 medium-resolution bands are used.
        use_s2lr: Whether S2 low-resolution bands are used.
        use_s1: Whether S1 SAR bands are used.

    Returns:
        List of central wavelengths in micrometers, in dataset channel order.
    """
    bands: list[int] = []
    if use_s2hr:
        bands.extend(S2_BANDS_HR)
    if use_s2mr:
        bands.extend(S2_BANDS_MR)
    if use_s2lr:
        bands.extend(S2_BANDS_LR)
    bands.sort()

    wavelengths: list[float] = [S2_WAVELENGTHS_UM[b] for b in bands]
    if use_s1:
        wavelengths.extend(S1_WAVELENGTHS)
    return wavelengths


# Model variant configurations
FLEXIMO_VARIANTS: dict[str, dict[str, Any]] = {
    "base": {
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "patch_size": 16,
    },
    "small": {
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
        "patch_size": 16,
    },
    "large": {
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "patch_size": 16,
    },
}


class FlexiMo(nn.Module):
    """FlexiMo: Flexible Remote Sensing Foundation Model.

    A foundation model for Earth observation that uses wavelength-conditioned
    dynamic convolutions for patch embedding, enabling a single model to
    process inputs with any number of spectral channels from any sensor.

    Key FlexiMo features beyond DOFA:
    - PI-Resize: pseudo-inverse bilinear interpolation for flexible patch sizes
      at inference time without retraining (FlexiViT, Beyer et al. CVPR 2023)
    - Learnable positional embeddings with bicubic interpolation for variable
      grid sizes

    Args:
        n_channels: Number of input channels.
        n_classes: Number of output segmentation classes.
        img_size: Input image size (assumes square images).
        patch_size: Base patch size for the weight generator (DOFA pretrained
            size, typically 16). The weight generator always produces kernels
            at this size.
        target_patch_size: Actual patch size for convolution stride. If None,
            uses patch_size (no PI-Resize, degrades to DOFA). Set to
            {4, 8, 16, 32, 64} for flexible patch sizes via PI-Resize.
        embed_dim: Embedding dimension.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        mlp_ratio: Ratio of MLP hidden dimension.
        qkv_bias: If True, add learnable bias to Q, K, V projections.
        drop_rate: Dropout rate.
        attn_drop_rate: Attention dropout rate.
        drop_path_rate: Stochastic depth rate.
        wv_planes: Dimension of wavelength embeddings.
        pos_embed_type: Positional embedding type. 'learnable' (FlexiMo paper
            default) or 'sincos' (fixed, DOFA-compatible).
        decoder_type: Decoder type ('upernet', 'fpn', or 'linear').
        decoder_channels: Decoder channel dimension.
        decoder_dropout: Decoder dropout rate.
        task_type: Task type ("segmentation" or "classification").
        classifier_dropout: Dropout before CLS classification head.
        wavelengths: List of central wavelengths (micrometers) for input bands.
            If None, defaults to uniform spacing.
        use_s2hr: Whether S2 HR bands are used (for auto wavelength mapping).
        use_s2mr: Whether S2 MR bands are used.
        use_s2lr: Whether S2 LR bands are used.
        use_s1: Whether S1 SAR bands are used.
        use_pretrained: Whether to load DOFA pretrained weights.
        pretrained_path: HuggingFace path for DOFA weights.
        freeze_encoder: Whether to freeze encoder weights.
        freeze_encoder_layers: Number of encoder layers to freeze.
        freeze_encoder_layers_mode: Freezing mode ('individual' or 'grouped').
        variant: Model variant shortcut (e.g., 'base', 'large', 'small').

    Example:
        >>> model = FlexiMo(
        ...     n_channels=13, n_classes=10, variant='base',
        ...     target_patch_size=8,  # PI-Resize: 16x16 kernels -> 8x8
        ...     use_s1=True, use_s2hr=True, use_s2mr=True, use_s2lr=True,
        ... )
        >>> x = torch.randn(2, 13, 256, 256)
        >>> out = model(x)  # (2, 10, 256, 256)
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        img_size: int,
        patch_size: int,
        target_patch_size: int | None,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        drop_rate: float,
        attn_drop_rate: float,
        drop_path_rate: float,
        wv_planes: int,
        pos_embed_type: str,
        decoder_type: str,
        decoder_channels: int,
        decoder_dropout: float,
        task_type: str,
        classifier_dropout: float,
        wavelengths: list[float] | None,
        use_s2hr: bool,
        use_s2mr: bool,
        use_s2lr: bool,
        use_s1: bool,
        use_pretrained: bool,
        pretrained_path: str,
        freeze_encoder: bool,
        freeze_encoder_layers: int,
        freeze_encoder_layers_mode: str,
        variant: str | None,
    ) -> None:
        super().__init__()

        # Apply variant configuration if specified
        if variant is not None and variant in FLEXIMO_VARIANTS:
            config = FLEXIMO_VARIANTS[variant]
            embed_dim = config["embed_dim"]
            depth = config["depth"]
            num_heads = config["num_heads"]
            patch_size = config["patch_size"]
            logger.info(f"Using FlexiMo variant '{variant}': {config}")

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.img_size = img_size
        self.base_patch_size = patch_size
        self.target_patch_size = target_patch_size if target_patch_size is not None else patch_size
        self.patch_size = self.target_patch_size  # effective patch size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.wv_planes = wv_planes
        self.pos_embed_type = pos_embed_type
        self.decoder_type = decoder_type
        self.task_type = task_type.lower()
        if self.task_type not in {"segmentation", "classification"}:
            raise ValueError(f"Unknown task_type: {task_type}. Expected one of: segmentation, classification.")

        # Validate image size is divisible by target patch size
        if img_size % self.target_patch_size != 0:
            raise ValueError(f"img_size ({img_size}) must be divisible by target_patch_size ({self.target_patch_size})")

        self.grid_size = img_size // self.target_patch_size
        self.num_patches = self.grid_size**2

        # Build wavelength list from sensor config or explicit list
        if wavelengths is not None:
            self._wavelengths = wavelengths
        else:
            auto_wavelengths = build_wavelength_list(
                use_s2hr=use_s2hr,
                use_s2mr=use_s2mr,
                use_s2lr=use_s2lr,
                use_s1=use_s1,
            )
            if auto_wavelengths:
                self._wavelengths = auto_wavelengths
            else:
                # Fallback: uniform spacing if no sensor config provided
                self._wavelengths = [0.5 + i * 0.1 for i in range(n_channels)]
                logger.warning("No sensor configuration or wavelengths provided. Using uniform wavelength spacing as fallback.")

        # Fail loudly rather than truncating or padding. The wavelength list is positional:
        # silently reshaping it to fit lets the model train every channel against the wrong
        # spectral filter with no error, which is exactly how a band-ordering bug survived a
        # full experiment sweep undetected.
        if not isinstance(self._wavelengths, (list, tuple)):
            raise TypeError(
                f"FlexiMo wavelengths must be a list or tuple of floats, got {type(self._wavelengths).__name__}. "
                "Note the `wavelengths` args-file key is read as a raw string and is not parsed into a list."
            )
        if len(self._wavelengths) != n_channels:
            raise ValueError(
                f"FlexiMo wavelength count ({len(self._wavelengths)}) does not match n_channels ({n_channels}). "
                "The wavelength list is positional and must name each input channel's band in dataset order "
                "(ascending S2 band number, then S1 VV/VH). Check the use_s2hr/use_s2mr/use_s2lr/use_s1 flags."
            )

        # Register wavelengths as a buffer (not a parameter)
        self.register_buffer(
            "wavelengths",
            torch.tensor(self._wavelengths, dtype=torch.float32),
        )

        # Dynamic patch embedding with PI-Resize for flexible patch sizes
        self.patch_embed = DynamicPatchEmbed(
            wv_planes=wv_planes,
            base_kernel_size=patch_size,
            target_patch_size=self.target_patch_size if self.target_patch_size != patch_size else None,
            embed_dim=embed_dim,
        )

        # Positional embeddings (initialized with sincos values)
        is_learnable = pos_embed_type == "learnable"
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim),
            requires_grad=is_learnable,
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        # CLS token (for compatibility with DOFA pretrained weights)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Calculate output indices for multi-scale features
        out_indices = tuple(depth // 4 * (i + 1) - 1 for i in range(4))

        # Encoder
        self.encoder = FlexiMoEncoder(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            out_indices=out_indices,
        )

        # All encoder outputs have the same embed_dim
        in_channels = [embed_dim] * 4

        # Decoder
        self.decoder: UPerNetDecoder | FPNDecoder | LinearDecoder
        if decoder_type == "upernet":
            self.decoder = UPerNetDecoder(
                in_channels=in_channels,
                channels=decoder_channels,
                num_classes=n_classes,
                dropout_rate=decoder_dropout,
            )
        elif decoder_type == "fpn":
            self.decoder = FPNDecoder(
                in_channels=in_channels,
                channels=decoder_channels,
                num_classes=n_classes,
                dropout_rate=decoder_dropout,
            )
        elif decoder_type == "linear":
            self.decoder = LinearDecoder(
                in_channels=embed_dim,
                num_classes=n_classes,
                dropout_rate=decoder_dropout,
            )
        else:
            raise ValueError(f"Unknown decoder type: {decoder_type}")

        self.classifier_dropout = nn.Dropout(p=classifier_dropout)
        self.classifier_head = nn.Linear(embed_dim, n_classes)

        # Initialize weights
        self._init_weights()

        # Load pretrained weights if requested
        if use_pretrained:
            self._load_pretrained(pretrained_path)

        if freeze_encoder:
            self._freeze_encoder()
        elif freeze_encoder_layers > 0:
            self._freeze_encoder_layers(freeze_encoder_layers, mode=freeze_encoder_layers_mode)

    def _init_weights(self) -> None:
        """Initialize model weights including positional embeddings.

        Positional embeddings are always initialized with sincos values.
        For 'learnable' type, these values serve as the starting point
        and are updated during fine-tuning.
        """
        pos_embed = get_2d_sincos_pos_embed(self.embed_dim, self.grid_size)
        self.pos_embed.data.copy_(pos_embed.unsqueeze(0))

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.classifier_head.weight, std=0.02)
        if self.classifier_head.bias is not None:
            nn.init.zeros_(self.classifier_head.bias)

    def _load_pretrained(self, pretrained_path: str) -> None:
        """Load pretrained DOFA weights.

        Args:
            pretrained_path: HuggingFace model identifier (e.g. 'earthflow/DOFA').
        """
        try:
            from huggingface_hub import hf_hub_download

            logger.info(f"Loading DOFA pretrained weights from {pretrained_path}")

            weights_file = hf_hub_download(
                repo_id=pretrained_path,
                filename="DOFA_ViT_base_e100.pth",
            )

            state_dict = torch.load(weights_file, map_location="cpu", weights_only=True)

            adapted = self._adapt_pretrained_weights(state_dict)

            missing, unexpected = self.load_state_dict(adapted, strict=False)

            log_pretrained_load_result(
                missing,
                unexpected,
                logger,
                model_name="FlexiMo",
                expected_missing_prefixes=["decoder.", "classifier_head."],
            )

            logger.info("Successfully loaded DOFA pretrained weights")

        except ImportError:
            logger.warning("huggingface_hub not installed. Install with: pip install huggingface_hub")
        except Exception as e:
            logger.warning(f"Failed to load pretrained weights: {e}")
            logger.info("Continuing with random initialization")

    def _adapt_pretrained_weights(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Adapt DOFA checkpoint keys to our architecture.

        DOFA uses flat key names (e.g. 'blocks.0.attn.qkv.weight'),
        while our model nests them under 'encoder' (e.g.
        'encoder.blocks.0.attn.qkv.weight').
        """
        adapted: dict[str, torch.Tensor] = {}

        for key, value in state_dict.items():
            new_key = key

            # Map DOFA block keys to our encoder
            if key.startswith("blocks."):
                new_key = "encoder." + key

            # patch_embed.* keys pass through unchanged — DOFA uses the same
            # DynamicPatchEmbed structure (weight_generator, fclayer), so key
            # names already match our model exactly.

            # Skip MAE pretraining artifacts not present in our model
            if key in {"mask_token", "projector.weight", "projector.bias"}:
                continue

            # Map DOFA norm to our encoder norm
            if key == "fc_norm.weight" or key == "fc_norm.bias":
                new_key = key.replace("fc_norm", "encoder.norm")
            elif key == "norm.weight" or key == "norm.bias":
                new_key = "encoder." + key

            # Skip head (classification layer) - we have our own decoder
            if key.startswith("head."):
                continue

            # Handle positional embedding: strip CLS token, interpolate if needed
            if key == "pos_embed":
                if value.shape[1] < 2:
                    logger.info(f"Skipping pos_embed: unexpected shape {value.shape}")
                    continue
                # DOFA pos_embed includes CLS token at position 0
                # Strip CLS token position (we handle it separately)
                value = value[:, 1:, :]
                pretrained_num_patches = value.shape[1]
                if pretrained_num_patches != self.num_patches:
                    # Interpolate positional embeddings to match new grid size
                    logger.info(f"Interpolating pos_embed: {pretrained_num_patches} -> {self.num_patches} patches")
                    value = self._interpolate_pos_embed_static(value, self.grid_size, self.grid_size)

            adapted[new_key] = value

        return adapted

    def _freeze_encoder(self) -> None:
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.cls_token.requires_grad = False

        logger.info("Encoder weights frozen")

    def _freeze_encoder_layers(self, n_layers: int, mode: str = "individual") -> None:
        """Freeze first N layers of the encoder.

        Args:
            n_layers: Number of layers to freeze.
            mode: 'individual' (patch_embed + each block) or
                  'grouped' (patch_embed + blocks in thirds).
        """
        if n_layers <= 0:
            return

        if mode == "individual":
            max_layers = 1 + len(self.encoder.blocks)
            if n_layers > max_layers:
                logger.warning(f"freeze_encoder_layers={n_layers} exceeds max {max_layers}, using {max_layers}")
                n_layers = max_layers

            frozen_count = 0
            if n_layers > 0:
                for param in self.patch_embed.parameters():
                    param.requires_grad = False
                self.cls_token.requires_grad = False
                frozen_count = 1

            for i in range(min(n_layers - 1, len(self.encoder.blocks))):
                for param in self.encoder.blocks[i].parameters():
                    param.requires_grad = False
                frozen_count += 1

        elif mode == "grouped":
            num_blocks = len(self.encoder.blocks)
            group_size = num_blocks // 3
            max_groups = 4

            if n_layers > max_groups:
                logger.warning(f"freeze_encoder_layers={n_layers} exceeds max groups {max_groups}, using {max_groups}")
                n_layers = max_groups

            frozen_count = 0
            if n_layers > 0:
                for param in self.patch_embed.parameters():
                    param.requires_grad = False
                self.cls_token.requires_grad = False
                frozen_count += 1

            for group_idx in range(1, n_layers):
                start_block = (group_idx - 1) * group_size
                end_block = min(group_idx * group_size, num_blocks)
                for block_idx in range(start_block, end_block):
                    for param in self.encoder.blocks[block_idx].parameters():
                        param.requires_grad = False
                frozen_count += 1

            logger.info(f"Froze {frozen_count} encoder groups ({n_layers} requested, {mode} mode)")
            return
        else:
            logger.warning(f"Unknown freeze mode '{mode}', using 'individual'")
            self._freeze_encoder_layers(n_layers, mode="individual")
            return

        logger.info(f"Froze first {frozen_count} encoder layers")

    @staticmethod
    def _interpolate_pos_embed_static(
        pos_embed: torch.Tensor,
        target_h: int,
        target_w: int,
    ) -> torch.Tensor:
        """Interpolate positional embeddings to a new grid size.

        Uses bicubic interpolation (standard ViT practice) to adapt
        positional embeddings when the number of patches changes due to
        different patch sizes or image sizes.

        Args:
            pos_embed: Positional embeddings of shape (1, N, D).
            target_h: Target grid height.
            target_w: Target grid width.

        Returns:
            Interpolated embeddings of shape (1, target_h * target_w, D).
        """
        N = pos_embed.shape[1]
        D = pos_embed.shape[2]
        base_grid = int(N**0.5)

        if base_grid * base_grid != N:
            raise ValueError(f"Positional embedding count {N} is not a perfect square. Cannot reshape to 2D grid for interpolation.")

        # (1, N, D) -> (1, D, H_base, W_base) -> interpolate -> (1, D, H_new, W_new) -> (1, N_new, D)
        pos_2d = pos_embed.reshape(1, base_grid, base_grid, D).permute(0, 3, 1, 2)
        pos_2d = F.interpolate(
            pos_2d.float(),
            size=(target_h, target_w),
            mode="bicubic",
            align_corners=False,
        )
        result: torch.Tensor = pos_2d.permute(0, 2, 3, 1).reshape(1, target_h * target_w, D)
        return result

    def set_target_patch_size(self, target_patch_size: int) -> None:
        """Update the target patch size at runtime.

        Updates the PI-Resize matrix in the patch embedding and recalculates
        grid metadata. Useful for inference-time flexibility.

        Args:
            target_patch_size: New target patch size (e.g., 4, 8, 16, 32, 64).
        """
        self.target_patch_size = target_patch_size
        self.patch_size = target_patch_size
        self.patch_embed.set_target_patch_size(target_patch_size)
        logger.info(f"Target patch size updated to {target_patch_size}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Segmentation logits of shape (B, n_classes, H, W) when
            task_type="segmentation", otherwise classification logits
            of shape (B, n_classes).
        """
        B, C, H, W = x.shape

        # Dynamic patch embedding with wavelength conditioning
        # Weight generator produces kernels at base_patch_size,
        # PI-Resize adapts to target_patch_size before convolution
        x_tokens, _ = self.patch_embed(x, self.wavelengths)  # (B, N, E)

        # Compute actual grid dimensions from input
        grid_h = H // self.target_patch_size
        grid_w = W // self.target_patch_size
        num_tokens = grid_h * grid_w

        # Interpolate positional embeddings if grid size differs
        pos_embed = self._interpolate_pos_embed_static(self.pos_embed, grid_h, grid_w) if num_tokens != self.num_patches else self.pos_embed

        # Add positional embeddings
        x_tokens = x_tokens + pos_embed
        x_tokens = self.pos_drop(x_tokens)

        if self.task_type == "classification":
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x_tokens = torch.cat((cls_tokens, x_tokens), dim=1)
            _, final_tokens = self.encoder(x_tokens, return_final_tokens=True)
            cls_features = final_tokens[:, 0]
            return cast(torch.Tensor, self.classifier_head(self.classifier_dropout(cls_features)))

        # Encode with multi-scale feature extraction
        features = self.encoder(x_tokens)  # List of (B, N, E)

        # Reshape features to spatial format for decoder
        spatial_features = []
        for feat in features:
            feat = feat.transpose(1, 2)  # (B, E, N)
            feat = feat.reshape(B, -1, grid_h, grid_w)
            spatial_features.append(feat)

        # Decode
        output = self.decoder(spatial_features, output_size=(H, W))

        result: torch.Tensor = output
        return result

    def get_encoder_params(self) -> Iterator[nn.Parameter]:
        """Get encoder parameters (for differential learning rates)."""
        yield from self.patch_embed.parameters()
        yield self.pos_embed
        yield self.cls_token
        yield from self.encoder.parameters()

    def get_decoder_params(self) -> Iterator[nn.Parameter]:
        """Get non-encoder parameters (decoder or classification head)."""
        yield from self.decoder.parameters()
        yield from self.classifier_head.parameters()

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
