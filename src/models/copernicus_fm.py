# Implemented from: Wang et al., "Towards a Unified Copernicus Foundation Model for Earth Vision",
#   arXiv:2503.11849, 2025
# Reference implementation: https://github.com/zhu-xlab/Copernicus-FM
# Module and parameter names follow the reference implementation so that the official checkpoints load.

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any, cast

import torch
import torch.nn as nn

from .copernicus_fm_utils import (
    CopernicusEncoder,
    DynamicSpectralPatchEmbed,
    FourierExpansion,
    FPNDecoder,
    LinearDecoder,
    StaticPatchEmbed,
    UPerNetDecoder,
)
from .fleximo_utils.dynamic_patch_embed import get_2d_sincos_pos_embed
from .pretrained_loading import log_pretrained_load_result

logger = logging.getLogger(__name__)


S2_BAND_SPECS_NM: dict[int, tuple[float, float]] = {
    1: (440.0, 20.0),
    2: (490.0, 65.0),
    3: (560.0, 35.0),
    4: (665.0, 30.0),
    5: (705.0, 15.0),
    6: (740.0, 15.0),
    7: (783.0, 20.0),
    8: (842.0, 115.0),
    9: (860.0, 20.0),
    10: (940.0, 20.0),
    11: (1370.0, 30.0),
    12: (1610.0, 90.0),
    13: (2190.0, 180.0),
}

S1_WAVELENGTH_NM: float = 50_000_000.0
S1_BANDWIDTH_NM: float = 1_000_000_000.0

_S2_HR_BANDS = (2, 3, 4, 8)
_S2_MR_BANDS = (5, 6, 7, 9, 12, 13)
_S2_LR_BANDS = (1, 10, 11)


COPERNICUS_FM_VARIANTS: dict[str, dict[str, Any]] = {
    "base": {"embed_dim": 768, "depth": 12, "num_heads": 12, "patch_size": 16},
    "large": {"embed_dim": 1024, "depth": 24, "num_heads": 16, "patch_size": 16},
    "small": {"embed_dim": 384, "depth": 12, "num_heads": 6, "patch_size": 16},
}


def build_wavelength_list_nm(
    use_s2hr: bool = False,
    use_s2mr: bool = False,
    use_s2lr: bool = False,
    use_s1: bool = False,
) -> list[float]:
    """Build wavelength list in nm matching dataset channel order."""
    bands: list[int] = []
    if use_s2hr:
        bands.extend(_S2_HR_BANDS)
    if use_s2mr:
        bands.extend(_S2_MR_BANDS)
    if use_s2lr:
        bands.extend(_S2_LR_BANDS)
    bands.sort()

    wavelengths = [S2_BAND_SPECS_NM[b][0] for b in bands]
    if use_s1:
        wavelengths.extend([S1_WAVELENGTH_NM, S1_WAVELENGTH_NM])
    return wavelengths


def build_bandwidth_list_nm(
    use_s2hr: bool = False,
    use_s2mr: bool = False,
    use_s2lr: bool = False,
    use_s1: bool = False,
) -> list[float]:
    """Build bandwidth list in nm matching dataset channel order."""
    bands: list[int] = []
    if use_s2hr:
        bands.extend(_S2_HR_BANDS)
    if use_s2mr:
        bands.extend(_S2_MR_BANDS)
    if use_s2lr:
        bands.extend(_S2_LR_BANDS)
    bands.sort()

    bandwidths = [S2_BAND_SPECS_NM[b][1] for b in bands]
    if use_s1:
        bandwidths.extend([S1_BANDWIDTH_NM, S1_BANDWIDTH_NM])
    return bandwidths


class CopernicusFM(nn.Module):
    def __init__(
        self,
        n_channels: int,
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
        drop_path_rate: float,
        decoder_type: str,
        decoder_channels: int,
        decoder_dropout: float,
        use_spectral_hypernetwork: bool,
        wv_planes: int,
        wavelengths: list[float] | None,
        bandwidths: list[float] | None,
        use_variable_hypernetwork: bool,
        variable_embed_dim: int,
        use_metadata_encoding: bool,
        metadata_dropout_prob: float,
        use_adapters: bool,
        adapter_rank: int,
        use_pretrained: bool,
        pretrained_path: str,
        freeze_encoder: bool,
        freeze_encoder_layers: int,
        freeze_encoder_layers_mode: str,
        variant: str | None,
    ) -> None:
        super().__init__()

        if variant is not None:
            variant_value = str(variant).lower()
            if variant_value in COPERNICUS_FM_VARIANTS:
                config = COPERNICUS_FM_VARIANTS[variant_value]
                embed_dim = config["embed_dim"]
                depth = config["depth"]
                num_heads = config["num_heads"]
                patch_size = config["patch_size"]
                logger.info(f"Using Copernicus-FM variant '{variant_value}': {config}")

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.use_spectral_hypernetwork = use_spectral_hypernetwork
        self.use_metadata_encoding = use_metadata_encoding
        self.metadata_dropout_prob = metadata_dropout_prob
        self.freeze_encoder = freeze_encoder
        self.decoder_type = decoder_type

        if use_spectral_hypernetwork:
            self.grid_size = (img_size + 2 - patch_size) // patch_size + 1
            self.patch_padding = 1
        else:
            self.grid_size = img_size // patch_size
            self.patch_padding = 0
        self.num_patches = self.grid_size**2

        if wavelengths is not None:
            self.register_buffer("wavelengths", torch.tensor(wavelengths, dtype=torch.float32))
        else:
            self.register_buffer("wavelengths", torch.ones(n_channels, dtype=torch.float32) * 500.0)

        if bandwidths is not None:
            self.register_buffer("bandwidths", torch.tensor(bandwidths, dtype=torch.float32))
        else:
            self.register_buffer("bandwidths", torch.ones(n_channels, dtype=torch.float32) * 50.0)

        self.patch_embed: DynamicSpectralPatchEmbed | StaticPatchEmbed
        if use_spectral_hypernetwork:
            self.patch_embed = DynamicSpectralPatchEmbed(
                wv_planes=wv_planes,
                kernel_size=patch_size,
                embed_dim=embed_dim,
                use_variable_hypernetwork=use_variable_hypernetwork,
                variable_embed_dim=variable_embed_dim,
            )
        else:
            self.patch_embed = StaticPatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_channels=n_channels,
                embed_dim=embed_dim,
                norm_layer=nn.LayerNorm,
            )

        pos_embed = get_2d_sincos_pos_embed(embed_dim, self.grid_size)
        cls_pos = torch.zeros(1, embed_dim)
        self.register_buffer("pos_embed", torch.cat([cls_pos, pos_embed], dim=0).unsqueeze(0))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        out_indices = tuple(depth // 4 * (i + 1) - 1 for i in range(4))
        self.encoder = CopernicusEncoder(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            use_adapters=use_adapters,
            adapter_rank=adapter_rank,
            out_indices=out_indices,
        )

        in_channels = [embed_dim] * 4
        self.decoder: UPerNetDecoder | FPNDecoder | LinearDecoder
        if decoder_type == "upernet":
            self.decoder = UPerNetDecoder(in_channels=in_channels, channels=decoder_channels, num_classes=n_classes, dropout_rate=decoder_dropout)
        elif decoder_type == "fpn":
            self.decoder = FPNDecoder(in_channels=in_channels, channels=decoder_channels, num_classes=n_classes, dropout_rate=decoder_dropout)
        elif decoder_type == "linear":
            self.decoder = LinearDecoder(in_channels=embed_dim, num_classes=n_classes, dropout_rate=decoder_dropout)
        else:
            raise ValueError(f"Unknown decoder type: {decoder_type}")

        # Metadata encoders (Fourier + MLP)
        self.coord_fourier = FourierExpansion(1e-4, 720.0, assert_range=False)
        self.area_fourier = FourierExpansion(1e-3, 5.1e8, assert_range=False)
        self.time_fourier = FourierExpansion(1.0, 365.25, assert_range=False)

        self.coord_mlp = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))
        self.area_mlp = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))
        self.time_mlp = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))

        self.coord_missing_token = nn.Parameter(torch.zeros(1, embed_dim))
        self.area_missing_token = nn.Parameter(torch.zeros(1, embed_dim))
        self.time_missing_token = nn.Parameter(torch.zeros(1, embed_dim))

        self._init_weights()

        if use_pretrained:
            self._load_pretrained(pretrained_path)

        if freeze_encoder:
            self._freeze_encoder()
        elif freeze_encoder_layers > 0:
            self._freeze_encoder_layers(freeze_encoder_layers, mode=freeze_encoder_layers_mode)

    def _init_weights(self) -> None:
        """Initialize selected learnable parameters."""
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.coord_missing_token, std=0.02)
        nn.init.trunc_normal_(self.area_missing_token, std=0.02)
        nn.init.trunc_normal_(self.time_missing_token, std=0.02)

    def _get_pos_embed_for_tokens(self, num_patches: int, embed_dim: int, pos_embed: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Resize/create sine-cosine positional embeddings for a patch count."""
        if pos_embed.size(1) == num_patches + 1:
            return pos_embed.to(device)

        grid_new = int(num_patches**0.5)
        new_pos_no_cls = get_2d_sincos_pos_embed(embed_dim, grid_new).unsqueeze(0).to(device)
        cls_pos = pos_embed[:, :1, :].to(device)
        return torch.cat([cls_pos, new_pos_no_cls], dim=1)

    def _extract_metadata_tensor(
        self, metadata: dict[str, torch.Tensor] | None, key: str, batch_size: int, device: torch.device
    ) -> torch.Tensor | None:
        if metadata is None or key not in metadata:
            return None
        value = metadata[key]
        if not isinstance(value, torch.Tensor):
            return None
        if value.shape[0] != batch_size:
            return None
        return value.to(device)

    def _metadata_encoding(self, metadata: dict[str, torch.Tensor] | None, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Build additive metadata encoding (B, 1, E)."""
        if not self.use_metadata_encoding:
            return torch.zeros(batch_size, 1, self.embed_dim, device=device, dtype=dtype)

        coord_raw = self._extract_metadata_tensor(metadata, "coord", batch_size, device)
        lon_raw = self._extract_metadata_tensor(metadata, "lon", batch_size, device)
        lat_raw = self._extract_metadata_tensor(metadata, "lat", batch_size, device)
        area_raw = self._extract_metadata_tensor(metadata, "area", batch_size, device)
        time_raw = self._extract_metadata_tensor(metadata, "time", batch_size, device)

        drop_mask = None
        if self.training and self.metadata_dropout_prob > 0:
            drop_mask = torch.rand(batch_size, 3, device=device) < self.metadata_dropout_prob

        # Coord
        if coord_raw is None and lon_raw is not None and lat_raw is not None:
            coord_raw = torch.stack([lon_raw.view(batch_size), lat_raw.view(batch_size)], dim=-1)

        if coord_raw is None:
            coord_encoded = self.coord_missing_token.expand(batch_size, -1)
        else:
            coord_raw = coord_raw.view(batch_size, 2)
            lon_feat = self.coord_fourier(coord_raw[:, 0], self.embed_dim // 2)
            lat_feat = self.coord_fourier(coord_raw[:, 1], self.embed_dim // 2)
            coord_feat = torch.cat([lon_feat, lat_feat], dim=-1)
            coord_encoded = self.coord_mlp(coord_feat)

        # Area
        if area_raw is None:
            area_encoded = self.area_missing_token.expand(batch_size, -1)
        else:
            area_feat = self.area_fourier(area_raw.view(batch_size), self.embed_dim)
            area_encoded = self.area_mlp(area_feat)

        # Time
        if time_raw is None:
            time_encoded = self.time_missing_token.expand(batch_size, -1)
        else:
            time_feat = self.time_fourier(time_raw.view(batch_size), self.embed_dim)
            time_encoded = self.time_mlp(time_feat)

        if drop_mask is not None:
            coord_encoded = torch.where(drop_mask[:, 0:1], self.coord_missing_token.expand(batch_size, -1), coord_encoded)
            area_encoded = torch.where(drop_mask[:, 1:2], self.area_missing_token.expand(batch_size, -1), area_encoded)
            time_encoded = torch.where(drop_mask[:, 2:3], self.time_missing_token.expand(batch_size, -1), time_encoded)

        combined = coord_encoded + area_encoded + time_encoded
        return combined.unsqueeze(1).to(dtype=dtype)

    def _encode_full_sequence(
        self,
        x_tokens: torch.Tensor,
        metadata: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], int]:
        B, N, _ = x_tokens.shape
        pos_embed = self._get_pos_embed_for_tokens(N, self.embed_dim, cast(torch.Tensor, self.pos_embed), x_tokens.device)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        meta_enc = self._metadata_encoding(metadata, B, x_tokens.device, x_tokens.dtype)

        cls_tokens = cls_tokens + pos_embed[:, :1, :] + meta_enc
        x_tokens = x_tokens + pos_embed[:, 1 : N + 1, :] + meta_enc.expand(-1, N, -1)

        x_seq = torch.cat([cls_tokens, x_tokens], dim=1)
        x_seq = self.pos_drop(x_seq)
        features = self.encoder(x_seq)
        latent = features[-1]
        return latent, features, N

    def _resolve_pretrained_filename(self) -> str:
        """Resolve the correct checkpoint filename based on the model variant."""
        variant_to_filename: dict[str, str] = {
            "large": "CopernicusFM_ViT_large_varlang_e100.pth",
            "base": "CopernicusFM_ViT_base_varlang_e100.pth",
            "small": "CopernicusFM_ViT_base_varlang_e100.pth",  # no small checkpoint; fall back to base
        }
        if self.embed_dim == 1024:
            return variant_to_filename["large"]
        return variant_to_filename["base"]

    def _load_pretrained(self, pretrained_path: str) -> None:
        """Load pretrained weights from HuggingFace."""
        try:
            from huggingface_hub import hf_hub_download

            filename = self._resolve_pretrained_filename()
            logger.info(f"Loading pretrained weights from {pretrained_path} (file: {filename})")
            weights_file = hf_hub_download(repo_id=pretrained_path, filename=filename)
            state_dict = torch.load(weights_file, map_location="cpu", weights_only=False)

            if "model" in state_dict:
                state_dict = state_dict["model"]

            adapted_state_dict = self._adapt_pretrained_weights(state_dict)
            missing, unexpected = self.load_state_dict(adapted_state_dict, strict=False)

            log_pretrained_load_result(
                missing,
                unexpected,
                logger,
                model_name="CopernicusFM",
                expected_missing_prefixes=["decoder."],
            )

            logger.info("Successfully loaded pretrained weights")

        except ImportError:
            logger.warning("huggingface_hub not installed. Install with: pip install huggingface_hub")
        except Exception as e:
            logger.warning(f"Failed to load pretrained weights: {e}")
            logger.info("Continuing with random initialization")

    def _adapt_pretrained_weights(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Adapt official checkpoint keys to this codebase naming."""
        adapted: dict[str, torch.Tensor] = {}

        skip_prefixes = (
            "decoder_embed",
            "decoder_blocks",
            "decoder_pred",
            "decoder_pos_embed",
            "mask_token",
            # Single-linear FC layers from checkpoint don't match our 2-layer MLP structure
            "coord_fc",
            "scale_fc",
            "time_fc",
        )

        # Missing-token shape in the checkpoint is [1, 1, E]; ours is [1, E] — squeeze dim 1
        missing_token_map = {
            "coord_token": "coord_missing_token",
            "scale_token": "area_missing_token",
            "time_token": "time_missing_token",
        }

        for key, value in state_dict.items():
            if any(key.startswith(p) for p in skip_prefixes):
                continue

            # pos_embed grid size may differ — skip and rely on sincos recomputation
            if key == "pos_embed":
                continue

            # Remap missing-token keys and fix shape [1,1,E] -> [1,E]
            if key in missing_token_map:
                new_key = missing_token_map[key]
                adapted[new_key] = value.squeeze(1)
                continue

            new_key = key

            if key.startswith("patch_embed_spectral."):
                new_key = key.replace("patch_embed_spectral.", "patch_embed.spectral_hypernetwork.")
            elif key.startswith("patch_embed_variable."):
                new_key = key.replace("patch_embed_variable.", "patch_embed.variable_hypernetwork.")
            elif key.startswith("blocks."):
                new_key = key.replace("blocks.", "encoder.blocks.")

            adapted[new_key] = value

        return adapted

    def _freeze_encoder(self) -> None:
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        for param in self.encoder.parameters():
            param.requires_grad = False

        for param in self.coord_mlp.parameters():
            param.requires_grad = False
        for param in self.area_mlp.parameters():
            param.requires_grad = False
        for param in self.time_mlp.parameters():
            param.requires_grad = False

        self.cls_token.requires_grad = False
        self.coord_missing_token.requires_grad = False
        self.area_missing_token.requires_grad = False
        self.time_missing_token.requires_grad = False

        logger.info("Encoder weights frozen")

    def _freeze_encoder_layers(self, n_layers: int, mode: str = "individual") -> None:
        """Freeze first N layers of the encoder."""
        if n_layers <= 0:
            return

        if mode == "individual":
            max_layers = 1 + len(self.encoder.blocks)
            if n_layers > max_layers:
                logger.warning(f"freeze_encoder_layers={n_layers} exceeds max layers {max_layers}, using {max_layers}")
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
            group_size = max(1, num_blocks // 3)
            max_layers = 4
            if n_layers > max_layers:
                logger.warning(f"freeze_encoder_layers={n_layers} exceeds max groups {max_layers}, using {max_layers}")
                n_layers = max_layers

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
        else:
            logger.warning(f"Unknown freeze_encoder_layers_mode '{mode}', using 'individual'")
            self._freeze_encoder_layers(n_layers, mode="individual")
            return

        logger.info(f"Froze first {frozen_count} encoder layers")

    def get_frozen_layer_info(self) -> dict[str, int | list[str]]:
        """Get information about frozen layers for logging."""
        layer_groups: list[tuple[str, list[nn.Module]]] = [("patch_embed", [cast(nn.Module, self.patch_embed)])]
        for i, block in enumerate(self.encoder.blocks):
            layer_groups.append((f"block_{i}", [block]))

        frozen_layers = []
        for layer_name, modules in layer_groups:
            is_frozen = all(not param.requires_grad for module in modules for param in module.parameters())
            if is_frozen:
                frozen_layers.append(layer_name)

        return {"total_layers": len(layer_groups), "frozen_layers": len(frozen_layers), "frozen_layer_names": frozen_layers}

    def _get_patch_tokens(self, x: torch.Tensor, variable_embeddings: torch.Tensor | None = None) -> torch.Tensor:
        if self.use_spectral_hypernetwork:
            assert isinstance(self.patch_embed, DynamicSpectralPatchEmbed)
            x_tokens, _ = self.patch_embed(x, self.wavelengths, self.bandwidths, variable_embeddings=variable_embeddings)
            return cast(torch.Tensor, x_tokens)

        assert isinstance(self.patch_embed, StaticPatchEmbed)
        return cast(torch.Tensor, self.patch_embed(x))

    def _forward_segmentation(
        self,
        x: torch.Tensor,
        metadata: dict[str, torch.Tensor] | None = None,
        variable_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, _, H, W = x.shape
        x_tokens = self._get_patch_tokens(x, variable_embeddings=variable_embeddings)

        _, features, _ = self._encode_full_sequence(x_tokens, metadata=metadata)

        spatial_features = []
        for feat in features:
            feat = feat[:, 1:, :]
            feat = feat.transpose(1, 2)
            feat = feat.reshape(B, -1, self.grid_size, self.grid_size)
            spatial_features.append(feat)

        return cast(torch.Tensor, self.decoder(spatial_features, output_size=(H, W)))

    def forward(
        self,
        x: torch.Tensor,
        metadata: dict[str, torch.Tensor] | None = None,
        *,
        variable_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass for segmentation."""
        return self._forward_segmentation(x, metadata=metadata, variable_embeddings=variable_embeddings)

    def get_encoder_params(self) -> Iterator[nn.Parameter]:
        """Get encoder-side parameters for differential learning rates."""
        yield from self.patch_embed.parameters()
        yield self.cls_token
        yield from self.encoder.parameters()
        yield from self.coord_mlp.parameters()
        yield from self.area_mlp.parameters()
        yield from self.time_mlp.parameters()
        yield self.coord_missing_token
        yield self.area_missing_token
        yield self.time_missing_token

    def get_decoder_params(self) -> Iterator[nn.Parameter]:
        yield from self.decoder.parameters()

    def get_param_groups(self, encoder_lr: float, decoder_lr: float) -> list[dict[str, Any]]:
        """Get optimizer parameter groups with differential learning rates."""
        return [
            {"params": list(self.get_encoder_params()), "lr": encoder_lr},
            {"params": list(self.get_decoder_params()), "lr": decoder_lr},
        ]
