from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

INPUT_PROJECTION_TYPES: tuple[str, ...] = ("linear", "mlp", "fc", "fc_mlp")


def _parse_dims(dims: str | list[int] | None) -> list[int] | None:
    """Parse a comma-separated string of ints into a list, or pass through a list."""
    if dims is None:
        return None
    if isinstance(dims, list):
        return dims
    parsed = [int(d.strip()) for d in dims.split(",") if d.strip()]
    return parsed if parsed else None


class InputProjection(nn.Module):
    """Learned spectral projection: n_channels -> target_channels.

    Four variants:
    - ``"linear"``: single 1x1 Conv2d (no activation). A learned linear
      combination of input channels. Minimal parameters, proven in OneFormer.
    - ``"mlp"``: stack of 1x1 Conv2d layers with BatchNorm + ReLU between them.
      Hidden layer sizes are controlled by ``dims`` (e.g. ``[32, 64]`` gives two
      hidden layers). Falls back to ``hidden_channels`` for a single hidden layer
      when ``dims`` is ``None``.
    - ``"fc"``: single ``nn.Linear`` applied per-pixel (channel projection).
      Equivalent to ``"linear"`` but implemented with ``nn.Linear``.
    - ``"fc_mlp"``: stack of ``nn.Linear`` layers with ReLU between them,
      applied per-pixel. Hidden layer sizes controlled by ``dims``; falls back
      to ``hidden_channels`` for a single hidden layer when ``dims`` is ``None``.

    The module is a no-op when ``n_channels == target_channels`` (returns input
    unchanged, no parameters created).

    Args:
        n_channels: Number of input channels from the dataset.
        target_channels: Number of output channels expected by the backbone.
        projection_type: ``"linear"``, ``"mlp"``, ``"fc"``, or ``"fc_mlp"``.
        hidden_channels: Hidden dimension fallback for ``"mlp"`` / ``"fc_mlp"``
            when ``dims`` is ``None``.
        dims: Explicit hidden layer sizes as a comma-separated string (e.g.
            ``"32,64"``) or a list of ints. Overrides ``hidden_channels`` for
            multi-layer projections. Ignored for ``"linear"`` and ``"fc"``.
    """

    def __init__(
        self,
        n_channels: int,
        target_channels: int,
        projection_type: str = "linear",
        hidden_channels: int = 32,
        dims: str | list[int] | None = None,
    ) -> None:
        super().__init__()

        self.n_channels = n_channels
        self.target_channels = target_channels
        self.projection_type = projection_type
        self.identity = n_channels == target_channels
        self._use_fc = projection_type in ("fc", "fc_mlp")

        if self.identity:
            self.proj: nn.Module = nn.Identity()
            logger.debug(f"InputProjection: identity (n_channels={n_channels} == target_channels={target_channels})")
            return

        parsed_dims = _parse_dims(dims)

        if projection_type == "linear":
            self.proj = nn.Conv2d(n_channels, target_channels, kernel_size=1, bias=False)
            nn.init.kaiming_uniform_(self.proj.weight, nonlinearity="linear")

        elif projection_type == "mlp":
            hidden_dims = parsed_dims if parsed_dims is not None else [hidden_channels]
            layer_sizes = [n_channels] + hidden_dims + [target_channels]
            layers: list[nn.Module] = []
            for i in range(len(layer_sizes) - 1):
                is_last = i == len(layer_sizes) - 2
                layers.append(nn.Conv2d(layer_sizes[i], layer_sizes[i + 1], kernel_size=1, bias=False))
                if not is_last:
                    layers.append(nn.BatchNorm2d(layer_sizes[i + 1]))
                    layers.append(nn.ReLU(inplace=True))
            self.proj = nn.Sequential(*layers)
            for m in self.proj.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")

        elif projection_type == "fc":
            self.proj = nn.Linear(n_channels, target_channels, bias=False)
            nn.init.kaiming_uniform_(self.proj.weight, nonlinearity="linear")

        elif projection_type == "fc_mlp":
            hidden_dims = parsed_dims if parsed_dims is not None else [hidden_channels]
            layer_sizes = [n_channels] + hidden_dims + [target_channels]
            fc_layers: list[nn.Module] = []
            for i in range(len(layer_sizes) - 1):
                is_last = i == len(layer_sizes) - 2
                fc_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1], bias=False))
                if not is_last:
                    fc_layers.append(nn.ReLU(inplace=True))
            self.proj = nn.Sequential(*fc_layers)
            for m in self.proj.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")

        else:
            raise ValueError(f"projection_type must be one of {INPUT_PROJECTION_TYPES}, got '{projection_type}'")

        dims_str = str(parsed_dims) if parsed_dims is not None else f"[{hidden_channels}]"
        logger.info(
            f"InputProjection({projection_type}): {n_channels} -> {target_channels} channels"
            + (f" (hidden={dims_str})" if projection_type in ("mlp", "fc_mlp") else "")
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_fc and not self.identity:
            # x: (B, C, H, W) -> permute to (B, H, W, C) for nn.Linear
            x = x.permute(0, 2, 3, 1)
            x = self.proj(x)
            result: torch.Tensor = x.permute(0, 3, 1, 2)
            return result
        result = self.proj(x)
        return result
