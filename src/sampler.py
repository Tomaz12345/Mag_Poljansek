import logging
from collections.abc import Iterator, Sized
from typing import Any

import torch
from torch.utils.data import Sampler

from .utils import log_and_raise

logger = logging.getLogger(__name__)


class ResumableRandomSampler(Sampler[int]):
    """A random sampler that can be resumed from a specific position.

    Args:
        data_source: Dataset to sample from (must have __len__).
        generator: Optional random generator for reproducibility.

    Raises:
        ValueError: If data_source is not Sized or is empty.
    """

    def __init__(self, data_source: Sized, generator: torch.Generator | None = None):
        if not hasattr(data_source, "__len__"):
            msg = f"data_source must be a Sized object (have __len__), got {type(data_source).__name__}"
            log_and_raise(ValueError, msg, logger_instance=logger)

        num_samples = len(data_source)
        if num_samples == 0:
            msg = "data_source must not be empty. Received dataset with length 0."
            log_and_raise(ValueError, msg, logger_instance=logger)

        self.data_source = data_source
        self.generator = generator
        self._num_samples = num_samples
        self._start_idx = 0
        self._perm: list[int] | None = None
        self._current_idx = 0  # Tracks absolute position for external use

    def _generate_perm(self) -> None:
        if self.generator is not None:
            self._perm = torch.randperm(self._num_samples, generator=self.generator).tolist()
        else:
            self._perm = torch.randperm(self._num_samples).tolist()

    def __iter__(self) -> Iterator[int]:
        if self._perm is None:
            self._generate_perm()

        assert self._perm is not None

        for i in range(self._start_idx, len(self._perm)):
            self._current_idx = i  # Track current absolute index
            self._start_idx = i + 1
            yield self._perm[i]

        # Reset for next epoch
        self._start_idx = 0
        self._current_idx = 0
        self._perm = None

    def __len__(self) -> int:
        if self._perm is None:
            return self._num_samples
        return max(0, len(self._perm) - self._start_idx)

    def reset(self) -> None:
        """Reset sampler to the beginning for a new epoch."""
        self._start_idx = 0
        self._current_idx = 0

    def get_state(self) -> dict[str, Any]:
        return {
            "perm": self._perm,
            "start_idx": self._start_idx,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore sampler state from a previously saved state.

        Args:
            state: Dictionary containing 'perm' and 'start_idx' keys.

        Raises:
            ValueError: If start_idx is invalid for the current dataset.
        """
        self._perm = state.get("perm")
        start_idx = state.get("start_idx", 0)

        if start_idx < 0 or start_idx > self._num_samples:
            msg = f"start_idx must be non-negative and less than dataset size. Got start_idx={start_idx}, dataset_size={self._num_samples}"
            log_and_raise(ValueError, msg, logger_instance=logger)

        self._start_idx = start_idx
        self._current_idx = self._start_idx
