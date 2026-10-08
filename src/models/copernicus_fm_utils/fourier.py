# Implemented from: Wang et al., "Towards a Unified Copernicus Foundation Model for Earth Vision",
#   arXiv:2503.11849, 2025
# Adapted from: https://github.com/zhu-xlab/Copernicus-FM/blob/main/Copernicus-FM/src/aurora/fourier.py
#   MIT License, Copyright (c) Microsoft Corporation (originally https://github.com/microsoft/aurora); modified for this project.

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn


class FourierExpansion(nn.Module):
    """Fourier feature expansion for continuous scalar inputs.

    Maps input values to high-dimensional sin/cos representations using
    log-spaced frequency bands. This enables neural networks to learn
    high-frequency functions in low-dimensional domains.

    The encoding is:
        FE(x) = [cos(2*pi*x / omega_i), sin(2*pi*x / omega_i)]
    where omega_i are log-spaced between lower and upper bounds.

    Args:
        lower: Minimum wavelength bound for frequency bands.
        upper: Maximum wavelength bound for frequency bands.
        assert_range: Whether to validate that inputs fall within
            the configured [lower, upper] range.
    """

    def __init__(
        self,
        lower: float,
        upper: float,
        assert_range: bool = True,
    ) -> None:
        super().__init__()
        self.lower = lower
        self.upper = upper
        self.assert_range = assert_range

    def forward(self, x: torch.Tensor, d: int) -> torch.Tensor:
        """Encode input values using Fourier features.

        Args:
            x: Input tensor of shape (..., n) with values to encode.
            d: Output dimensionality per input element (must be even).

        Returns:
            Fourier encoding of shape (..., n, d) as float32.

        Raises:
            AssertionError: If assert_range is True and values are outside
                the configured [lower, upper] range.
            ValueError: If d is not even.
        """
        if self.assert_range:
            in_range = torch.logical_and(
                self.lower <= x.abs(),
                x.abs() <= self.upper,
            )
            in_range_or_zero = torch.all(torch.logical_or(in_range, x == 0))
            if not in_range_or_zero:
                raise AssertionError(f"The input tensor is not within the configured range `[{self.lower}, {self.upper}]`.")

        if d % 2 != 0:
            raise ValueError(f"The dimensionality must be even, got {d}.")

        # Use float64 for precision in frequency computation
        x = x.double()

        # Log-spaced wavelengths from lower to upper bound
        wavelengths = torch.logspace(
            math.log10(self.lower),
            math.log10(self.upper),
            d // 2,
            base=10,
            device=x.device,
            dtype=x.dtype,
        )

        # Compute sin/cos features via einsum
        prod = torch.einsum("...i,j->...ij", x, 2 * np.pi / wavelengths)
        encoding = torch.cat([torch.cos(prod), torch.sin(prod)], dim=-1)

        return encoding.float()
