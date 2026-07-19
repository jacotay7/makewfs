"""Internal protocol shared by WFS optical engines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class OpticalResult:
    """Ideal detector-plane result before detector noise."""

    photon_rate: NDArray[np.float64]
    launched_rate_per_s: float
    captured_rate_per_s: float
    opd_m: NDArray[np.float64]


class SensorEngine:
    """Structural interface implemented by each deterministic optical engine."""

    kind: str
    output_shape: tuple[int, int]

    def render(self, wavefront: NDArray[np.float64]) -> OpticalResult:
        """Render one validated OPD input."""
        raise NotImplementedError


__all__ = ["OpticalResult", "SensorEngine"]
