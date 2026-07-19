"""Internal protocol shared by WFS optical engines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..backend import ArrayBackend, cpu_backend
from ..source import SourceState


@dataclass(frozen=True)
class OpticalResult:
    """Ideal detector-plane result before detector noise."""

    photon_rate: NDArray[np.float64]
    launched_rate_per_s: float
    captured_rate_per_s: float
    opd_m: NDArray[np.float64]
    spectral_photon_rate: NDArray[np.float64] | None = None
    spectral_wavelengths_m: tuple[float, ...] | None = None


class SensorEngine:
    """Structural interface implemented by each deterministic optical engine."""

    kind: str
    output_shape: tuple[int, int]
    source_states: tuple[SourceState, ...]
    file_digests: dict[str, str]
    backend: ArrayBackend

    @staticmethod
    def resolve_backend(backend: ArrayBackend | None) -> ArrayBackend:
        """Resolve the private backend injection point."""
        return backend or cpu_backend()

    def render(self, wavefront: NDArray[np.float64]) -> OpticalResult:
        """Render one validated OPD input."""
        raise NotImplementedError


__all__ = ["OpticalResult", "SensorEngine"]
