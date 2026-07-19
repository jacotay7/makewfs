"""Wavefront units, coordinates, and input validation."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .backend import ArrayBackend, cpu_backend
from .config import WFSConfig


def _coordinates(
    shape: tuple[int, int], extent_m: float, *, backend: ArrayBackend | None = None
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return centered physical ``(x, y)`` coordinates for an array shape."""
    resolved = backend or cpu_backend()
    height, width = shape
    x = (resolved.arange(width, dtype=np.float64) - (width - 1) / 2.0) * extent_m / width
    y = (resolved.arange(height, dtype=np.float64) - (height - 1) / 2.0) * extent_m / height
    xx, yy = resolved.meshgrid(x, y)
    return xx, yy


def load_static_opd(config: WFSConfig) -> NDArray[np.float64] | None:
    """Load the optional static OPD map and validate its shape."""
    path = config.input.static_opd_path
    if path is None:
        return None
    source = Path(path)
    if source.suffix.lower() == ".npy":
        array = np.load(source)
    elif source.suffix.lower() == ".npz":
        archive = np.load(source)
        if not archive.files:
            raise ValueError(f"static OPD archive {source} contains no arrays")
        array = archive[archive.files[0]]
    elif source.suffix.lower() in {".fits", ".fit"}:
        try:
            from astropy.io import fits  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover - optional file format
            raise ImportError("reading FITS OPD maps requires astropy") from exc
        array = fits.getdata(source)
    else:
        raise ValueError(f"unsupported static OPD format {source.suffix!r}")
    result = np.asarray(array, dtype=np.float64)
    if result.shape != config.input.shape:
        raise ValueError(
            f"static OPD shape {result.shape} does not match input.shape {config.input.shape}"
        )
    if not np.all(np.isfinite(result)):
        raise ValueError("static OPD contains non-finite values")
    return result


class WavefrontInput:
    """Validated OPD input on the configured physical grid."""

    def __init__(
        self,
        config: WFSConfig,
        static_opd: NDArray[np.float64] | None = None,
        *,
        backend: ArrayBackend | None = None,
    ) -> None:
        self.config = config
        self.backend = backend or cpu_backend()
        self.static_opd = (
            None if static_opd is None else self.backend.asarray(static_opd, dtype=np.float64)
        )
        self._input_coordinates = _coordinates(
            config.input.shape,
            config.input.grid_extent_m,
            backend=self.backend,
        )

    def opd(
        self, value: ArrayLike, *, target_shape: tuple[int, int] | None = None
    ) -> NDArray[np.float64]:
        """Validate, convert to OPD metres, add static OPD, and optionally regrid."""
        array = self.backend.asarray(value)
        if array.ndim != 2 or tuple(array.shape) != self.config.input.shape:
            raise ValueError(
                f"wavefront must have shape {self.config.input.shape}, got {array.shape}"
            )
        if not np.issubdtype(array.dtype, np.number):
            raise TypeError("wavefront must be numeric")
        converted = self.backend.asarray(array, dtype=np.float64)
        if self.config.input.quantity == "phase":
            assert self.config.input.reference_wavelength_m is not None
            converted = converted * self.config.input.reference_wavelength_m / (2.0 * np.pi)
        if self.static_opd is not None:
            converted = converted + self.static_opd
        if not self.backend.scalar(self.backend.all(self.backend.isfinite(converted))):
            raise ValueError("wavefront contains non-finite OPD")
        if target_shape is not None and tuple(target_shape) != converted.shape:
            converted = resample_opd(
                converted,
                target_shape,
                self.config.input.grid_extent_m,
                backend=self.backend,
            )
        return cast(NDArray[np.float64], converted)

    def validate_finite_inside(self, opd: NDArray[np.float64], pupil: NDArray[np.float64]) -> None:
        """Reject non-finite OPD only where the configured pupil is illuminated."""
        if not self.backend.scalar(self.backend.all(self.backend.isfinite(opd[pupil > 0]))):
            raise ValueError("wavefront contains non-finite OPD inside the illuminated pupil")


def resample_opd(
    opd: NDArray[np.float64],
    target_shape: tuple[int, int],
    extent_m: float,
    *,
    backend: ArrayBackend | None = None,
) -> NDArray[np.float64]:
    """Resample OPD on physical coordinates without wrapping phase.

    Linear interpolation is intentional for the first CPU path: it is stable
    for arbitrary OPD maps and preserves a phase ramp exactly up to floating
    point error. Higher-order or band-limited resampling can be added behind the
    same contract later.
    """
    resolved = backend or cpu_backend()
    source_height, source_width = opd.shape
    target_height, target_width = target_shape
    source_y = (
        resolved.arange(target_height, dtype=np.float64) + 0.5
    ) * source_height / target_height - 0.5
    source_x = (
        resolved.arange(target_width, dtype=np.float64) + 0.5
    ) * source_width / target_width - 0.5
    yy, xx = resolved.meshgrid(source_y, source_x, indexing="ij")
    return cast(
        NDArray[np.float64],
        resolved.map_coordinates(opd, [yy, xx], order=1, mode="nearest"),
    )


def iter_phase_samples(
    value: ArrayLike | Iterable[ArrayLike],
    shape: tuple[int, int],
    *,
    backend: ArrayBackend | None = None,
) -> Iterable[ArrayLike]:
    """Yield one or more samples for an integrated exposure."""
    resolved = backend or cpu_backend()
    array = resolved.asarray(value) if not isinstance(value, (list, tuple)) else None
    if array is not None and array.ndim == 3:
        if tuple(array.shape[1:]) != shape:
            raise ValueError(f"integrated wavefront stack must end in shape {shape}")
        yield from array
        return
    if array is not None and array.ndim == 2:
        yield array
        return
    for sample in value:  # type: ignore[union-attr]
        sample_array = resolved.asarray(sample)
        if sample_array.shape != shape:
            raise ValueError(f"integrated wavefront sample must have shape {shape}")
        yield sample_array


__all__ = [
    "WavefrontInput",
    "_coordinates",
    "iter_phase_samples",
    "load_static_opd",
    "resample_opd",
]
