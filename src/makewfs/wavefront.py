"""Wavefront units, coordinates, and input validation."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import map_coordinates

from .config import WFSConfig


def _coordinates(
    shape: tuple[int, int], extent_m: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return centered physical ``(x, y)`` coordinates for an array shape."""
    height, width = shape
    x = (np.arange(width, dtype=np.float64) - (width - 1) / 2.0) * extent_m / width
    y = (np.arange(height, dtype=np.float64) - (height - 1) / 2.0) * extent_m / height
    xx, yy = np.meshgrid(x, y)
    return np.asarray(xx, dtype=np.float64), np.asarray(yy, dtype=np.float64)


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

    def __init__(self, config: WFSConfig, static_opd: NDArray[np.float64] | None = None) -> None:
        self.config = config
        self.static_opd = static_opd
        self._input_coordinates = _coordinates(config.input.shape, config.input.grid_extent_m)

    def opd(
        self, value: ArrayLike, *, target_shape: tuple[int, int] | None = None
    ) -> NDArray[np.float64]:
        """Validate, convert to OPD metres, add static OPD, and optionally regrid."""
        array = np.asarray(value)
        if array.ndim != 2 or tuple(array.shape) != self.config.input.shape:
            raise ValueError(
                f"wavefront must have shape {self.config.input.shape}, got {array.shape}"
            )
        if not np.issubdtype(array.dtype, np.number):
            raise TypeError("wavefront must be numeric")
        converted = np.asarray(array, dtype=np.float64)
        if self.config.input.quantity == "phase":
            assert self.config.input.reference_wavelength_m is not None
            converted = converted * self.config.input.reference_wavelength_m / (2.0 * np.pi)
        if self.static_opd is not None:
            converted = converted + self.static_opd
        if not np.all(np.isfinite(converted)):
            raise ValueError("wavefront contains non-finite OPD")
        if target_shape is not None and tuple(target_shape) != converted.shape:
            converted = resample_opd(converted, target_shape, self.config.input.grid_extent_m)
        return converted

    def validate_finite_inside(self, opd: NDArray[np.float64], pupil: NDArray[np.float64]) -> None:
        """Reject non-finite OPD only where the configured pupil is illuminated."""
        if np.any(~np.isfinite(opd[pupil > 0])):
            raise ValueError("wavefront contains non-finite OPD inside the illuminated pupil")


def resample_opd(
    opd: NDArray[np.float64], target_shape: tuple[int, int], extent_m: float
) -> NDArray[np.float64]:
    """Resample OPD on physical coordinates without wrapping phase.

    Linear interpolation is intentional for the first CPU path: it is stable
    for arbitrary OPD maps and preserves a phase ramp exactly up to floating
    point error. Higher-order or band-limited resampling can be added behind the
    same contract later.
    """
    source_height, source_width = opd.shape
    target_height, target_width = target_shape
    source_y = (
        np.arange(target_height, dtype=np.float64) + 0.5
    ) * source_height / target_height - 0.5
    source_x = (np.arange(target_width, dtype=np.float64) + 0.5) * source_width / target_width - 0.5
    yy, xx = np.meshgrid(source_y, source_x, indexing="ij")
    return np.asarray(map_coordinates(opd, [yy, xx], order=1, mode="nearest"), dtype=opd.dtype)


def iter_phase_samples(
    value: ArrayLike | Iterable[ArrayLike], shape: tuple[int, int]
) -> Iterable[ArrayLike]:
    """Yield one or more samples for an integrated exposure."""
    array = np.asarray(value) if not isinstance(value, (list, tuple)) else None
    if array is not None and array.ndim == 3:
        if tuple(array.shape[1:]) != shape:
            raise ValueError(f"integrated wavefront stack must end in shape {shape}")
        yield from array
        return
    if array is not None and array.ndim == 2:
        yield array
        return
    for sample in value:  # type: ignore[union-attr]
        sample_array = np.asarray(sample)
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
