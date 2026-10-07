"""Wavefront units, coordinates, and input validation."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import numpy as np
from aocore import phase_to_opd
from numpy.typing import ArrayLike, NDArray

from .backend import ArrayBackend, cpu_backend
from .config import WFSConfig


def _coordinates(
    shape: tuple[int, int],
    extent_m: float,
    *,
    backend: ArrayBackend | None = None,
    dtype: object = np.float64,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return centered physical ``(x, y)`` coordinates for an array shape.

    Pixel centres follow ``aocore.centered_coordinates`` (CONVENTIONS 1.2),
    built on the backend's device. The unit-pitch offsets are exact in either
    precision; scaling to metres happens on the backend in ``dtype``.
    """
    resolved = backend or cpu_backend()
    height, width = shape
    x = resolved.centered_coordinates(width, dtype=dtype) * extent_m / width
    y = resolved.centered_coordinates(height, dtype=dtype) * extent_m / height
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
        self._resample_coordinates: dict[tuple[int, int], Any] = {}

    def input_opd(self, value: ArrayLike) -> NDArray[np.float64]:
        """The input wavefront in OPD metres, before static OPD and regridding.

        OPD input is returned as given. Phase input is converted at the
        configured reference wavelength.
        """
        if self.config.input.quantity != "phase":
            return cast(NDArray[np.float64], value)
        assert self.config.input.reference_wavelength_m is not None
        return cast(
            NDArray[np.float64],
            phase_to_opd(
                self.backend.asarray(value, dtype=np.float64),
                self.config.input.reference_wavelength_m,
            ),
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
            converted = phase_to_opd(converted, self.config.input.reference_wavelength_m)
        if self.static_opd is not None:
            converted = converted + self.static_opd
        if not self.backend.scalar(self.backend.all(self.backend.isfinite(converted))):
            raise ValueError("wavefront contains non-finite OPD")
        if target_shape is not None and tuple(target_shape) != converted.shape:
            coordinates = self._resample_coordinates.get(target_shape)
            if coordinates is None:
                coordinates = _resampling_coordinates(
                    converted.shape,
                    target_shape,
                    backend=self.backend,
                    dtype=converted.dtype,
                )
                self._resample_coordinates[target_shape] = coordinates
            converted = resample_opd(
                converted,
                target_shape,
                self.config.input.grid_extent_m,
                backend=self.backend,
                coordinates=coordinates,
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
    coordinates: Any | None = None,
) -> NDArray[np.float64]:
    """Resample OPD on physical coordinates without wrapping phase.

    Linear interpolation is intentional for the first CPU path: it is stable
    for arbitrary OPD maps and preserves a phase ramp exactly up to floating
    point error. Higher-order or band-limited resampling can be added behind the
    same contract later.
    """
    resolved = backend or cpu_backend()
    sample_coordinates = (
        _resampling_coordinates(
            cast(tuple[int, int], opd.shape),
            target_shape,
            backend=resolved,
            dtype=opd.dtype,
        )
        if coordinates is None
        else coordinates
    )
    return cast(
        NDArray[np.float64],
        resolved.map_coordinates(opd, sample_coordinates, order=1, mode="nearest"),
    )


def _resampling_coordinates(
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
    *,
    backend: ArrayBackend,
    dtype: object,
) -> Any:
    """Build a backend-native interpolation grid for one immutable shape pair."""
    source_height, source_width = source_shape
    target_height, target_width = target_shape
    source_y = (
        backend.arange(target_height, dtype=dtype) + 0.5
    ) * source_height / target_height - 0.5
    source_x = (backend.arange(target_width, dtype=dtype) + 0.5) * source_width / target_width - 0.5
    yy, xx = backend.meshgrid(source_y, source_x, indexing="ij")
    return backend.stack((yy, xx), axis=0)


def pupil_weights(intensity: NDArray[Any], *, backend: ArrayBackend | None = None) -> Any:
    """Normalize a pupil intensity map into RMS weights that sum to one.

    Parameters
    ----------
    intensity
        Non-negative pupil intensity (amplitude squared) on the OPD grid.
    backend
        Array backend holding ``intensity``.

    Returns
    -------
    numpy.ndarray or cupy.ndarray
        Float64 weights on the same backend.

    Raises
    ------
    ValueError
        If the pupil transmits nothing on this grid.
    """
    resolved = backend or cpu_backend()
    weights = resolved.asarray(intensity, dtype=np.float64)
    total = resolved.sum(weights)
    if not resolved.scalar(total) > 0.0:
        raise ValueError("pupil has no illuminated pixels on the input grid")
    return weights / total


def pupil_rms(opd: Any, weights: Any, *, backend: ArrayBackend | None = None) -> Any:
    """Pupil-weighted, piston-removed OPD RMS, left on the device.

    Implements ``rms`` of aocore CONVENTIONS 4.1,
    ``sqrt(sum a^2 (opd - <opd>_a)^2 / sum a^2)``, where ``weights`` is the
    intensity ``a^2`` already normalized by :func:`pupil_weights`. The piston is
    subtracted before squaring rather than taken from ``<opd^2> - <opd>^2``, so
    a large piston does not cancel away the residual's precision and a
    piston-only input gives zero to rounding.

    ``aocore.rms`` defines the same quantity but reduces on the host with
    NumPy; this stays on the selected backend so frame metadata needs no extra
    device-to-host copy. The tests check it with ``aocore.conformance.check_rms``.

    Parameters
    ----------
    opd
        OPD in metres on the same grid as ``weights``.
    weights
        Normalized pupil intensity weights.
    backend
        Array backend holding both arrays.

    Returns
    -------
    numpy.ndarray or cupy.ndarray
        A zero-dimensional float64 array; no host synchronization happens here.
    """
    resolved = backend or cpu_backend()
    values = resolved.asarray(opd, dtype=np.float64)
    residual = values - resolved.sum(weights * values)
    return resolved.sqrt(resolved.sum(weights * residual * residual))


def grid_rms(opd: Any, *, backend: ArrayBackend | None = None) -> Any:
    """Unweighted OPD RMS over the whole grid, piston included, on the device.

    This is ``rms_unweighted`` in the sense of aocore CONVENTIONS 4.1: every
    grid pixel counts equally, including pixels outside the pupil, and the mean
    is not removed. It equals ``aocore.rms_unweighted(opd)``, which returns a
    host float and so would synchronize on its own; this returns a device
    scalar for the one batched metadata crossing.
    """
    resolved = backend or cpu_backend()
    values = resolved.asarray(opd, dtype=np.float64)
    return resolved.sqrt(resolved.mean(values * values))


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
    "grid_rms",
    "iter_phase_samples",
    "load_static_opd",
    "pupil_rms",
    "pupil_weights",
    "resample_opd",
]
