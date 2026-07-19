"""Analytic and custom entrance-pupil masks."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from .backend import ArrayBackend, cpu_backend
from .config import TelescopeConfig
from .wavefront import _coordinates


def _load_mask(path: str, shape: tuple[int, int]) -> NDArray[np.float64]:
    source = Path(path)
    if source.suffix.lower() == ".npy":
        array = np.load(source)
    elif source.suffix.lower() == ".npz":
        archive = np.load(source)
        if not archive.files:
            raise ValueError(f"custom pupil archive {source} contains no arrays")
        array = archive[archive.files[0]]
    elif source.suffix.lower() in {".fits", ".fit"}:
        try:
            from astropy.io import fits  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover - optional file format
            raise ImportError("reading FITS pupil masks requires astropy") from exc
        array = fits.getdata(source)
    else:
        raise ValueError("custom pupil masks support .npy, .npz, or FITS")
    result = np.asarray(array, dtype=np.float64)
    if result.shape != shape:
        raise ValueError(f"custom pupil shape {result.shape} does not match {shape}")
    if np.any(result < 0) or np.any(result > 1) or not np.all(np.isfinite(result)):
        raise ValueError("custom pupil mask must be finite and in [0, 1]")
    return result


def make_pupil(
    config: TelescopeConfig,
    shape: tuple[int, int],
    extent_m: float,
    *,
    supersampling: int = 1,
    backend: ArrayBackend | None = None,
    dtype: Any = np.float64,
) -> NDArray[np.float64]:
    """Sample a circular/annular pupil with optional spiders or custom mask."""
    resolved = backend or cpu_backend()
    if supersampling < 1:
        raise ValueError("supersampling must be >= 1")
    if config.custom_mask_path is not None:
        return cast(
            NDArray[np.float64],
            resolved.asarray(_load_mask(config.custom_mask_path, shape), dtype=dtype),
        )
    if supersampling == 1:
        xx, yy = _coordinates(shape, extent_m, backend=resolved)
        return _analytic_mask(config, xx, yy, backend=resolved, dtype=dtype)
    height, width = shape
    centre_x, centre_y = _coordinates(shape, extent_m, backend=resolved)
    offsets = (resolved.arange(supersampling, dtype=dtype) + 0.5) / supersampling - 0.5
    mask = resolved.zeros(shape, dtype=dtype)
    for offset_y in offsets:
        for offset_x in offsets:
            xx = centre_x + offset_x * extent_m / width
            yy = centre_y + offset_y * extent_m / height
            mask += _analytic_mask(config, xx, yy, backend=resolved, dtype=dtype)
    return cast(NDArray[np.float64], mask / (supersampling * supersampling))


def _analytic_mask(
    config: TelescopeConfig,
    xx: NDArray[np.float64],
    yy: NDArray[np.float64],
    *,
    backend: ArrayBackend | None = None,
    dtype: Any = np.float64,
) -> NDArray[np.float64]:
    resolved = backend or cpu_backend()
    rotation = math.radians(config.pupil_rotation_deg)
    feature_x: NDArray[np.float64] = xx * math.cos(rotation) + yy * math.sin(rotation)
    feature_y: NDArray[np.float64] = -xx * math.sin(rotation) + yy * math.cos(rotation)
    radius = resolved.hypot(feature_x, feature_y)
    pupil_radius = config.pupil_diameter_m / 2.0
    mask = resolved.astype(radius <= pupil_radius, dtype)
    inner_radius = pupil_radius * config.central_obscuration_ratio
    if inner_radius > 0:
        mask[radius < inner_radius] = 0.0
    if config.segments_across_pupil is not None and config.segment_gap_fraction > 0:
        pitch = config.pupil_diameter_m / config.segments_across_pupil
        gap = pitch * config.segment_gap_fraction / 2.0
        local_x = resolved.mod(feature_x + pupil_radius, pitch)
        local_y = resolved.mod(feature_y + pupil_radius, pitch)
        in_gap = (local_x < gap) | (local_x > pitch - gap)
        in_gap |= (local_y < gap) | (local_y > pitch - gap)
        mask[in_gap] = 0.0
    if config.spiders:
        angle = resolved.arctan2(feature_y, feature_x)
        for spider in config.spiders:
            target = math.radians(spider.angle_deg)
            difference = resolved.arctan2(
                resolved.sin(angle - target), resolved.cos(angle - target)
            )
            mask[resolved.abs(difference) <= spider.width_fraction * math.pi / 2.0] = 0.0
    return cast(NDArray[np.float64], mask)


__all__ = ["make_pupil"]
