"""Analytic and custom entrance-pupil masks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

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
) -> NDArray[np.float64]:
    """Sample a circular/annular pupil with optional spiders or custom mask."""
    if supersampling < 1:
        raise ValueError("supersampling must be >= 1")
    if config.custom_mask_path is not None:
        return _load_mask(config.custom_mask_path, shape)
    if supersampling == 1:
        xx, yy = _coordinates(shape, extent_m)
        return _analytic_mask(config, xx, yy)
    height, width = shape
    centre_x, centre_y = _coordinates(shape, extent_m)
    offsets = (np.arange(supersampling, dtype=np.float64) + 0.5) / supersampling - 0.5
    mask = np.zeros(shape, dtype=np.float64)
    for offset_y in offsets:
        for offset_x in offsets:
            xx = centre_x + offset_x * extent_m / width
            yy = centre_y + offset_y * extent_m / height
            mask += _analytic_mask(config, xx, yy)
    return mask / (supersampling * supersampling)


def _analytic_mask(
    config: TelescopeConfig,
    xx: NDArray[np.float64],
    yy: NDArray[np.float64],
) -> NDArray[np.float64]:
    radius = np.hypot(xx, yy)
    pupil_radius = config.pupil_diameter_m / 2.0
    mask = (radius <= pupil_radius).astype(np.float64)
    inner_radius = pupil_radius * config.central_obscuration_ratio
    if inner_radius > 0:
        mask[radius < inner_radius] = 0.0
    if config.spiders:
        angle = np.arctan2(yy, xx) - np.deg2rad(config.pupil_rotation_deg)
        for spider in config.spiders:
            target = np.deg2rad(spider.angle_deg)
            difference = np.arctan2(np.sin(angle - target), np.cos(angle - target))
            mask[np.abs(difference) <= spider.width_fraction * np.pi / 2.0] = 0.0
    return mask


__all__ = ["make_pupil"]
