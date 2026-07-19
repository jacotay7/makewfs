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
    else:
        raise ValueError("custom pupil masks currently support .npy or .npz")
    result = np.asarray(array, dtype=np.float64)
    if result.shape != shape:
        raise ValueError(f"custom pupil shape {result.shape} does not match {shape}")
    if np.any(result < 0) or np.any(result > 1) or not np.all(np.isfinite(result)):
        raise ValueError("custom pupil mask must be finite and in [0, 1]")
    return result


def make_pupil(
    config: TelescopeConfig, shape: tuple[int, int], extent_m: float
) -> NDArray[np.float64]:
    """Sample a circular/annular pupil with optional spiders or custom mask."""
    if config.custom_mask_path is not None:
        return _load_mask(config.custom_mask_path, shape)
    xx, yy = _coordinates(shape, extent_m)
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
