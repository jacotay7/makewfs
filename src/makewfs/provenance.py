"""Metadata helpers for reproducible frame provenance."""

from __future__ import annotations

import importlib.metadata
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .config import WFSConfig


def metadata(
    config: WFSConfig,
    *,
    sensor_kind: str,
    launched_rate: float,
    captured_rate: float,
    opd_m: NDArray[Any],
    seed: int | None,
) -> dict[str, Any]:
    """Build serializable metadata for an ideal or detector frame."""
    result: dict[str, Any] = {
        "frame_type": "wfs",
        "wfs_sensor": sensor_kind,
        "wfs_config": config.digest,
        "wfs_wavelength_m": config.sensor.wavelength_m,
        "wfs_launched_photons_s": float(launched_rate),
        "wfs_captured_photons_s": float(captured_rate),
        "wfs_input_opd_rms_m": float(np.sqrt(np.mean(np.asarray(opd_m) ** 2))),
        "wfs_seed": seed if seed is not None else "internal",
    }
    for package in ("makewfs", "getframes"):
        try:
            result[f"{package}_version"] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[f"{package}_version"] = "uninstalled"
    return result


__all__ = ["metadata"]
