"""Metadata helpers for reproducible frame provenance."""

from __future__ import annotations

import hashlib
import importlib.metadata
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .config import WFSConfig
from .source import SourceState, iter_source_states


def _file_digest(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def referenced_file_digests(config: WFSConfig) -> dict[str, str]:
    """Hash configured external arrays/curves once at sensor construction."""
    references = {
        "input_static_opd": config.input.static_opd_path,
        "telescope_custom_mask": config.telescope.custom_mask_path,
        "source_sed": config.source.sed_path,
        "source_transmission": config.source.transmission_path,
    }
    return {
        f"wfs_{name}_sha256": _file_digest(path)
        for name, path in references.items()
        if path is not None
    }


def metadata(
    config: WFSConfig,
    *,
    sensor_kind: str,
    launched_rate: float,
    captured_rate: float,
    opd_m: NDArray[Any],
    seed: int | None,
    source_states: tuple[SourceState, ...] | None = None,
    file_digests: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build serializable metadata for an ideal or detector frame."""
    states = iter_source_states(config) if source_states is None else source_states
    result: dict[str, Any] = {
        "frame_type": "wfs",
        "wfs_sensor": sensor_kind,
        "wfs_config": config.digest,
        "wfs_wavelength_m": config.sensor.wavelength_m,
        "wfs_launched_photons_s": float(launched_rate),
        "wfs_captured_photons_s": float(captured_rate),
        "wfs_input_opd_rms_m": float(np.sqrt(np.mean(np.asarray(opd_m) ** 2))),
        "wfs_seed": seed if seed is not None else "internal",
        "wfs_source_kind": config.source.kind,
        "wfs_source_state_count": len(states),
        "wfs_source_wavelengths_m": sorted({state.wavelength_m for state in states}),
        "wfs_source_states": [
            {
                "wavelength_m": state.wavelength_m,
                "weight": state.weight,
                "angle_x_rad": state.angle_x_rad,
                "angle_y_rad": state.angle_y_rad,
                "range_m": state.range_m,
            }
            for state in states
        ],
    }
    result.update(referenced_file_digests(config) if file_digests is None else file_digests)
    for package in ("makewfs", "getframes"):
        try:
            result[f"{package}_version"] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[f"{package}_version"] = "uninstalled"
    if sensor_kind == "pyramid":
        result["wfs_pyramid_face_order"] = [
            "upper_left",
            "upper_right",
            "lower_left",
            "lower_right",
        ]
    return result


__all__ = ["metadata"]
