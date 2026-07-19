"""Thin adapter from ideal photon-rate maps to ``getframes``."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .config import DetectorConfig


class DetectorAdapter:
    """Own a configured :class:`getframes.Camera` and nothing else."""

    def __init__(self, config: DetectorConfig, optical_shape: tuple[int, int]) -> None:
        try:
            import getframes as gf
        except ImportError as exc:  # pragma: no cover - dependency is required at install time
            raise ImportError("makewfs requires getframes for detector frames") from exc
        if config.preset is not None:
            camera = gf.Camera.from_preset(
                config.preset,
                precision=config.precision,
                default_temperature_c=config.temperature_c,
            )
        else:
            camera = gf.Camera(
                gf.CameraConfig.from_dict(config.inline),
                precision=config.precision,
                default_temperature_c=config.temperature_c,
            )
        if camera.resolution != optical_shape:
            camera = camera.with_config(resolution=list(optical_shape))
        self.camera = camera
        self.config = config
        self.optical_shape = optical_shape

    def expose(
        self, photon_rate: NDArray[Any], *, metadata: dict[str, Any], seed: int | None
    ) -> Any:
        """Run the existing detector signal chain."""
        frame = self.camera.expose(
            np.asarray(photon_rate),
            self.config.exposure_s,
            self.config.temperature_c,
            binning=self.config.binning,
            binning_mode=self.config.binning_mode,
            seed=seed,
            include_truth=self.config.include_truth,
        )
        frame.metadata.update(metadata)
        frame.metadata["detector_binning"] = self.config.binning
        frame.metadata["detector_binning_mode"] = self.config.binning_mode
        return frame


__all__ = ["DetectorAdapter"]
