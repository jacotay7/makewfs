"""Thin adapter from ideal photon-rate maps to ``getframes``."""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .config import DetectorConfig


class DetectorAdapter:
    """Own a configured :class:`getframes.Camera` and nothing else."""

    def __init__(
        self, config: DetectorConfig, optical_shape: tuple[int, int], *, device: str = "cpu"
    ) -> None:
        try:
            import getframes as gf
        except ImportError as exc:  # pragma: no cover - dependency is required at install time
            raise ImportError("makewfs requires getframes for detector frames") from exc
        camera_kwargs: dict[str, Any] = {
            "precision": config.precision,
            "default_temperature_c": config.temperature_c,
        }
        if device == "gpu":
            if "device" not in inspect.signature(gf.Camera).parameters:
                raise RuntimeError(
                    "GPU detector execution requires a getframes version with "
                    "Camera(..., device='gpu') support"
                )
            camera_kwargs["device"] = "gpu"
        if config.preset is not None:
            camera = gf.Camera.from_preset(
                config.preset,
                **camera_kwargs,
            )
        else:
            camera = gf.Camera(
                gf.CameraConfig.from_dict(config.inline),
                **camera_kwargs,
            )
        if config.qe_curve_path is not None:
            camera = camera.with_config(qe_curve=gf.QE.from_file(config.qe_curve_path))
        if camera.resolution != optical_shape:
            camera = camera.with_config(resolution=list(optical_shape))
        self.camera = camera
        self.config = config
        self.optical_shape = optical_shape
        self.device = device

    def _detector_array(self, value: Any) -> Any:
        """Preserve device arrays; normalise CPU inputs to NumPy."""
        return np.asarray(value) if self.device == "cpu" else value

    def _expose_camera(
        self,
        photon_rate: NDArray[Any],
        *,
        seed: int | None,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "seed": seed,
            "include_truth": self.config.include_truth,
            "binning": self.config.binning,
            "binning_mode": self.config.binning_mode,
        }
        return self.camera.expose(
            self._detector_array(photon_rate),
            self.config.exposure_s,
            self.config.temperature_c,
            **kwargs,
        )

    def expose(
        self,
        photon_rate: NDArray[Any],
        *,
        metadata: dict[str, Any],
        seed: int | None,
        spectral_photon_rate: NDArray[Any] | None = None,
        spectral_wavelengths_m: tuple[float, ...] | None = None,
    ) -> Any:
        """Run the existing detector signal chain."""
        if (
            spectral_photon_rate is not None
            and spectral_wavelengths_m is not None
            and self.camera.config.qe_curve is not None
        ):
            cube = self._detector_array(spectral_photon_rate)
            wavelengths_nm = np.asarray(spectral_wavelengths_m) * 1e9
            frame = self.camera.expose_spectral(
                cube,
                wavelengths_nm,
                self.config.exposure_s,
                self.config.temperature_c,
                binning=self.config.binning,
                binning_mode=self.config.binning_mode,
                seed=seed,
                include_truth=self.config.include_truth,
            )
        else:
            frame = self._expose_camera(
                self._detector_array(photon_rate),
                seed=seed,
            )
        frame.metadata.update(metadata)
        frame.metadata["detector_binning"] = self.config.binning
        frame.metadata["detector_binning_mode"] = self.config.binning_mode
        frame.metadata["wfs_device"] = self.device
        return frame


__all__ = ["DetectorAdapter"]
