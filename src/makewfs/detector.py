"""Thin adapter from ideal photon-rate maps to ``getframes``."""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .config import DetectorConfig


def resolve_camera_config(config: DetectorConfig) -> Any:
    """Resolve the ``getframes`` camera configuration this detector declares.

    Optics that must know a sensor property before a frame exists -- such as the
    lateral charge diffusion that acts on the focal-plane irradiance ahead of
    pixel integration -- read it from here, so the value keeps its single owner
    in ``getframes`` instead of being restated in WFS configuration.
    """
    try:
        import getframes as gf
    except ImportError as exc:  # pragma: no cover - dependency is required at install time
        raise ImportError("makewfs requires getframes for detector frames") from exc
    if config.preset is not None:
        return gf.load_preset(config.preset)
    return gf.CameraConfig.from_dict(config.inline)


def charge_diffusion_fwhm_px(config: DetectorConfig) -> float:
    """Return the sensor's lateral charge-diffusion FWHM in native pixels."""
    return float(getattr(resolve_camera_config(config), "charge_diffusion_fwhm_px", 0.0))


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
        camera = gf.Camera(resolve_camera_config(config), **camera_kwargs)
        if config.qe_curve_path is not None:
            camera = camera.with_config(qe_curve=gf.QE.from_file(config.qe_curve_path))
        if config.roi is not None:
            if "roi" not in gf.CameraConfig.__dataclass_fields__:
                raise RuntimeError(
                    "detector ROI execution requires a getframes version with "
                    "CameraConfig.roi support"
                )
            camera = camera.with_config(roi=config.roi.getframes_tuple)
            if camera.resolution != optical_shape:
                raise ValueError(
                    f"detector.roi produces shape {camera.resolution}, but the configured "
                    f"optics produce {optical_shape}"
                )
        elif camera.resolution != optical_shape:
            camera = camera.with_config(resolution=list(optical_shape))
        self.camera = camera
        self.config = config
        self.optical_shape = optical_shape
        self.device = device
        workspace_type = getattr(gf, "DetectorWorkspace", None)
        self._workspace = None if workspace_type is None else workspace_type()

    def _detector_array(self, value: Any) -> Any:
        """Preserve device arrays; normalise CPU inputs to NumPy."""
        return np.asarray(value) if self.device == "cpu" else value

    def _expose_camera(
        self,
        photon_rate: NDArray[Any],
        *,
        seed: int | None,
        out: Any | None,
    ) -> Any:
        if self.config.readout_mode == "cds":
            self._reject_caller_owned_cds_output(out)
            return self._correlated_double_sample(photon_rate, seed=seed)
        kwargs: dict[str, Any] = {
            "seed": seed,
            "include_truth": self.config.include_truth,
            "binning": self.config.binning,
            "binning_mode": self.config.binning_mode,
            "background": self.config.background_photon_rate_per_s,
        }
        if out is not None:
            if self._workspace is None:
                raise RuntimeError(
                    "caller-owned detector output requires a getframes version "
                    "with DetectorWorkspace support"
                )
            kwargs["workspace"] = self._workspace
            kwargs["out"] = out
        return self.camera.expose(
            self._detector_array(photon_rate),
            self.config.exposure_s,
            self.config.temperature_c,
            **kwargs,
        )

    @staticmethod
    def _reject_caller_owned_cds_output(out: Any | None) -> None:
        """Refuse a caller-owned buffer rather than silently ignoring it.

        Correlated double sampling allocates the signed difference of two reads,
        so there is no single unsigned buffer for the caller to own. Say so
        instead of writing the frame somewhere the caller is not looking.
        """
        if out is not None:
            raise RuntimeError(
                "detector.readout_mode = 'cds' does not support caller-owned output "
                "storage; the returned frame is a freshly allocated signed difference"
            )

    def _require_cds_support(self, method: str) -> Any:
        """Return the camera's CDS entry point, or explain which version is needed."""
        entry_point = getattr(self.camera, method, None)
        if entry_point is None:
            raise RuntimeError(
                f"detector.readout_mode = 'cds' requires a getframes version with "
                f"Camera.{method} support"
            )
        return entry_point

    def _correlated_double_sample(self, photon_rate: NDArray[Any], *, seed: int | None) -> Any:
        """Read one two-read global-reset ramp and return its difference."""
        return self._require_cds_support("correlated_double_sample")(
            self._detector_array(photon_rate),
            self.config.exposure_s,
            self.config.temperature_c,
            pedestal_interval_s=self.config.cds_pedestal_interval_s,
            background=self.config.background_photon_rate_per_s,
            seed=seed,
            include_truth=self.config.include_truth,
        )

    def expose(
        self,
        photon_rate: NDArray[Any],
        *,
        metadata: dict[str, Any],
        seed: int | None,
        spectral_photon_rate: NDArray[Any] | None = None,
        spectral_wavelengths_m: tuple[float, ...] | None = None,
        out: Any | None = None,
    ) -> Any:
        """Run the existing detector signal chain into optional caller-owned storage."""
        if (
            spectral_photon_rate is not None
            and spectral_wavelengths_m is not None
            and self.camera.config.qe_curve is not None
        ):
            cube = self._detector_array(spectral_photon_rate)
            wavelengths_nm = np.asarray(spectral_wavelengths_m) * 1e9
            if self.config.readout_mode == "cds":
                self._reject_caller_owned_cds_output(out)
                frame = self._require_cds_support("correlated_double_sample_spectral")(
                    cube,
                    wavelengths_nm,
                    self.config.exposure_s,
                    self.config.temperature_c,
                    pedestal_interval_s=self.config.cds_pedestal_interval_s,
                    background=self.config.background_photon_rate_per_s,
                    seed=seed,
                    include_truth=self.config.include_truth,
                )
            else:
                spectral_kwargs: dict[str, Any] = {}
                if out is not None:
                    if self._workspace is None:
                        raise RuntimeError(
                            "caller-owned detector output requires a getframes version "
                            "with DetectorWorkspace support"
                        )
                    if "workspace" not in inspect.signature(self.camera.expose_spectral).parameters:
                        raise RuntimeError(
                            "caller-owned spectral detector output requires a getframes "
                            "version with expose_spectral(workspace=..., out=...) support"
                        )
                    spectral_kwargs = {"workspace": self._workspace, "out": out}
                frame = self.camera.expose_spectral(
                    cube,
                    wavelengths_nm,
                    self.config.exposure_s,
                    self.config.temperature_c,
                    binning=self.config.binning,
                    binning_mode=self.config.binning_mode,
                    background=self.config.background_photon_rate_per_s,
                    seed=seed,
                    include_truth=self.config.include_truth,
                    **spectral_kwargs,
                )
        else:
            frame = self._expose_camera(
                self._detector_array(photon_rate),
                seed=seed,
                out=out,
            )
        frame.metadata.update(metadata)
        frame.metadata["detector_binning"] = self.config.binning
        frame.metadata["detector_binning_mode"] = self.config.binning_mode
        frame.metadata["detector_readout_mode"] = self.config.readout_mode
        frame.metadata["wfs_device"] = self.device
        return frame


__all__ = ["DetectorAdapter"]
