"""Small public facade for configured wavefront-sensor simulations."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .backend import ArrayBackend, cpu_backend, cupy_backend
from .config import WFSConfig, load_config
from .detector import DetectorAdapter
from .provenance import metadata as build_metadata
from .sensors.base import OpticalResult, SensorEngine
from .sensors.pyramid import PyramidEngine
from .sensors.shack_hartmann import ShackHartmannEngine
from .wavefront import iter_phase_samples


class WavefrontSensor:
    """Configured wavefront-sensor facade.

    Construct once and call :meth:`expose` for each closed-loop residual OPD.
    ``photon_rate`` exposes the deterministic optical result for workflows that
    use another detector or need an ideal reference image.
    """

    def __init__(self, config: WFSConfig, *, _backend: ArrayBackend | None = None) -> None:
        self.config = config
        self.backend = _backend or (
            cupy_backend() if config.numerics.device == "gpu" else cpu_backend()
        )
        self.engine: SensorEngine
        if config.sensor.kind == "shack_hartmann":
            self.engine = ShackHartmannEngine(config, backend=self.backend)
        elif config.sensor.kind == "pyramid":
            self.engine = PyramidEngine(config, backend=self.backend)
        else:
            raise NotImplementedError(f"unsupported sensor kind {config.sensor.kind!r}")
        self.detector = DetectorAdapter(
            config.detector,
            self.engine.output_shape,
            device="cpu" if self.backend.is_cpu else "gpu",
        )
        self._metadata_base = build_metadata(
            self.config,
            sensor_kind=self.config.sensor.kind,
            launched_rate=0.0,
            captured_rate=0.0,
            opd_rms_m=0.0,
            seed=None,
            source_states=self.engine.source_states,
            file_digests=self.engine.file_digests,
        )

    @classmethod
    def from_toml(cls, path: str | Path) -> WavefrontSensor:
        """Build a sensor from a validated TOML file."""
        return cls(load_config(path))

    def _render(self, wavefront: ArrayLike) -> OpticalResult:
        return self.engine.render(cast(NDArray[np.float64], wavefront))

    def _opd_rms(self, opd: Any) -> Any:
        """Reduce OPD RMS on-device before the batched metadata crossing."""
        return self.backend.sqrt(self.backend.mean(opd**2))

    def _frame_metadata(
        self,
        *,
        launched_rate: float,
        captured_rate: float,
        opd_rms_m: float,
        seed: int | None,
    ) -> dict[str, Any]:
        """Copy cached static provenance and fill the per-frame values."""
        result = dict(self._metadata_base)
        result.update(
            {
                "wfs_launched_photons_s": float(launched_rate),
                "wfs_captured_photons_s": float(captured_rate),
                "wfs_input_opd_rms_m": float(opd_rms_m),
                "wfs_seed": seed if seed is not None else "internal",
            }
        )
        return result

    def photon_rate(self, wavefront: ArrayLike) -> Any:
        """Return the ideal native-pixel photon rate before detector noise."""
        return self._render(wavefront).photon_rate

    def reference(self) -> Any:
        """Return the ideal image for a zero dynamic OPD."""
        dtype = np.dtype(self.config.numerics.dtype)
        return self.photon_rate(self.backend.zeros(self.config.input.shape, dtype=dtype))

    def valid_subapertures(self) -> Any:
        """Return the configured static Shack--Hartmann lenslet-valid mask.

        The mask is derived from pupil illumination and the configured minimum
        illuminated fraction, not from a detector frame.  It is therefore safe
        to freeze into a calibration artifact and use for active-slope vector
        ordering.  Pyramid sensors do not have lenslet subapertures.
        """
        if self.config.sensor.kind != "shack_hartmann":
            raise ValueError("valid_subapertures is defined only for Shack--Hartmann")
        return cast(ShackHartmannEngine, self.engine).lenslet_valid.copy()

    def expose(self, wavefront: ArrayLike, *, seed: int | None = None) -> Any:
        """Render one wavefront and expose it through the configured detector."""
        total_start = perf_counter()
        optical_start = total_start
        result = self._render(wavefront)
        captured_rate, opd_rms = self.backend.scalars(
            result.captured_rate_per_s, self._opd_rms(result.opd_m)
        )
        optical_elapsed = perf_counter() - optical_start
        frame_metadata = self._frame_metadata(
            launched_rate=result.launched_rate_per_s,
            captured_rate=captured_rate,
            opd_rms_m=opd_rms,
            seed=seed,
        )
        detector_start = perf_counter()
        frame = self.detector.expose(
            result.photon_rate,
            metadata=frame_metadata,
            seed=seed,
            spectral_photon_rate=(
                None if result.spectral_photon_rate is None else result.spectral_photon_rate
            ),
            spectral_wavelengths_m=result.spectral_wavelengths_m,
        )
        detector_elapsed = perf_counter() - detector_start
        frame.metadata["wfs_optical_render_s"] = optical_elapsed
        frame.metadata["wfs_detector_expose_s"] = detector_elapsed
        frame.metadata["wfs_total_expose_s"] = perf_counter() - total_start
        return frame

    def expose_many(
        self, phases: Iterable[ArrayLike], seeds: Iterable[int | None] | None = None
    ) -> Iterator[Any]:
        """Yield one detector frame per phase sample without stacking the stream."""
        seed_iter = iter(seeds) if seeds is not None else None
        for phase in phases:
            seed = next(seed_iter) if seed_iter is not None else None
            yield self.expose(phase, seed=seed)

    def expose_integrated(
        self, phase_samples: ArrayLike | Iterable[ArrayLike], *, seed: int | None = None
    ) -> Any:
        """Expose one detector frame after uniformly averaging temporal OPD samples."""
        total_start = perf_counter()
        optical_start = total_start
        rate_sum: Any | None = None
        spectral_rate_sum: Any | None = None
        spectral_wavelengths_m: tuple[float, ...] | None = None
        opd_sum: Any | None = None
        sample_count = 0
        launched = 0.0
        captured: Any = 0.0
        for sample in iter_phase_samples(
            phase_samples,
            self.config.input.shape,
            backend=self.backend,
        ):
            result = self._render(sample)
            if rate_sum is None:
                rate_sum = self.backend.zeros_like(result.photon_rate)
                opd_sum = self.backend.zeros_like(result.opd_m)
            rate_sum += result.photon_rate
            assert opd_sum is not None
            opd_sum += result.opd_m
            if result.spectral_photon_rate is not None:
                if spectral_rate_sum is None:
                    spectral_rate_sum = self.backend.zeros_like(result.spectral_photon_rate)
                spectral_rate_sum += result.spectral_photon_rate
                if spectral_wavelengths_m is None:
                    spectral_wavelengths_m = result.spectral_wavelengths_m
                elif spectral_wavelengths_m != result.spectral_wavelengths_m:
                    raise RuntimeError("spectral wavelength nodes changed within one exposure")
            launched = result.launched_rate_per_s
            captured += result.captured_rate_per_s
            sample_count += 1
        if rate_sum is None or opd_sum is None:
            raise ValueError("phase_samples must contain at least one sample")
        average_rate = rate_sum / sample_count
        average_spectral_rate = (
            None if spectral_rate_sum is None else spectral_rate_sum / sample_count
        )
        average_opd = opd_sum / sample_count
        captured_rate, opd_rms = self.backend.scalars(
            captured / sample_count, self._opd_rms(average_opd)
        )
        frame_metadata = self._frame_metadata(
            launched_rate=launched,
            captured_rate=captured_rate,
            opd_rms_m=opd_rms,
            seed=seed,
        )
        frame_metadata["wfs_temporal_samples"] = sample_count
        optical_elapsed = perf_counter() - optical_start
        detector_start = perf_counter()
        frame = self.detector.expose(
            average_rate,
            metadata=frame_metadata,
            seed=seed,
            spectral_photon_rate=average_spectral_rate,
            spectral_wavelengths_m=spectral_wavelengths_m,
        )
        frame.metadata["wfs_optical_render_s"] = optical_elapsed
        frame.metadata["wfs_detector_expose_s"] = perf_counter() - detector_start
        frame.metadata["wfs_total_expose_s"] = perf_counter() - total_start
        return frame


def simulate(
    wavefront: ArrayLike, config: WFSConfig | str | Path, *, seed: int | None = None
) -> Any:
    """One-shot convenience wrapper around :class:`WavefrontSensor`."""
    resolved = load_config(config) if isinstance(config, (str, Path)) else config
    return WavefrontSensor(resolved).expose(wavefront, seed=seed)


__all__ = ["WavefrontSensor", "simulate"]
