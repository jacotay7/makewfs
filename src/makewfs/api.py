"""Small public facade for configured wavefront-sensor simulations."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import numpy as np
from aocore import RAD_TO_ARCSEC
from numpy.typing import ArrayLike, NDArray

from .backend import ArrayBackend, cpu_backend, cupy_backend
from .config import WFSConfig, load_config
from .detector import DetectorAdapter
from .provenance import metadata as build_metadata
from .pupil import make_pupil
from .sampling import area_rebin
from .sensors.base import OpticalResult, SensorEngine
from .sensors.pyramid import PyramidEngine
from .sensors.shack_hartmann import ShackHartmannEngine
from .wavefront import (
    amplitude_samples,
    grid_rms,
    iter_phase_samples,
    pupil_rms,
    pupil_weights,
)


@dataclass(frozen=True)
class SubapertureFlux:
    """Ideal (noise-free) per-subaperture photon rates of one Shack--Hartmann frame.

    Every array is indexed ``(lenslet_y, lenslet_x)`` like
    :meth:`WavefrontSensor.valid_subapertures` and stays on the sensor's
    device (a CuPy array on a GPU sensor).

    Attributes
    ----------
    detector_photons_s
        Photon rate collected by each subaperture's own
        ``pixels_per_subaperture x pixels_per_subaperture`` pixel block, in
        photons/s: the block sum of :attr:`photon_rate` without the detector
        margin. It excludes light its spot sends beyond the block and, with
        ``crosstalk_reach_subapertures > 0``, includes light neighbouring spots
        send into it. This is the flux a detector reading those pixels
        measures, before QE and noise.
    pupil_photons_s
        Photon rate entering each lenslet's clear aperture in the pupil plane,
        in photons/s: the source rate times the integral of the entrance-field
        intensity (pupil times amplitude squared, after the spatial filter and
        the lenslet fill factor) over the lenslet, divided by the unscintillated
        pupil's integral. It does not depend on the detector window.
    valid
        Static valid-subaperture map of the offset state used.
    photon_rate
        The ideal photon-rate image the detector fluxes were summed from, in
        photons/s per native detector pixel.
    """

    detector_photons_s: Any
    pupil_photons_s: Any
    valid: Any
    photon_rate: Any


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
            opd_rms_unweighted_m=0.0,
            seed=None,
            source_states=self.engine.source_states,
            file_digests=self.engine.file_digests,
            pyramid_fft_size=getattr(self.engine, "nfft", None),
        )
        if isinstance(self.engine, ShackHartmannEngine):
            self._metadata_base["wfs_lenslet_offset_state_count"] = len(self.engine.offset_states)
            if config.source.lgs_model == "kernel":
                # Native pixels: the kernel half-width, and how far beyond the
                # window light is propagated so it can be convolved inward.
                oversampling = config.numerics.fft_oversampling
                self._metadata_base["wfs_lgs_kernel_radius_px"] = (
                    self.engine.lgs_kernel_radius_samples / oversampling
                )
                self._metadata_base["wfs_lgs_kernel_margin_px"] = (
                    self.engine.lgs_kernel_margin_samples / oversampling
                )
        self._rms_weights = pupil_weights(self._input_pupil_intensity(), backend=self.backend)

    @classmethod
    def from_toml(cls, path: str | Path) -> WavefrontSensor:
        """Build a sensor from a validated TOML file."""
        return cls(load_config(path))

    def _render(
        self,
        wavefront: ArrayLike,
        offset_state: int = 0,
        amplitude: ArrayLike | None = None,
        *,
        lenslet_flux: bool = False,
    ) -> OpticalResult:
        opd = cast(NDArray[np.float64], wavefront)
        if isinstance(self.engine, ShackHartmannEngine):
            return self.engine.render(
                opd, offset_state=offset_state, amplitude=amplitude, lenslet_flux=lenslet_flux
            )
        self._check_offset_state(offset_state)
        if amplitude is None:
            return self.engine.render(opd)
        return cast(PyramidEngine, self.engine).render(opd, amplitude=amplitude)

    @property
    def wavelengths_m(self) -> tuple[float, ...]:
        """Wavelength nodes of the source quadrature, in metres.

        The order is that of the spectral photon-rate cube and of a
        per-wavelength ``amplitude`` stack: an amplitude of shape
        ``(len(wavelengths_m), *input.shape)`` gives map ``k`` to
        ``wavelengths_m[k]``.
        """
        return tuple(self.engine._wavelengths)

    def _check_offset_state(self, offset_state: int) -> None:
        """Validate an offset-state index; only the SH engine has more than one."""
        if isinstance(self.engine, ShackHartmannEngine):
            self.engine.geometry(offset_state)
        elif isinstance(offset_state, bool) or offset_state != 0:
            raise ValueError("offset_state is defined only for Shack--Hartmann sensors; use 0")

    def lenslet_grid_offsets_fraction(self) -> tuple[tuple[float, float], ...]:
        """Return the effective ``[x, y]`` lenslet-grid offset of each offset state.

        Shack--Hartmann only. Offsets are in lenslet pitches: each entry of
        ``shack_hartmann.lenslet_grid_offset_states_fraction`` added to the
        static ``lenslet_grid_offset_fraction``, or the static offset alone
        when no states are declared. The position in this tuple is the
        ``offset_state`` index that :meth:`photon_rate`, :meth:`expose` and the
        other render methods accept.
        """
        if not isinstance(self.engine, ShackHartmannEngine):
            raise ValueError("lenslet-grid offsets are defined only for Shack--Hartmann")
        return self.engine.offset_states

    def _input_pupil_intensity(self) -> Any:
        """Return the pupil intensity the optics use, on the input grid.

        An analytic pupil is evaluated on ``input.shape`` from the same
        telescope model and ``numerics.pupil_supersampling`` the engine uses on
        its own grid, as :meth:`pupil_illumination` does; both grids span
        ``input.grid_extent_m``. A custom mask exists only on the engine's
        configured pupil grid, so its intensity is area-averaged from there
        onto the input grid, which is exact when the two shapes match. The
        engines use the pupil as a field amplitude, so the intensity is its
        square.
        """
        if self.config.telescope.custom_mask_path is None:
            amplitude = make_pupil(
                self.config.telescope,
                self.config.input.shape,
                self.config.input.grid_extent_m,
                supersampling=self.config.numerics.pupil_supersampling,
                backend=self.backend,
                dtype=np.float64,
            )
            return amplitude * amplitude
        amplitude = self.backend.asarray(self.engine.configured_pupil, dtype=np.float64)
        return area_rebin(amplitude * amplitude, self.config.input.shape, backend=self.backend)

    def _opd_rms(self, opd: Any) -> tuple[Any, Any]:
        """Reduce both input-OPD RMS values on the device.

        Returns the pupil-weighted, piston-removed RMS and the unweighted
        whole-grid RMS as device scalars, for the one batched metadata
        crossing in ``backend.scalars``.
        """
        return (
            pupil_rms(opd, self._rms_weights, backend=self.backend),
            grid_rms(opd, backend=self.backend),
        )

    def _frame_metadata(
        self,
        *,
        launched_rate: float,
        captured_rate: float,
        opd_rms_m: float,
        opd_rms_unweighted_m: float,
        seed: int | None,
        offset_state: int = 0,
        amplitude: bool = False,
    ) -> dict[str, Any]:
        """Copy cached static provenance and fill the per-frame values."""
        result = dict(self._metadata_base)
        result["wfs_input_amplitude"] = bool(amplitude)
        if isinstance(self.engine, ShackHartmannEngine):
            result["wfs_lenslet_offset_state"] = int(offset_state)
            result["wfs_lenslet_grid_offset_fraction"] = list(
                self.engine.offset_states[int(offset_state)]
            )
        result.update(
            {
                "wfs_launched_photons_s": float(launched_rate),
                "wfs_captured_photons_s": float(captured_rate),
                "wfs_input_opd_rms_m": float(opd_rms_m),
                "wfs_input_opd_rms_unweighted_m": float(opd_rms_unweighted_m),
                "wfs_seed": seed if seed is not None else "internal",
            }
        )
        return result

    def photon_rate(
        self,
        wavefront: ArrayLike,
        *,
        amplitude: ArrayLike | None = None,
        offset_state: int = 0,
    ) -> Any:
        """Return the ideal native-pixel photon rate before detector noise.

        ``offset_state`` selects one of the configured Shack--Hartmann
        lenslet-grid offset states (see :meth:`lenslet_grid_offsets_fraction`);
        it must be ``0`` for a sensor with one state and for a pyramid.

        ``amplitude`` is an optional per-frame pupil-plane field amplitude
        ``|U| / |U_0|`` (dimensionless, non-negative), relative to the
        unscintillated field the source normalization describes. It multiplies
        the configured pupil amplitude, so the pupil intensity, and with it the
        photon rate, scales with its square; the wavefront still carries the
        phase. Pass an ``input.shape`` map for every wavelength, or a
        ``(len(wavelengths_m), *input.shape)`` stack with one map per
        wavelength node. ``None`` (the default) and an amplitude of exactly one
        give the OPD-only result. See ``docs/scintillation.md``.
        """
        return self._render(wavefront, offset_state, amplitude).photon_rate

    def subaperture_flux(
        self,
        wavefront: ArrayLike,
        *,
        amplitude: ArrayLike | None = None,
        offset_state: int = 0,
    ) -> SubapertureFlux:
        """Return the ideal per-subaperture photon rates of one wavefront.

        Shack--Hartmann only. One render gives the ideal image and, for each
        lenslet, the photon rate its pixel block collects and the photon rate
        entering its aperture in the pupil plane (see :class:`SubapertureFlux`).
        Under scintillation (``amplitude``, as in :meth:`photon_rate`) these
        fluxes measure the pupil intensity. They are noise-free; a measured
        flux comes from summing the same pixel blocks of an exposed frame.
        """
        if not isinstance(self.engine, ShackHartmannEngine):
            raise ValueError("subaperture_flux is defined only for Shack--Hartmann")
        engine = self.engine
        result = engine.render(
            cast(NDArray[np.float64], wavefront),
            offset_state=offset_state,
            amplitude=amplitude,
            lenslet_flux=True,
        )
        n = engine.n_lenslets
        pixels = engine.settings.pixels_per_subaperture
        margin = engine.settings.detector_margin_pixels
        base = result.photon_rate[margin : margin + n * pixels, margin : margin + n * pixels]
        detector = self.backend.sum(base.reshape(n, pixels, n, pixels), axis=(1, 3))
        return SubapertureFlux(
            detector_photons_s=detector,
            pupil_photons_s=result.lenslet_rate_per_s,
            valid=engine.geometry(offset_state).lenslet_valid.copy(),
            photon_rate=result.photon_rate,
        )

    def reference(self, *, offset_state: int = 0) -> Any:
        """Return the ideal image for a zero dynamic OPD in one offset state."""
        dtype = np.dtype(self.config.numerics.dtype)
        return self.photon_rate(
            self.backend.zeros(self.config.input.shape, dtype=dtype), offset_state=offset_state
        )

    def valid_subapertures(self, *, offset_state: int = 0) -> Any:
        """Return the configured static Shack--Hartmann lenslet-valid mask.

        The mask is derived from pupil illumination and the configured minimum
        illuminated fraction, not from a detector frame.  It is therefore safe
        to freeze into a calibration artifact and use for active-slope vector
        ordering. Each lenslet-grid offset state has its own map, selected by
        ``offset_state``. Pyramid sensors do not have lenslet subapertures.
        """
        if not isinstance(self.engine, ShackHartmannEngine):
            raise ValueError("valid_subapertures is defined only for Shack--Hartmann")
        return self.engine.geometry(offset_state).lenslet_valid.copy()

    def subaperture_plate_scale_arcsec(self) -> float:
        """Return the on-sky angle one detector pixel subtends per subaperture.

        Shack--Hartmann only. Derived from the configured spot sampling and the
        subaperture's own size on sky, so it stays consistent with the geometry
        that actually forms the spots rather than restating it.
        """
        if self.config.sensor.kind != "shack_hartmann":
            raise ValueError("plate scale is defined only for Shack--Hartmann")
        engine = cast(ShackHartmannEngine, self.engine)
        sampling = engine._spot_sampling(self.config.sensor.wavelength_m)
        subaperture_m = self.config.input.grid_extent_m / engine.n_lenslets
        lambda_over_d_rad = self.config.sensor.wavelength_m / subaperture_m
        return float(lambda_over_d_rad / sampling * RAD_TO_ARCSEC)

    def subaperture_field_of_view_arcsec(self) -> float:
        """Return the on-sky field of view one subaperture's window spans.

        By default a Shack--Hartmann's detector window is itself a field stop.
        Each subaperture's spot is formed and integrated only over its own
        ``pixels_per_subaperture`` block, and the blocks are tiled without
        overlap, so light beyond this field is neither recorded nor allowed to
        contaminate a neighbour. An instrument whose physical field stop matches
        its pixel field is therefore already represented. One without such a
        stop spills light between subapertures: set
        ``shack_hartmann.crosstalk_reach_subapertures`` to model that. Reporting
        the value makes the assumption checkable instead of leaving it implied
        by the pixel count.
        """
        settings = self.config.shack_hartmann
        if settings is None:
            raise ValueError("field of view is defined only for Shack--Hartmann")
        return self.subaperture_plate_scale_arcsec() * settings.pixels_per_subaperture

    def pupil_illumination(self, shape: tuple[int, int] | None = None) -> Any:
        """Return the configured pupil illumination on a requested grid.

        Consumers that own actuator or wavefront models need the illumination on
        the same grid their own arrays use, and pupil formation belongs here, so
        this evaluates the configured telescope pupil on ``shape`` (default
        ``config.input.shape``) rather than the sensor's internal propagation
        grid. A configured ``custom_mask_path`` is not resampled: it must already
        match the requested shape, so supply the mask at that sampling instead of
        letting a mask be silently interpolated.
        """
        resolved = self.config.input.shape if shape is None else shape
        return make_pupil(
            self.config.telescope,
            resolved,
            self.config.input.grid_extent_m,
            supersampling=self.config.numerics.pupil_supersampling,
            backend=self.backend,
            dtype=np.dtype(self.config.numerics.dtype),
        )

    @property
    def charge_diffusion_fwhm_px(self) -> float:
        """Return the resolved sensor's lateral charge-diffusion FWHM in pixels.

        Charge diffusion is detector physics owned by ``getframes``; this only
        reports the resolved value, so a consumer recording provenance does not
        have to restate a detector measurement of its own.
        """
        return float(getattr(self.detector.camera.config, "charge_diffusion_fwhm_px", 0.0))

    def expose(
        self,
        wavefront: ArrayLike,
        *,
        amplitude: ArrayLike | None = None,
        seed: int | None = None,
        out: Any | None = None,
        offset_state: int = 0,
    ) -> Any:
        """Render one wavefront into optional caller-owned detector storage.

        Besides provenance and timings, the frame metadata records the input
        wavefront's ``wfs_input_opd_rms_m``, its RMS in OPD metres weighted by
        the pupil intensity with piston removed (aocore CONVENTIONS 4.1), and
        ``wfs_input_opd_rms_unweighted_m``, its RMS over the whole input grid
        with piston kept. Both use the configured pupil, not the per-frame
        amplitude.

        ``amplitude`` is the optional per-frame pupil amplitude of
        :meth:`photon_rate`; frames record whether one was given as
        ``wfs_input_amplitude``.

        ``offset_state`` selects a configured Shack--Hartmann lenslet-grid
        offset state (``shack_hartmann.lenslet_grid_offset_states_fraction``);
        a Shack--Hartmann frame records it as ``wfs_lenslet_offset_state`` and
        the effective offset as ``wfs_lenslet_grid_offset_fraction``.
        """
        total_start = perf_counter()
        optical_start = total_start
        result = self._render(wavefront, offset_state, amplitude)
        captured_rate, opd_rms, opd_rms_unweighted = self.backend.scalars(
            result.captured_rate_per_s, *self._opd_rms(result.opd_m)
        )
        optical_elapsed = perf_counter() - optical_start
        frame_metadata = self._frame_metadata(
            launched_rate=result.launched_rate_per_s,
            captured_rate=captured_rate,
            opd_rms_m=opd_rms,
            opd_rms_unweighted_m=opd_rms_unweighted,
            seed=seed,
            offset_state=offset_state,
            amplitude=amplitude is not None,
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
            out=out,
        )
        detector_elapsed = perf_counter() - detector_start
        frame.metadata["wfs_optical_render_s"] = optical_elapsed
        frame.metadata["wfs_detector_expose_s"] = detector_elapsed
        frame.metadata["wfs_total_expose_s"] = perf_counter() - total_start
        return frame

    def expose_many(
        self,
        phases: Iterable[ArrayLike],
        seeds: Iterable[int | None] | None = None,
        *,
        offset_states: Iterable[int] | None = None,
        amplitudes: Iterable[ArrayLike | None] | None = None,
    ) -> Iterator[Any]:
        """Yield one detector frame per phase sample without stacking the stream.

        ``offset_states`` optionally gives the lenslet-grid offset state of each
        frame, for example ``itertools.cycle(range(n))`` for a dithered
        Shack--Hartmann; the default renders every frame in state ``0``.
        ``amplitudes`` optionally gives each frame's pupil amplitude (see
        :meth:`photon_rate`), consumed in step with ``phases``.
        """
        seed_iter = iter(seeds) if seeds is not None else None
        state_iter = iter(offset_states) if offset_states is not None else None
        amplitude_iter = iter(amplitudes) if amplitudes is not None else None
        for phase in phases:
            seed = next(seed_iter) if seed_iter is not None else None
            offset_state = next(state_iter) if state_iter is not None else 0
            amplitude = next(amplitude_iter) if amplitude_iter is not None else None
            yield self.expose(phase, amplitude=amplitude, seed=seed, offset_state=offset_state)

    def expose_integrated(
        self,
        phase_samples: ArrayLike | Iterable[ArrayLike],
        *,
        amplitude: ArrayLike | Iterable[ArrayLike] | None = None,
        seed: int | None = None,
        out: Any | None = None,
        offset_state: int = 0,
    ) -> Any:
        """Expose temporally averaged OPD into optional caller-owned detector storage.

        Every sample uses the same lenslet-grid ``offset_state`` (see
        :meth:`expose`). ``amplitude`` optionally gives one pupil amplitude per
        temporal sample: a sequence, or an array whose leading axis is the
        sample axis, each entry an ``input.shape`` map or a per-wavelength stack
        (see :meth:`photon_rate`).
        """
        total_start = perf_counter()
        optical_start = total_start
        self._check_offset_state(offset_state)
        batched = getattr(self.engine, "render_integrated", None)
        if callable(batched):
            # An engine that can render the whole exposure at once does so far
            # more cheaply than one call per sample, because the per-call
            # dispatch cost dominates the arithmetic. The sequential path below
            # remains the reference and is still used by engines without it.
            samples = list(
                iter_phase_samples(phase_samples, self.config.input.shape, backend=self.backend)
            )
            if not samples:
                raise ValueError("phase_samples must contain at least one sample")
            result = (
                batched(samples, offset_state=offset_state)
                if amplitude is None
                else batched(
                    samples,
                    offset_state=offset_state,
                    amplitudes=amplitude_samples(amplitude, len(samples)),
                )
            )
            captured_rate, opd_rms, opd_rms_unweighted = self.backend.scalars(
                result.captured_rate_per_s, *self._opd_rms(result.opd_m)
            )
            frame_metadata = self._frame_metadata(
                launched_rate=result.launched_rate_per_s,
                captured_rate=captured_rate,
                opd_rms_m=opd_rms,
                opd_rms_unweighted_m=opd_rms_unweighted,
                seed=seed,
                offset_state=offset_state,
                amplitude=amplitude is not None,
            )
            frame_metadata["wfs_temporal_samples"] = len(samples)
            optical_elapsed = perf_counter() - optical_start
            detector_start = perf_counter()
            frame = self.detector.expose(
                result.photon_rate,
                metadata=frame_metadata,
                seed=seed,
                spectral_photon_rate=result.spectral_photon_rate,
                spectral_wavelengths_m=result.spectral_wavelengths_m,
                out=out,
            )
            frame.metadata["wfs_optical_render_s"] = optical_elapsed
            frame.metadata["wfs_detector_expose_s"] = perf_counter() - detector_start
            frame.metadata["wfs_total_expose_s"] = perf_counter() - total_start
            return frame
        rate_sum: Any | None = None
        spectral_rate_sum: Any | None = None
        spectral_wavelengths_m: tuple[float, ...] | None = None
        opd_sum: Any | None = None
        sample_count = 0
        launched = 0.0
        captured: Any = 0.0
        samples = list(
            iter_phase_samples(
                phase_samples,
                self.config.input.shape,
                backend=self.backend,
            )
        )
        for sample, sample_amplitude in zip(samples, amplitude_samples(amplitude, len(samples))):
            result = self._render(sample, offset_state, sample_amplitude)
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
        captured_rate, opd_rms, opd_rms_unweighted = self.backend.scalars(
            captured / sample_count, *self._opd_rms(average_opd)
        )
        frame_metadata = self._frame_metadata(
            launched_rate=launched,
            captured_rate=captured_rate,
            opd_rms_m=opd_rms,
            opd_rms_unweighted_m=opd_rms_unweighted,
            seed=seed,
            offset_state=offset_state,
            amplitude=amplitude is not None,
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
            out=out,
        )
        frame.metadata["wfs_optical_render_s"] = optical_elapsed
        frame.metadata["wfs_detector_expose_s"] = perf_counter() - detector_start
        frame.metadata["wfs_total_expose_s"] = perf_counter() - total_start
        return frame


def simulate(
    wavefront: ArrayLike,
    config: WFSConfig | str | Path,
    *,
    seed: int | None = None,
    amplitude: ArrayLike | None = None,
) -> Any:
    """One-shot convenience wrapper around :class:`WavefrontSensor`.

    ``amplitude`` is the optional per-frame pupil amplitude of
    :meth:`WavefrontSensor.photon_rate`.
    """
    resolved = load_config(config) if isinstance(config, (str, Path)) else config
    return WavefrontSensor(resolved).expose(wavefront, amplitude=amplitude, seed=seed)


__all__ = ["SubapertureFlux", "WavefrontSensor", "simulate"]
