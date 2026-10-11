"""Batched Fourier-optics Shack-Hartmann wavefront sensor."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from aocore import opd_to_phase
from numpy.typing import NDArray

from ..backend import ArrayBackend, complex_dtype, real_dtype
from ..config import WFSConfig
from ..detector import charge_diffusion_fwhm_px
from ..provenance import referenced_file_digests
from ..pupil import make_pupil
from ..radiometry import clear_aperture_fraction, source_rate_per_s
from ..sampling import (
    _SpotPropagationPlan,
    lenslet_field_upsampling,
    load_blur_kernel,
    spot_intensity,
    spot_sampling_geometry,
)
from ..sensors.base import OpticalResult, SensorEngine
from ..source import SourceState, iter_source_states, lgs_kernel_morphology
from ..wavefront import WavefrontInput, _coordinates, load_static_opd
from ._shack_hartmann_cuda import (
    _compiled_executor_rejection,
    _CompiledExecutorUnavailable,
    _CompiledShackHartmannExecutor,
)
from .lgs import _KernelOffsets, _LGSKernelPlan, kernel_offsets, kernel_radius_samples


@dataclass
class _LensletGeometry:
    """Cached lenslet-grid geometry of one configured offset state.

    Everything here depends on where the lenslet grid sits on the pupil, so a
    sensor with several ``lenslet_grid_offset_states_fraction`` holds one per
    state and selects it per call. Arrays are on the engine's backend.
    """

    offset_fraction: tuple[float, float]
    sample_indices: tuple[Any, Any] | None
    """Physical-grid sample coordinates of each lenslet-grid sample, or ``None``
    for the aligned grid, which needs no resampling."""
    lenslet_mask: Any
    lenslet_illumination: Any
    lenslet_valid: Any
    centers_x: Any
    """Flattened ``(lenslet_y, lenslet_x)`` subaperture centres, pupil metres."""
    centers_y: Any
    total_field_flux: Any
    piston_index: tuple[int, int]
    field_angle_opd: tuple[Any, ...]
    lgs_kernel: _LGSKernelPlan | None = None
    compiled_executors: dict[int, _CompiledShackHartmannExecutor] = field(default_factory=dict)


class ShackHartmannEngine(SensorEngine):
    """Render square-lenslet Shack-Hartmann spot mosaics.

    The implementation uses one batched FFT per wavelength/source state.  The
    internal pupil is resampled once at construction so arbitrary input array
    sizes can be used without dropping edge pixels to force a reshape.
    """

    kind = "shack_hartmann"

    def __init__(self, config: WFSConfig, *, backend: ArrayBackend | None = None) -> None:
        if config.shack_hartmann is None:
            raise ValueError("Shack-Hartmann configuration is missing")
        self.config = config
        self.backend = self.resolve_backend(backend)
        self._real_dtype = real_dtype(config.numerics.dtype)
        self._rate_dtype = real_dtype("float64")
        self.settings = config.shack_hartmann
        self.n_lenslets = self.settings.lenslets_across_pupil
        requested = config.numerics.pupil_samples_per_lenslet
        input_samples = max(config.input.shape) / self.n_lenslets
        self.source_states = iter_source_states(config)
        self._state_spot_sampling = tuple(
            self._spot_sampling(state.wavelength_m) for state in self.source_states
        )
        # The configured pupil grid holds the amplitude mask (including a
        # custom mask, which must match it) and defines lenslet illumination.
        self.pupil_samples_per_lenslet = requested or max(8, math.ceil(input_samples))
        self.pupil_shape = (self.n_lenslets * self.pupil_samples_per_lenslet,) * 2
        # Each lenslet is propagated on an integer refinement of that grid,
        # fine enough that the sampled field's periodic far field does not
        # repeat inside the detector window at the shortest wavelength. See
        # ``lenslet_field_upsampling``; adequate grids keep a factor of one.
        self.detector_window_lambda_over_d = self._widest_detector_window()
        self.field_upsampling = lenslet_field_upsampling(
            pupil_samples_per_lenslet=self.pupil_samples_per_lenslet,
            window_lambda_over_d=self.detector_window_lambda_over_d,
        )
        self.samples_per_lenslet = self.pupil_samples_per_lenslet * self.field_upsampling
        self.internal_shape = (self.n_lenslets * self.samples_per_lenslet,) * 2
        pupil = make_pupil(
            config.telescope,
            self.pupil_shape,
            config.input.grid_extent_m,
            supersampling=config.numerics.pupil_supersampling,
            backend=self.backend,
            dtype=self._real_dtype,
        )
        self.configured_pupil = pupil
        if self.field_upsampling > 1:
            # Hold each configured pupil cell's area-weighted transmission
            # constant over its refined sub-cells, so the illuminated area and
            # lenslet illumination are exactly those of the configured grid.
            pupil = self.backend.repeat(
                self.backend.repeat(pupil, self.field_upsampling, axis=0),
                self.field_upsampling,
                axis=1,
            )
        self.pupil = pupil
        self.wavefront = WavefrontInput(
            config,
            load_static_opd(config),
            backend=self.backend,
        )
        self.xx, self.yy = _coordinates(
            self.internal_shape,
            config.input.grid_extent_m,
            backend=self.backend,
        )
        self._rotation_rad = math.radians(self.settings.lenslet_grid_rotation_deg)
        self._local_centers = self._local_lenslet_centers()
        self._lgs_mean_range_m: float | None
        if config.source.lgs_ranges_m:
            weights = self.backend.asarray(config.source.lgs_range_weights, dtype=np.float64)
            if not len(weights):
                weights = self.backend.full(len(config.source.lgs_ranges_m), 1.0, dtype=np.float64)
            self._lgs_mean_range_m = self.backend.scalar(
                self.backend.average(
                    self.backend.asarray(config.source.lgs_ranges_m, dtype=np.float64),
                    weights=weights,
                )
            )
        else:
            self._lgs_mean_range_m = None
        self.source_rate = source_rate_per_s(config.source, config.telescope)
        self.clear_aperture_fraction = 1.0
        if config.source.normalization == "magnitude":
            # Spiders, segment gaps and custom masks block light the analytic
            # annulus of the magnitude normalization would otherwise count.
            self.clear_aperture_fraction = clear_aperture_fraction(
                self.configured_pupil,
                config.telescope,
                self.pupil_shape,
                config.input.grid_extent_m,
                supersampling=config.numerics.pupil_supersampling,
                backend=self.backend,
            )
            self.source_rate *= self.clear_aperture_fraction
        self.file_digests = referenced_file_digests(config)
        self._complex_dtype = complex_dtype(config.numerics.dtype)
        self._optical_blur_kernel = (
            None
            if self.settings.optical_blur_kernel_path is None
            else load_blur_kernel(self.settings.optical_blur_kernel_path)
        )
        self._charge_diffusion_kernel = self._resolve_charge_diffusion_kernel()
        base_shape = self.n_lenslets * self.settings.pixels_per_subaperture
        self.output_shape = (
            base_shape + 2 * self.settings.detector_margin_pixels,
            base_shape + 2 * self.settings.detector_margin_pixels,
        )
        self._expected_output_shape = self.output_shape
        self._base_shape = base_shape
        self._wavelengths = tuple(dict.fromkeys(state.wavelength_m for state in self.source_states))
        wavelength_index = {value: index for index, value in enumerate(self._wavelengths)}
        self._state_wavelength_indices = tuple(
            wavelength_index[state.wavelength_m] for state in self.source_states
        )
        self._geometries = tuple(
            self._build_geometry(offset) for offset in self.settings.lenslet_grid_offsets_fraction
        )
        self.lgs_kernel_margin_samples = 0
        self.lgs_kernel_radius_samples = 0
        if config.source.lgs_model == "kernel":
            self.lgs_kernel_margin_samples = self._attach_lgs_kernels()
        self._state_groups = self._build_state_groups()
        self._spot_plans = self._build_spot_plans()
        self._compiled_executor_enabled = True
        self._compiled_executor_rejections: dict[int, str] = {}

    @property
    def offset_states(self) -> tuple[tuple[float, float], ...]:
        """Effective ``[x, y]`` lenslet-grid offset of each state, in lenslet pitches."""
        return tuple(geometry.offset_fraction for geometry in self._geometries)

    def geometry(self, offset_state: int = 0) -> _LensletGeometry:
        """Return the cached lenslet geometry of one configured offset state."""
        if isinstance(offset_state, bool) or not isinstance(offset_state, (int, np.integer)):
            raise TypeError("offset_state must be an integer index")
        count = len(self._geometries)
        if not 0 <= int(offset_state) < count:
            raise ValueError(
                f"offset_state {offset_state} is out of range: the configuration declares "
                f"{count} lenslet-grid offset state(s) "
                "(shack_hartmann.lenslet_grid_offset_states_fraction)"
            )
        return self._geometries[int(offset_state)]

    # The aligned (first) state keeps the attribute names the engine has always
    # exposed, so code that inspects one-state sensors is unchanged.
    @property
    def lenslet_mask(self) -> Any:
        """Lenslet-grid pupil amplitude of offset state 0."""
        return self._geometries[0].lenslet_mask

    @property
    def lenslet_illumination(self) -> Any:
        """Mean lenslet illumination of offset state 0."""
        return self._geometries[0].lenslet_illumination

    @property
    def lenslet_valid(self) -> Any:
        """Valid-subaperture map of offset state 0."""
        return self._geometries[0].lenslet_valid

    @property
    def _total_field_flux(self) -> Any:
        return self._geometries[0].total_field_flux

    @property
    def _piston_index(self) -> tuple[int, int]:
        return self._geometries[0].piston_index

    @property
    def _field_angle_opd(self) -> tuple[Any, ...]:
        return self._geometries[0].field_angle_opd

    @property
    def _compiled_executors(self) -> dict[int, _CompiledShackHartmannExecutor]:
        return self._geometries[0].compiled_executors

    def _local_lenslet_centers(self) -> tuple[Any, Any]:
        """Return lenslet-frame centre coordinates, one per lenslet, in metres."""
        shape = (
            self.n_lenslets,
            self.samples_per_lenslet,
            self.n_lenslets,
            self.samples_per_lenslet,
        )
        return (
            self.backend.mean(self.xx.reshape(shape), axis=(1, 3)),
            self.backend.mean(self.yy.reshape(shape), axis=(1, 3)),
        )

    def _build_geometry(self, offset: tuple[float, float]) -> _LensletGeometry:
        """Build the lenslet geometry for one ``[x, y]`` grid offset.

        The lenslet grid is rotated by ``lenslet_grid_rotation_deg`` and
        shifted by ``offset`` lenslet pitches on the pupil. Equivalently, the
        pupil illumination and the OPD translate under a fixed lenslet grid:
        both are sampled at the transformed coordinates by linear
        interpolation of the area-weighted pupil, which moves the illumination
        continuously with sub-pixel offsets instead of snapping edge pixels.
        """
        config = self.config
        extent = config.input.grid_extent_m
        pitch = extent / self.n_lenslets
        offset_x, offset_y = offset
        cosine = math.cos(self._rotation_rad)
        sine = math.sin(self._rotation_rad)
        sample_indices: tuple[Any, Any] | None = None
        if abs(self._rotation_rad) > 1e-15 or bool(offset_x or offset_y):
            field_x = cosine * self.xx - sine * self.yy + offset_x * pitch
            field_y = sine * self.xx + cosine * self.yy + offset_y * pitch
            sample_x = (field_x / extent + 0.5) * self.internal_shape[1]
            sample_y = (field_y / extent + 0.5) * self.internal_shape[0]
            sample_indices = (sample_y - 0.5, sample_x - 0.5)
            pupil_on_grid = self._sample_to_lenslet_grid(
                self.pupil, sample_indices, outside="grid-constant"
            )
        else:
            field_x, field_y = self.xx, self.yy
            pupil_on_grid = self.pupil
        lenslet_mask = self._make_lenslet_mask(pupil_on_grid)
        n = self.n_lenslets
        s = self.samples_per_lenslet
        illumination = self.backend.mean(lenslet_mask.reshape(n, s, n, s), axis=(1, 3))
        local_x, local_y = self._local_centers
        centers_x = cosine * local_x - sine * local_y + offset_x * pitch
        centers_y = sine * local_x + cosine * local_y + offset_y * pitch
        total_field_flux = self.backend.asarray(
            self.backend.sum(self.backend.abs(lenslet_mask) ** 2),
            dtype=self._rate_dtype,
        )
        if self.backend.scalar(total_field_flux) <= 0.0:
            raise ValueError(
                "pupil has no illuminated pixels"
                + ("" if sample_indices is None else f" for lenslet-grid offset {list(offset)}")
            )
        piston_score = lenslet_mask / (1.0 + self.xx**2 + self.yy**2)
        flat_piston_index = int(self.backend.scalar(self.backend.argmax(piston_score)))
        subap_x = self._repeat_to_lenslet_samples(centers_x)
        subap_y = self._repeat_to_lenslet_samples(centers_y)
        return _LensletGeometry(
            offset_fraction=(float(offset_x), float(offset_y)),
            sample_indices=sample_indices,
            lenslet_mask=lenslet_mask,
            lenslet_illumination=illumination,
            lenslet_valid=illumination >= self.settings.minimum_illuminated_fraction,
            centers_x=centers_x.reshape(n * n),
            centers_y=centers_y.reshape(n * n),
            total_field_flux=total_field_flux,
            piston_index=divmod(flat_piston_index, self.internal_shape[1]),
            field_angle_opd=tuple(
                self._field_angle_for_state(state, field_x, field_y, subap_x, subap_y)
                for state in self.source_states
            ),
        )

    def _repeat_to_lenslet_samples(self, per_lenslet: Any) -> Any:
        """Hold one value per lenslet constant over that lenslet's samples."""
        s = self.samples_per_lenslet
        return self.backend.repeat(self.backend.repeat(per_lenslet, s, axis=0), s, axis=1)

    def _attach_lgs_kernels(self) -> int:
        """Build the fixed LGS kernel of every offset state; return the margin.

        The margin is the number of oversampled focal-plane samples propagated
        beyond each side of the detector window: the kernel half-width, capped
        at one alias-free period of the sampled lenslet field (the shortest
        wavelength's ``samples_per_lenslet * sampling * oversampling``).
        """
        oversampling = self.config.numerics.fft_oversampling
        morphology = lgs_kernel_morphology(self.config)
        reference_sampling = self._spot_sampling(self.config.sensor.wavelength_m)
        pitch = self.config.input.grid_extent_m / self.n_lenslets
        sample_angle_rad = (
            self.config.sensor.wavelength_m / pitch / reference_sampling / oversampling
        )
        sigma_samples = morphology.sigma_rad / sample_angle_rad
        offsets: list[_KernelOffsets] = [
            kernel_offsets(
                geometry.centers_x,
                geometry.centers_y,
                launch_m=self.config.source.lgs_launch_position_m,
                mean_range_m=self._lgs_mean_range_m,
                morphology=morphology,
                rotation_rad=self._rotation_rad,
                sample_angle_rad=sample_angle_rad,
                backend=self.backend,
            )
            for geometry in self._geometries
        ]
        radius = max(
            kernel_radius_samples(item, sigma_samples, backend=self.backend) for item in offsets
        )
        window = self.settings.pixels_per_subaperture * oversampling
        period = min(
            self.samples_per_lenslet * sampling * oversampling
            for sampling in self._state_spot_sampling
        )
        margin = max(0, min(radius, math.floor((period - window) / 2 + 1e-9)))
        self.lgs_kernel_radius_samples = radius
        for index, item in enumerate(offsets):
            self._geometries[index].lgs_kernel = _LGSKernelPlan.build(
                item,
                sigma_samples=sigma_samples,
                radius=radius,
                window=window,
                margin=margin,
                workers=self.config.numerics.fft_workers,
                complex_dtype=self._complex_dtype,
                backend=self.backend,
            )
        return margin

    def _build_state_groups(self) -> tuple[tuple[int, ...], ...]:
        """Group GPU states that share one exact focal-plane FFT geometry."""
        if self.backend.is_cpu or self.settings.field_stop_radius_lambda_over_d is not None:
            return tuple((index,) for index in range(len(self.source_states)))
        groups: dict[tuple[str, int | float], list[int]] = {}
        for index, sampling in enumerate(self._state_spot_sampling):
            geometry = spot_sampling_geometry(
                pixels=self.settings.pixels_per_subaperture,
                samples_per_lenslet=self.samples_per_lenslet,
                sampling=sampling,
                oversampling=self.config.numerics.fft_oversampling,
            )
            groups.setdefault(geometry, []).append(index)
        return tuple(tuple(indices) for indices in groups.values())

    def _build_spot_plans(self) -> tuple[_SpotPropagationPlan, ...]:
        """Cache geometry-only propagation arrays for each source state."""
        plans: dict[tuple[str, int | float, float | None], _SpotPropagationPlan] = {}
        keys: list[tuple[str, int | float, float | None]] = []
        field_stop = self.settings.field_stop_radius_lambda_over_d
        for sampling in self._state_spot_sampling:
            geometry = spot_sampling_geometry(
                pixels=self.settings.pixels_per_subaperture,
                samples_per_lenslet=self.samples_per_lenslet,
                sampling=sampling,
                oversampling=self.config.numerics.fft_oversampling,
            )
            # For an ordinary FFT without a field stop, sampling does not enter
            # the transform geometry. Sharing that plan avoids duplicate ramps.
            sampling_key = (
                float(sampling) if geometry[0] == "dft" or field_stop is not None else None
            )
            key = (geometry[0], geometry[1], sampling_key)
            keys.append(key)
            if key not in plans:
                plans[key] = _SpotPropagationPlan.build(
                    pixels=self.settings.pixels_per_subaperture,
                    samples_per_lenslet=self.samples_per_lenslet,
                    sampling=sampling,
                    oversampling=self.config.numerics.fft_oversampling,
                    field_dtype=self._complex_dtype,
                    field_stop_radius_lambda_over_d=field_stop,
                    optical_blur_kernel=self._optical_blur_kernel,
                    charge_diffusion_kernel=self._charge_diffusion_kernel,
                    backend=self.backend,
                    window_margin=self.lgs_kernel_margin_samples,
                )
        return tuple(plans[key] for key in keys)

    def _resolve_charge_diffusion_kernel(self) -> NDArray[np.float64] | None:
        """Ask ``getframes`` for the configured sensor's charge-diffusion kernel.

        The width is a property of the detector, so ``getframes`` owns both the
        measured value and the kernel model. This engine only needs the operator
        at its own focal-plane oversampling, because the spots are integrated
        onto native pixels inside the optical propagation.
        """
        fwhm_px = charge_diffusion_fwhm_px(self.config.detector)
        if fwhm_px <= 0.0:
            return None
        import getframes as gf

        if not hasattr(gf, "charge_diffusion_kernel"):
            raise RuntimeError(
                "charge_diffusion_fwhm_px requires a getframes version with "
                "charge_diffusion_kernel support"
            )
        kernel = gf.charge_diffusion_kernel(
            fwhm_px, oversampling=self.config.numerics.fft_oversampling
        )
        return cast(NDArray[np.float64], self.backend.asarray(kernel, dtype=self._rate_dtype))

    def _make_lenslet_mask(self, pupil_on_grid: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply optional square lenslet fill factor to the entrance pupil."""
        fill = self.settings.lenslet_fill_factor
        if fill >= 1.0:
            return pupil_on_grid
        s = self.samples_per_lenslet
        coords = (self.backend.arange(s, dtype=self._real_dtype) + 0.5) / s - 0.5
        yy, xx = self.backend.meshgrid(coords, coords)
        local = ((self.backend.abs(xx) <= fill / 2) & (self.backend.abs(yy) <= fill / 2)).astype(
            self._real_dtype
        )
        tiled = self.backend.tile(local, (self.n_lenslets, self.n_lenslets))
        return cast(NDArray[np.float64], pupil_on_grid * tiled)

    def _sample_to_lenslet_grid(
        self,
        array: NDArray[np.float64],
        sample_indices: tuple[Any, Any] | None,
        *,
        outside: str,
    ) -> NDArray[np.float64]:
        """Sample a physical-grid array on a transformed lenslet grid.

        Linear interpolation of the area-weighted pupil is its exact
        area-weighted translation on the sampled grid, so the illumination of a
        lenslet changes continuously with sub-sample offsets. ``outside``
        extends the array beyond its outermost sample centres: the pupil uses
        ``"grid-constant"`` (zero transmission beyond the grid, interpolated
        across the last half sample) and the OPD ``"nearest"`` (the edge value,
        so a pupil sample there sees a continuous wavefront). Plain
        ``"constant"`` would drop each edge column to zero as soon as a sample
        passed the last centre, turning a smooth translation into a staircase.
        """
        if sample_indices is None:
            return array
        sampled = self.backend.map_coordinates(
            array,
            sample_indices,
            order=1,
            mode=outside,
        )
        return cast(NDArray[np.float64], self.backend.asarray(sampled, dtype=self._real_dtype))

    def _widest_detector_window(self) -> float:
        """Return the widest window light can reach, in lenslet ``lambda / d``.

        Spot sampling in pixels per ``lambda / d`` scales with wavelength, so the
        shortest propagated wavelength sees the widest window. A configured
        circular field stop of radius ``r`` admits nothing beyond a ``2 r`` wide
        region, so it bounds the window too.
        """
        pixels = self.settings.pixels_per_subaperture
        window = max(pixels / sampling for sampling in self._state_spot_sampling)
        field_stop = self.settings.field_stop_radius_lambda_over_d
        if field_stop is not None:
            window = min(window, 2.0 * field_stop)
        return window

    def _spot_sampling(self, wavelength_m: float) -> float:
        configured = self.settings.spot_sampling_pixels_per_lambda_over_d
        if configured is None:
            assert self.settings.lenslet_focal_length_m is not None
            assert self.settings.detector_pixel_pitch_m is not None
            lenslet_pitch = self.settings.lenslet_pitch_m
            if lenslet_pitch is None:
                lenslet_pitch = self.config.telescope.pupil_diameter_m / self.n_lenslets
            configured = (
                self.settings.lenslet_focal_length_m
                * self.config.sensor.wavelength_m
                * self.settings.relay_magnification
                / (lenslet_pitch * self.settings.detector_pixel_pitch_m)
            )
        return configured * wavelength_m / self.config.sensor.wavelength_m

    def _field_angle_for_state(
        self,
        state: SourceState,
        field_x: Any,
        field_y: Any,
        subap_x: Any,
        subap_y: Any,
    ) -> NDArray[np.float64]:
        """Build immutable source/range geometry once for a persistent sensor."""
        if state.range_m is None:
            return cast(
                NDArray[np.float64], field_x * state.angle_x_rad + field_y * state.angle_y_rad
            )
        if self._lgs_mean_range_m is None:
            raise ValueError("LGS range state is missing its weighted mean range")
        launch_x, launch_y = self.config.source.lgs_launch_position_m
        range_delta = 1.0 / state.range_m - 1.0 / self._lgs_mean_range_m
        angle_x = state.angle_x_rad + (launch_x - subap_x) * range_delta
        angle_y = state.angle_y_rad + (launch_y - subap_y) * range_delta
        return cast(NDArray[np.float64], field_x * angle_x + field_y * angle_y)

    def _field(
        self,
        internal: NDArray[np.float64],
        state: SourceState,
        state_index: int,
        geometry: _LensletGeometry | None = None,
    ) -> NDArray[Any]:
        geometry = self._geometries[0] if geometry is None else geometry
        total_opd = internal + geometry.field_angle_opd[state_index]
        piston = total_opd[geometry.piston_index]
        # Removing only a fixed global piston is a numerical stabilization:
        # it makes the exact physical piston invariance survive finite precision
        # in exp(i*phase) and has no effect on the intensity.
        relative_opd = total_opd - piston
        phase = opd_to_phase(relative_opd, state.wavelength_m)
        return cast(
            NDArray[Any],
            self.backend.asarray(
                geometry.lenslet_mask * self.backend.exp(1j * phase),
                dtype=self._complex_dtype,
            ),
        )

    def _fields(
        self,
        internal: NDArray[np.float64],
        state_group: tuple[int, ...],
        *,
        samples: int = 1,
        geometry: _LensletGeometry | None = None,
    ) -> NDArray[Any]:
        """Build one device batch for source states sharing FFT geometry.

        With ``samples`` greater than one, ``internal`` carries a leading
        temporal axis and the returned batch is ordered ``(sample, state)``.
        """
        geometry = self._geometries[0] if geometry is None else geometry
        angles = self.backend.stack([geometry.field_angle_opd[index] for index in state_group])
        if samples > 1:
            total_opd = (internal[:, None, ...] + angles[None, ...]).reshape(
                samples * len(state_group), *internal.shape[1:]
            )
        else:
            total_opd = internal[None, ...] + angles
        piston = total_opd[
            :,
            geometry.piston_index[0],
            geometry.piston_index[1],
        ][:, None, None]
        wavelengths = self.backend.asarray(
            [self.source_states[index].wavelength_m for index in state_group] * samples,
            dtype=self._real_dtype,
        )[:, None, None]
        phase = opd_to_phase(total_opd - piston, wavelengths)
        return cast(
            NDArray[Any],
            self.backend.asarray(
                geometry.lenslet_mask[None, ...] * self.backend.exp(1j * phase),
                dtype=self._complex_dtype,
            ),
        )

    def render(self, wavefront: NDArray[np.float64], *, offset_state: int = 0) -> OpticalResult:
        """Render one OPD sample with the lenslet grid of ``offset_state``."""
        return self._render_samples([wavefront], self.geometry(offset_state))

    def render_integrated(
        self, wavefronts: Sequence[NDArray[np.float64]], *, offset_state: int = 0
    ) -> OpticalResult:
        """Render several temporal OPD samples as one uniformly weighted average.

        Identical in result to rendering each sample and averaging the photon
        rates, and much cheaper. The cost of a render is not its transforms --
        those are a few percent -- but the fixed dispatch cost of the many small
        elementwise operations around them, which is paid per call regardless of
        how much data each call carries. Presenting every temporal sample to one
        call amortises that cost over all of them.

        The averaging is legitimate to do early because everything downstream of
        the spot intensities is linear in them: reshaping into the mosaic,
        scaling by source flux, and summing. Averaging the spots and then
        scaling is therefore the same arithmetic as scaling and then averaging,
        which is what the caller-side loop used to do.
        """
        if not wavefronts:
            raise ValueError("render_integrated requires at least one sample")
        return self._render_samples(wavefronts, self.geometry(offset_state))

    def _render_samples(
        self,
        wavefronts: Sequence[NDArray[np.float64]],
        geometry: _LensletGeometry,
    ) -> OpticalResult:
        internal_samples = [
            self._sample_to_lenslet_grid(
                self.wavefront.opd(wavefront, target_shape=self.internal_shape),
                geometry.sample_indices,
                outside="nearest",
            )
            for wavefront in wavefronts
        ]
        sample_count = len(internal_samples)
        internal = (
            internal_samples[0] if sample_count == 1 else self.backend.stack(internal_samples)
        )
        opd = self.wavefront.input_opd(
            wavefronts[0]
            if sample_count == 1
            else self.backend.mean(
                self.backend.stack([self.backend.asarray(sample) for sample in wavefronts]),
                axis=0,
            )
        )
        compiled = self._render_compiled(internal, sample_count, geometry)
        if compiled is not None:
            return OpticalResult(
                compiled.photon_rate,
                self.source_rate,
                compiled.captured_rate_per_s,
                opd,
                compiled.spectral_photon_rate,
                self._wavelengths,
            )
        states = self.source_states
        photon_rate = self.backend.zeros(self.output_shape, dtype=self._rate_dtype)
        spectral_photon_rate = (
            None
            if len(self._wavelengths) == 1
            else self.backend.zeros(
                (len(self._wavelengths), *self.output_shape), dtype=self._rate_dtype
            )
        )
        captured: Any = 0.0
        s = self.samples_per_lenslet
        n = self.n_lenslets
        margin = self.settings.detector_margin_pixels
        for state_group in self._state_groups:
            if sample_count == 1 and len(state_group) == 1:
                subapertures = (
                    self._field(
                        internal,
                        states[state_group[0]],
                        state_group[0],
                        geometry,
                    )
                    .reshape(n, s, n, s)
                    .transpose(0, 2, 1, 3)
                    .reshape(n * n, s, s)
                )
            else:
                # Leading axis is (sample, state), flattened into the batch the
                # transform already accepts.
                subapertures = (
                    self._fields(internal, state_group, samples=sample_count, geometry=geometry)
                    .reshape(sample_count * len(state_group), n, s, n, s)
                    .transpose(0, 1, 3, 2, 4)
                    .reshape(sample_count * len(state_group) * n * n, s, s)
                )
            spots = spot_intensity(
                subapertures,
                pixels=self.settings.pixels_per_subaperture,
                samples_per_lenslet=s,
                sampling=self._state_spot_sampling[state_group[0]],
                oversampling=self.config.numerics.fft_oversampling,
                workers=self.config.numerics.fft_workers,
                field_stop_radius_lambda_over_d=self.settings.field_stop_radius_lambda_over_d,
                optical_blur_fwhm_pixels=self.settings.optical_blur_fwhm_pixels,
                optical_blur_kernel=self._optical_blur_kernel,
                charge_diffusion_kernel=self._charge_diffusion_kernel,
                source_kernel=geometry.lgs_kernel,
                _plan=self._spot_plans[state_group[0]],
                backend=self.backend,
            )
            grouped_spots = spots.reshape(
                sample_count,
                len(state_group),
                n * n,
                self.settings.pixels_per_subaperture,
                self.settings.pixels_per_subaperture,
            )
            # The uniform temporal average, taken here rather than by the
            # caller because everything below is linear in these spots.
            grouped_spots = grouped_spots.mean(axis=0) if sample_count > 1 else grouped_spots[0]
            for group_index, state_index in enumerate(state_group):
                state = states[state_index]
                base_mosaic = (
                    grouped_spots[group_index]
                    .reshape(
                        n,
                        n,
                        self.settings.pixels_per_subaperture,
                        self.settings.pixels_per_subaperture,
                    )
                    .transpose(0, 2, 1, 3)
                    .reshape((self._base_shape, self._base_shape))
                )
                if margin:
                    mosaic = self.backend.zeros(self.output_shape, dtype=self._rate_dtype)
                    mosaic[
                        margin : margin + self._base_shape,
                        margin : margin + self._base_shape,
                    ] = base_mosaic
                else:
                    mosaic = base_mosaic
                cropped_flux = self.backend.sum(mosaic)
                contribution = mosaic * (
                    self.source_rate * state.weight / geometry.total_field_flux
                )
                photon_rate += contribution
                if spectral_photon_rate is not None:
                    spectral_photon_rate[self._state_wavelength_indices[state_index]] += (
                        contribution
                    )
                captured += (
                    self.source_rate * state.weight * cropped_flux / geometry.total_field_flux
                )
        if spectral_photon_rate is None:
            spectral_photon_rate = photon_rate[None, ...]
        return OpticalResult(
            photon_rate,
            self.source_rate,
            captured,
            opd,
            spectral_photon_rate,
            self._wavelengths,
        )

    def _render_compiled(
        self, internal: Any, sample_count: int, geometry: _LensletGeometry
    ) -> Any | None:
        """Use a first-call-JIT CUDA plan when this exact geometry supports it."""
        if not self._compiled_executor_enabled:
            return None
        executor = geometry.compiled_executors.get(sample_count)
        if executor is not None:
            batched_internal = internal[None, ...] if sample_count == 1 else internal
            return executor.render(batched_internal)
        if sample_count in self._compiled_executor_rejections:
            return None
        rejection = _compiled_executor_rejection(self, sample_count)
        if rejection is not None:
            self._compiled_executor_rejections[sample_count] = rejection
            return None
        try:
            executor = _CompiledShackHartmannExecutor(self, sample_count, geometry)
        except _CompiledExecutorUnavailable as exc:
            self._compiled_executor_rejections[sample_count] = str(exc)
            return None
        geometry.compiled_executors[sample_count] = executor
        batched_internal = internal[None, ...] if sample_count == 1 else internal
        return executor.render(batched_internal)


__all__ = ["ShackHartmannEngine"]
