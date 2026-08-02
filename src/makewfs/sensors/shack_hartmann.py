"""Batched Fourier-optics Shack-Hartmann wavefront sensor."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from ..backend import ArrayBackend, complex_dtype, real_dtype
from ..config import WFSConfig
from ..detector import charge_diffusion_fwhm_px
from ..provenance import referenced_file_digests
from ..pupil import make_pupil
from ..radiometry import source_rate_per_s
from ..sampling import (
    _SpotPropagationPlan,
    load_blur_kernel,
    spot_intensity,
    spot_sampling_geometry,
)
from ..sensors.base import OpticalResult, SensorEngine
from ..source import SourceState, iter_source_states
from ..wavefront import WavefrontInput, _coordinates, load_static_opd


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
        self.samples_per_lenslet = requested or max(8, math.ceil(input_samples))
        self.internal_shape = (self.n_lenslets * self.samples_per_lenslet,) * 2
        self.pupil = make_pupil(
            config.telescope,
            self.internal_shape,
            config.input.grid_extent_m,
            supersampling=config.numerics.pupil_supersampling,
            backend=self.backend,
            dtype=self._real_dtype,
        )
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
        rotation = math.radians(self.settings.lenslet_grid_rotation_deg)
        offset_x, offset_y = self.settings.lenslet_grid_offset_fraction
        self._grid_transform_enabled = abs(rotation) > 1e-15 or bool(offset_x or offset_y)
        self._sample_indices: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None
        if self._grid_transform_enabled:
            pitch = config.input.grid_extent_m / self.n_lenslets
            cosine = math.cos(rotation)
            sine = math.sin(rotation)
            self._field_x = cosine * self.xx - sine * self.yy + offset_x * pitch
            self._field_y = sine * self.xx + cosine * self.yy + offset_y * pitch
            sample_x = (self._field_x / config.input.grid_extent_m + 0.5) * self.internal_shape[1]
            sample_y = (self._field_y / config.input.grid_extent_m + 0.5) * self.internal_shape[0]
            self._sample_indices = (sample_y - 0.5, sample_x - 0.5)
            self._pupil_on_lenslet_grid = self._sample_to_lenslet_grid(self.pupil)
        else:
            self._field_x = self.xx
            self._field_y = self.yy
            self._pupil_on_lenslet_grid = self.pupil
        self.lenslet_mask = self._make_lenslet_mask()
        self.lenslet_illumination = self.backend.mean(
            self.lenslet_mask.reshape(
                self.n_lenslets,
                self.samples_per_lenslet,
                self.n_lenslets,
                self.samples_per_lenslet,
            ),
            axis=(1, 3),
        )
        self.lenslet_valid = self.lenslet_illumination >= self.settings.minimum_illuminated_fraction
        local_centers_x = self.backend.mean(
            self.xx.reshape(
                self.n_lenslets,
                self.samples_per_lenslet,
                self.n_lenslets,
                self.samples_per_lenslet,
            ),
            axis=(1, 3),
        )
        local_centers_y = self.backend.mean(
            self.yy.reshape(
                self.n_lenslets,
                self.samples_per_lenslet,
                self.n_lenslets,
                self.samples_per_lenslet,
            ),
            axis=(1, 3),
        )
        pitch = config.input.grid_extent_m / self.n_lenslets
        cosine = math.cos(rotation)
        sine = math.sin(rotation)
        centers = cosine * local_centers_x - sine * local_centers_y + offset_x * pitch
        centers_y = sine * local_centers_x + cosine * local_centers_y + offset_y * pitch
        self._subap_x = self.backend.repeat(
            self.backend.repeat(centers, self.samples_per_lenslet, axis=0),
            self.samples_per_lenslet,
            axis=1,
        )
        self._subap_y = self.backend.repeat(
            self.backend.repeat(centers_y, self.samples_per_lenslet, axis=0),
            self.samples_per_lenslet,
            axis=1,
        )
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
        self.source_states = iter_source_states(config)
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
        self._total_field_flux = self.backend.asarray(
            self.backend.sum(self.backend.abs(self.lenslet_mask) ** 2),
            dtype=self._rate_dtype,
        )
        if self.backend.scalar(self._total_field_flux) <= 0.0:
            raise ValueError("pupil has no illuminated pixels")
        piston_score = self.lenslet_mask / (1.0 + self.xx**2 + self.yy**2)
        flat_piston_index = int(self.backend.scalar(self.backend.argmax(piston_score)))
        self._piston_index = divmod(flat_piston_index, self.internal_shape[1])
        self._field_angle_opd = tuple(
            self._field_angle_for_state(state) for state in self.source_states
        )
        self._state_spot_sampling = tuple(
            self._spot_sampling(state.wavelength_m) for state in self.source_states
        )
        self._wavelengths = tuple(dict.fromkeys(state.wavelength_m for state in self.source_states))
        wavelength_index = {value: index for index, value in enumerate(self._wavelengths)}
        self._state_wavelength_indices = tuple(
            wavelength_index[state.wavelength_m] for state in self.source_states
        )
        self._state_groups = self._build_state_groups()
        self._spot_plans = self._build_spot_plans()

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

    def _make_lenslet_mask(self) -> NDArray[np.float64]:
        """Apply optional square lenslet fill factor to the entrance pupil."""
        fill = self.settings.lenslet_fill_factor
        if fill >= 1.0:
            return self._pupil_on_lenslet_grid
        s = self.samples_per_lenslet
        coords = (self.backend.arange(s, dtype=self._real_dtype) + 0.5) / s - 0.5
        yy, xx = self.backend.meshgrid(coords, coords)
        local = ((self.backend.abs(xx) <= fill / 2) & (self.backend.abs(yy) <= fill / 2)).astype(
            self._real_dtype
        )
        tiled = self.backend.tile(local, (self.n_lenslets, self.n_lenslets))
        return cast(NDArray[np.float64], self._pupil_on_lenslet_grid * tiled)

    def _sample_to_lenslet_grid(self, array: NDArray[np.float64]) -> NDArray[np.float64]:
        """Sample a physical-grid array on the configured lenslet grid."""
        if self._sample_indices is None:
            return array
        sampled = self.backend.map_coordinates(
            array,
            self._sample_indices,
            order=1,
            mode="constant",
        )
        return cast(NDArray[np.float64], self.backend.asarray(sampled, dtype=self._real_dtype))

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

    def _field_angle_for_state(self, state: SourceState) -> NDArray[np.float64]:
        """Build immutable source/range geometry once for a persistent sensor."""
        if state.range_m is None:
            return self._field_x * state.angle_x_rad + self._field_y * state.angle_y_rad
        if self._lgs_mean_range_m is None:
            raise ValueError("LGS range state is missing its weighted mean range")
        launch_x, launch_y = self.config.source.lgs_launch_position_m
        range_delta = 1.0 / state.range_m - 1.0 / self._lgs_mean_range_m
        angle_x = state.angle_x_rad + (launch_x - self._subap_x) * range_delta
        angle_y = state.angle_y_rad + (launch_y - self._subap_y) * range_delta
        return cast(NDArray[np.float64], self._field_x * angle_x + self._field_y * angle_y)

    def _field(
        self,
        internal: NDArray[np.float64],
        state: SourceState,
        state_index: int,
    ) -> NDArray[Any]:
        total_opd = internal + self._field_angle_opd[state_index]
        piston = total_opd[self._piston_index]
        # Removing only a fixed global piston is a numerical stabilization:
        # it makes the exact physical piston invariance survive finite precision
        # in exp(i*phase) and has no effect on the intensity.
        relative_opd = total_opd - piston
        phase = 2.0 * math.pi * relative_opd / state.wavelength_m
        return cast(
            NDArray[Any],
            self.backend.asarray(
                self.lenslet_mask * self.backend.exp(1j * phase),
                dtype=self._complex_dtype,
            ),
        )

    def _fields(
        self,
        internal: NDArray[np.float64],
        state_group: tuple[int, ...],
        *,
        samples: int = 1,
    ) -> NDArray[Any]:
        """Build one device batch for source states sharing FFT geometry.

        With ``samples`` greater than one, ``internal`` carries a leading
        temporal axis and the returned batch is ordered ``(sample, state)``.
        """
        angles = self.backend.stack([self._field_angle_opd[index] for index in state_group])
        if samples > 1:
            total_opd = (internal[:, None, ...] + angles[None, ...]).reshape(
                samples * len(state_group), *internal.shape[1:]
            )
        else:
            total_opd = internal[None, ...] + angles
        piston = total_opd[
            :,
            self._piston_index[0],
            self._piston_index[1],
        ][:, None, None]
        wavelengths = self.backend.asarray(
            [self.source_states[index].wavelength_m for index in state_group] * samples,
            dtype=self._real_dtype,
        )[:, None, None]
        phase = 2.0 * math.pi * (total_opd - piston) / wavelengths
        return cast(
            NDArray[Any],
            self.backend.asarray(
                self.lenslet_mask[None, ...] * self.backend.exp(1j * phase),
                dtype=self._complex_dtype,
            ),
        )

    def render(self, wavefront: NDArray[np.float64]) -> OpticalResult:
        return self._render_samples([wavefront])

    def render_integrated(self, wavefronts: Sequence[NDArray[np.float64]]) -> OpticalResult:
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
        return self._render_samples(wavefronts)

    def _render_samples(self, wavefronts: Sequence[NDArray[np.float64]]) -> OpticalResult:
        internal_samples = [
            self._sample_to_lenslet_grid(
                self.wavefront.opd(wavefront, target_shape=self.internal_shape)
            )
            for wavefront in wavefronts
        ]
        sample_count = len(internal_samples)
        internal = (
            internal_samples[0] if sample_count == 1 else self.backend.stack(internal_samples)
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
                    )
                    .reshape(n, s, n, s)
                    .transpose(0, 2, 1, 3)
                    .reshape(n * n, s, s)
                )
            else:
                # Leading axis is (sample, state), flattened into the batch the
                # transform already accepts.
                subapertures = (
                    self._fields(internal, state_group, samples=sample_count)
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
                contribution = mosaic * (self.source_rate * state.weight / self._total_field_flux)
                photon_rate += contribution
                if spectral_photon_rate is not None:
                    spectral_photon_rate[self._state_wavelength_indices[state_index]] += (
                        contribution
                    )
                captured += self.source_rate * state.weight * cropped_flux / self._total_field_flux
        if spectral_photon_rate is None:
            spectral_photon_rate = photon_rate[None, ...]
        # The reported OPD is the exposure's mean, matching the mean photon rate
        # the detector is handed; a single sample is its own mean.
        opd = (
            wavefronts[0]
            if sample_count == 1
            else self.backend.mean(
                self.backend.stack([self.backend.asarray(sample) for sample in wavefronts]),
                axis=0,
            )
        )
        return OpticalResult(
            photon_rate,
            self.source_rate,
            captured,
            opd,
            spectral_photon_rate,
            self._wavelengths,
        )


__all__ = ["ShackHartmannEngine"]
