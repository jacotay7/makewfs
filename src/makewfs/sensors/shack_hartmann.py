"""Batched Fourier-optics Shack-Hartmann wavefront sensor."""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from ..backend import ArrayBackend, complex_dtype, real_dtype
from ..config import WFSConfig
from ..provenance import referenced_file_digests
from ..pupil import make_pupil
from ..radiometry import source_rate_per_s
from ..sampling import load_blur_kernel, spot_intensity
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
            pitch = config.telescope.pupil_diameter_m / self.n_lenslets
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
        pitch = config.telescope.pupil_diameter_m / self.n_lenslets
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
        base_shape = self.n_lenslets * self.settings.pixels_per_subaperture
        self.output_shape = (
            base_shape + 2 * self.settings.detector_margin_pixels,
            base_shape + 2 * self.settings.detector_margin_pixels,
        )
        self._expected_output_shape = self.output_shape

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
            lenslet_pitch = self.config.telescope.pupil_diameter_m / self.n_lenslets
            configured = (
                self.settings.lenslet_focal_length_m
                * self.config.sensor.wavelength_m
                * self.settings.relay_magnification
                / (lenslet_pitch * self.settings.detector_pixel_pitch_m)
            )
        return configured * wavelength_m / self.config.sensor.wavelength_m

    def _field(
        self,
        internal: NDArray[np.float64],
        state: SourceState,
    ) -> NDArray[Any]:
        internal = self._sample_to_lenslet_grid(internal)
        self.wavefront.validate_finite_inside(internal, self.lenslet_mask)
        angle_x = self.backend.full(self.internal_shape, state.angle_x_rad, dtype=self._real_dtype)
        angle_y = self.backend.full(self.internal_shape, state.angle_y_rad, dtype=self._real_dtype)
        if state.range_m is not None:
            if self._lgs_mean_range_m is None:
                raise ValueError("LGS range state is missing its weighted mean range")
            launch_x, launch_y = self.config.source.lgs_launch_position_m
            range_delta = 1.0 / state.range_m - 1.0 / self._lgs_mean_range_m
            angle_x += (launch_x - self._subap_x) * range_delta
            angle_y += (launch_y - self._subap_y) * range_delta
        field_angle_opd = self._field_x * angle_x + self._field_y * angle_y
        total_opd = internal + field_angle_opd
        illuminated = self.lenslet_mask > 0
        piston = (
            self.backend.scalar(self.backend.mean(total_opd[illuminated]))
            if self.backend.scalar(self.backend.any(illuminated))
            else 0.0
        )
        # Removing only the weighted global piston is a numerical stabilization:
        # it makes the exact physical piston invariance survive finite precision
        # in exp(i*phase) and has no effect on the intensity.
        relative_opd = total_opd - piston
        if self.backend.scalar(self.backend.ptp(total_opd[illuminated])) == 0.0:
            relative_opd = self.backend.zeros_like(total_opd)
        phase = 2.0 * math.pi * relative_opd / state.wavelength_m
        return cast(
            NDArray[Any],
            self.backend.asarray(
                self.lenslet_mask * self.backend.exp(1j * phase),
                dtype=self._complex_dtype,
            ),
        )

    def render(self, wavefront: NDArray[np.float64]) -> OpticalResult:
        internal = self.wavefront.opd(wavefront, target_shape=self.internal_shape)
        self.wavefront.validate_finite_inside(internal, self.pupil)
        states = self.source_states
        photon_rate = self.backend.zeros(self.output_shape, dtype=self._rate_dtype)
        wavelengths = tuple(dict.fromkeys(state.wavelength_m for state in states))
        wavelength_index = {wavelength: index for index, wavelength in enumerate(wavelengths)}
        spectral_photon_rate = self.backend.zeros(
            (len(wavelengths), *self.output_shape), dtype=self._rate_dtype
        )
        total_field_flux: float | None = None
        captured = 0.0
        s = self.samples_per_lenslet
        n = self.n_lenslets
        base_shape = n * self.settings.pixels_per_subaperture
        margin = self.settings.detector_margin_pixels
        for state in states:
            field = self._field(internal, state)
            if total_field_flux is None:
                total_field_flux = self.backend.scalar(
                    self.backend.sum(self.backend.abs(field) ** 2)
                )
            subapertures = field.reshape(n, s, n, s).transpose(0, 2, 1, 3).reshape(n * n, s, s)
            sampling = self._spot_sampling(state.wavelength_m)
            spots = spot_intensity(
                subapertures,
                pixels=self.settings.pixels_per_subaperture,
                samples_per_lenslet=s,
                sampling=sampling,
                oversampling=self.config.numerics.fft_oversampling,
                workers=self.config.numerics.fft_workers,
                field_stop_radius_lambda_over_d=self.settings.field_stop_radius_lambda_over_d,
                optical_blur_fwhm_pixels=self.settings.optical_blur_fwhm_pixels,
                optical_blur_kernel=self._optical_blur_kernel,
                backend=self.backend,
            )
            base_mosaic = (
                spots.reshape(
                    n,
                    n,
                    self.settings.pixels_per_subaperture,
                    self.settings.pixels_per_subaperture,
                )
                .transpose(0, 2, 1, 3)
                .reshape((base_shape, base_shape))
            )
            mosaic = self.backend.zeros(self.output_shape, dtype=self._rate_dtype)
            mosaic[margin : margin + base_shape, margin : margin + base_shape] = base_mosaic
            cropped_flux = self.backend.scalar(self.backend.sum(mosaic))
            if cropped_flux < 0:
                raise ValueError("pupil propagation produced negative flux")
            contribution = mosaic * (self.source_rate * state.weight / total_field_flux)
            photon_rate += contribution
            spectral_photon_rate[wavelength_index[state.wavelength_m]] += contribution
            captured += self.source_rate * state.weight * cropped_flux / total_field_flux
        if total_field_flux is None or total_field_flux <= 0:
            raise ValueError("pupil has no illuminated pixels")
        return OpticalResult(
            photon_rate,
            self.source_rate,
            captured,
            wavefront,
            spectral_photon_rate,
            wavelengths,
        )


__all__ = ["ShackHartmannEngine"]
