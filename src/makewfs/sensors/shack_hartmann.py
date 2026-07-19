"""Batched Fourier-optics Shack-Hartmann wavefront sensor."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..backend import complex_dtype
from ..config import WFSConfig
from ..provenance import referenced_file_digests
from ..pupil import make_pupil
from ..radiometry import source_rate_per_s
from ..sampling import spot_intensity
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

    def __init__(self, config: WFSConfig) -> None:
        if config.shack_hartmann is None:
            raise ValueError("Shack-Hartmann configuration is missing")
        self.config = config
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
        )
        self.wavefront = WavefrontInput(config, load_static_opd(config))
        self.xx, self.yy = _coordinates(self.internal_shape, config.input.grid_extent_m)
        self.lenslet_mask = self._make_lenslet_mask()
        self.lenslet_illumination = np.asarray(
            self.lenslet_mask.reshape(
                self.n_lenslets,
                self.samples_per_lenslet,
                self.n_lenslets,
                self.samples_per_lenslet,
            ).mean(axis=(1, 3)),
            dtype=np.float64,
        )
        self.lenslet_valid = self.lenslet_illumination >= self.settings.minimum_illuminated_fraction
        centers = self.xx.reshape(
            self.n_lenslets, self.samples_per_lenslet, self.n_lenslets, self.samples_per_lenslet
        ).mean(axis=(1, 3))
        centers_y = self.yy.reshape(
            self.n_lenslets,
            self.samples_per_lenslet,
            self.n_lenslets,
            self.samples_per_lenslet,
        ).mean(axis=(1, 3))
        self._subap_x = np.repeat(
            np.repeat(centers, self.samples_per_lenslet, axis=0),
            self.samples_per_lenslet,
            axis=1,
        )
        self._subap_y = np.repeat(
            np.repeat(centers_y, self.samples_per_lenslet, axis=0),
            self.samples_per_lenslet,
            axis=1,
        )
        self._lgs_mean_range_m: float | None
        if config.source.lgs_ranges_m:
            weights = np.asarray(config.source.lgs_range_weights, dtype=np.float64)
            if not len(weights):
                weights = np.ones(len(config.source.lgs_ranges_m), dtype=np.float64)
            self._lgs_mean_range_m = float(
                np.average(np.asarray(config.source.lgs_ranges_m), weights=weights)
            )
        else:
            self._lgs_mean_range_m = None
        self.source_rate = source_rate_per_s(config.source, config.telescope)
        self.source_states = iter_source_states(config)
        self.file_digests = referenced_file_digests(config)
        self._complex_dtype = complex_dtype(config.numerics.dtype)
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
            return self.pupil
        s = self.samples_per_lenslet
        coords = (np.arange(s, dtype=np.float64) + 0.5) / s - 0.5
        yy, xx = np.meshgrid(coords, coords)
        local = ((np.abs(xx) <= fill / 2) & (np.abs(yy) <= fill / 2)).astype(np.float64)
        tiled = np.asarray(np.tile(local, (self.n_lenslets, self.n_lenslets)), dtype=np.float64)
        return np.asarray(self.pupil * tiled, dtype=np.float64)

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
        self.wavefront.validate_finite_inside(internal, self.lenslet_mask)
        angle_x = np.full(self.internal_shape, state.angle_x_rad, dtype=np.float64)
        angle_y = np.full(self.internal_shape, state.angle_y_rad, dtype=np.float64)
        if state.range_m is not None:
            if self._lgs_mean_range_m is None:
                raise ValueError("LGS range state is missing its weighted mean range")
            launch_x, launch_y = self.config.source.lgs_launch_position_m
            range_delta = 1.0 / state.range_m - 1.0 / self._lgs_mean_range_m
            angle_x += (launch_x - self._subap_x) * range_delta
            angle_y += (launch_y - self._subap_y) * range_delta
        field_angle_opd = self.xx * angle_x + self.yy * angle_y
        total_opd = internal + field_angle_opd
        illuminated = self.lenslet_mask > 0
        piston = float(np.mean(total_opd[illuminated])) if np.any(illuminated) else 0.0
        # Removing only the weighted global piston is a numerical stabilization:
        # it makes the exact physical piston invariance survive finite precision
        # in exp(i*phase) and has no effect on the intensity.
        relative_opd = total_opd - piston
        if np.ptp(total_opd[illuminated]) == 0.0:
            relative_opd = np.zeros_like(total_opd)
        phase = 2.0 * np.pi * relative_opd / state.wavelength_m
        return np.asarray(self.lenslet_mask * np.exp(1j * phase), dtype=self._complex_dtype)

    def render(self, wavefront: NDArray[np.float64]) -> OpticalResult:
        internal = self.wavefront.opd(wavefront, target_shape=self.internal_shape)
        self.wavefront.validate_finite_inside(internal, self.lenslet_mask)
        states = self.source_states
        photon_rate = np.zeros(self.output_shape, dtype=np.float64)
        total_field_flux: float | None = None
        captured = 0.0
        s = self.samples_per_lenslet
        n = self.n_lenslets
        base_shape = n * self.settings.pixels_per_subaperture
        margin = self.settings.detector_margin_pixels
        for state in states:
            field = self._field(internal, state)
            if total_field_flux is None:
                total_field_flux = float(np.sum(np.abs(field) ** 2))
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
            mosaic = np.zeros(self.output_shape, dtype=np.float64)
            mosaic[margin : margin + base_shape, margin : margin + base_shape] = base_mosaic
            cropped_flux = float(np.sum(mosaic))
            if cropped_flux < 0:
                raise ValueError("pupil propagation produced negative flux")
            photon_rate += mosaic * (self.source_rate * state.weight / total_field_flux)
            captured += self.source_rate * state.weight * cropped_flux / total_field_flux
        if total_field_flux is None or total_field_flux <= 0:
            raise ValueError("pupil has no illuminated pixels")
        return OpticalResult(photon_rate, self.source_rate, captured, wavefront)


__all__ = ["ShackHartmannEngine"]
