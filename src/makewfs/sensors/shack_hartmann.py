"""Batched Fourier-optics Shack-Hartmann wavefront sensor."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..backend import complex_dtype
from ..config import WFSConfig
from ..pupil import make_pupil
from ..radiometry import source_rate_per_s
from ..sampling import spot_intensity
from ..sensors.base import OpticalResult, SensorEngine
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
        self.pupil = make_pupil(config.telescope, self.internal_shape, config.input.grid_extent_m)
        self.wavefront = WavefrontInput(config, load_static_opd(config))
        self.xx, self.yy = _coordinates(self.internal_shape, config.input.grid_extent_m)
        self.lenslet_mask = self._make_lenslet_mask()
        self.source_rate = source_rate_per_s(config.source, config.telescope)
        self._complex_dtype = complex_dtype(config.numerics.dtype)
        self._expected_output_shape = (self.n_lenslets * self.settings.pixels_per_subaperture,) * 2

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

    def _field(self, opd: NDArray[np.float64]) -> NDArray[Any]:
        internal = self.wavefront.opd(opd, target_shape=self.internal_shape)
        self.wavefront.validate_finite_inside(internal, self.lenslet_mask)
        theta_x, theta_y = self.config.source.field_angle_arcsec
        field_angle_opd = self.xx * math.radians(theta_x / 3600.0) + self.yy * math.radians(
            theta_y / 3600.0
        )
        total_opd = internal + field_angle_opd
        illuminated = self.lenslet_mask > 0
        piston = float(np.mean(total_opd[illuminated])) if np.any(illuminated) else 0.0
        # Removing only the weighted global piston is a numerical stabilization:
        # it makes the exact physical piston invariance survive finite precision
        # in exp(i*phase) and has no effect on the intensity.
        relative_opd = total_opd - piston
        if np.ptp(total_opd[illuminated]) == 0.0:
            relative_opd = np.zeros_like(total_opd)
        phase = 2.0 * np.pi * relative_opd / self.config.sensor.wavelength_m
        return np.asarray(self.lenslet_mask * np.exp(1j * phase), dtype=self._complex_dtype)

    def render(self, wavefront: NDArray[np.float64]) -> OpticalResult:
        field = self._field(wavefront)
        s = self.samples_per_lenslet
        n = self.n_lenslets
        subapertures = field.reshape(n, s, n, s).transpose(0, 2, 1, 3).reshape(n * n, s, s)
        spots = spot_intensity(
            subapertures,
            pixels=self.settings.pixels_per_subaperture,
            samples_per_lenslet=s,
            sampling=self.settings.spot_sampling_pixels_per_lambda_over_d,
            oversampling=self.config.numerics.fft_oversampling,
            workers=self.config.numerics.fft_workers,
        )
        mosaic = (
            spots.reshape(
                n, n, self.settings.pixels_per_subaperture, self.settings.pixels_per_subaperture
            )
            .transpose(0, 2, 1, 3)
            .reshape(self._expected_output_shape)
        )
        total_field_flux = float(np.sum(np.abs(field) ** 2))
        cropped_flux = float(np.sum(mosaic))
        if total_field_flux <= 0 or cropped_flux < 0:
            raise ValueError("pupil has no illuminated pixels")
        # The unitary FFT preserves the total field energy.  Dividing by the
        # input energy retains explicit loss when the finite detector window
        # crops diffraction wings.
        photon_rate = np.asarray(mosaic * (self.source_rate / total_field_flux), dtype=np.float64)
        captured = self.source_rate * cropped_flux / total_field_flux
        return OpticalResult(photon_rate, self.source_rate, captured, wavefront)


__all__ = ["ShackHartmannEngine"]
