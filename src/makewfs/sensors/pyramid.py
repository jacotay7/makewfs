"""CPU Fourier-optics model of a four-face pyramid wavefront sensor."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..backend import centered_fft2, centered_ifft2, complex_dtype, next_fast_length
from ..config import WFSConfig
from ..provenance import referenced_file_digests
from ..pupil import make_pupil
from ..radiometry import source_rate_per_s
from ..sampling import crop_center, pad_center
from ..sensors.base import OpticalResult, SensorEngine
from ..source import SourceState, iter_source_states
from ..wavefront import WavefrontInput, _coordinates, load_static_opd


class PyramidEngine(SensorEngine):
    """Render a monochromatic four-face pyramid pupil image.

    The phase mask is represented by four piecewise-linear focal-plane phase
    ramps. Each ramp re-images the entrance pupil at one of four locations in
    the output plane. Modulation is implemented as a batch of source tilts and
    averaged before detector sampling.
    """

    kind = "pyramid"

    def __init__(self, config: WFSConfig) -> None:
        if config.pyramid is None:
            raise ValueError("pyramid configuration is missing")
        self.config = config
        self.settings = config.pyramid
        if config.source.lgs_ranges_m:
            raise NotImplementedError(
                "range-resolved LGS elongation is currently implemented for Shack-Hartmann only"
            )
        pixels = self.settings.pixels_across_pupil
        separation = self.settings.pupil_separation_pixels
        margin = self.settings.detector_margin_pixels
        self.internal_shape = (pixels, pixels)
        self.output_shape = (pixels + separation + 2 * margin, pixels + separation + 2 * margin)
        self.nfft = next_fast_length(max(self.output_shape))
        self.pupil = make_pupil(
            config.telescope,
            self.internal_shape,
            config.input.grid_extent_m,
            supersampling=config.numerics.pupil_supersampling,
        )
        self.wavefront = WavefrontInput(config, load_static_opd(config))
        self.xx, self.yy = _coordinates(self.internal_shape, config.input.grid_extent_m)
        self.source_rate = source_rate_per_s(config.source, config.telescope)
        self.source_states = iter_source_states(config)
        self.file_digests = referenced_file_digests(config)
        self._complex_dtype = complex_dtype(config.numerics.dtype)
        self._mask = self._make_pyramid_mask()

    def _make_pyramid_mask(self) -> NDArray[Any]:
        """Build four signed focal-plane ramps that separate the pupils."""
        frequencies = np.fft.fftshift(np.fft.fftfreq(self.nfft) * self.nfft)
        fy, fx = np.meshgrid(frequencies, frequencies, indexing="ij")
        sign_x = np.where(fx >= 0.0, 1.0, -1.0)
        sign_y = np.where(fy >= 0.0, 1.0, -1.0)
        half_separation = self.settings.pupil_separation_pixels / 2.0
        phase = -2.0 * np.pi * half_separation * (sign_x * fx + sign_y * fy) / self.nfft
        return np.asarray(np.exp(1j * phase), dtype=self._complex_dtype)

    def _base_field(self, internal: NDArray[np.float64], state: SourceState) -> NDArray[Any]:
        self.wavefront.validate_finite_inside(internal, self.pupil)
        field_angle_opd = self.xx * state.angle_x_rad + self.yy * state.angle_y_rad
        total_opd = internal + field_angle_opd
        illuminated = self.pupil > 0
        piston = float(np.mean(total_opd[illuminated])) if np.any(illuminated) else 0.0
        relative_opd = total_opd - piston
        if np.ptp(total_opd[illuminated]) == 0.0:
            relative_opd = np.zeros_like(total_opd)
        phase = 2.0 * np.pi * relative_opd / state.wavelength_m
        return np.asarray(self.pupil * np.exp(1j * phase), dtype=self._complex_dtype)

    def _fields(self, internal: NDArray[np.float64], state: SourceState) -> NDArray[Any]:
        base = self._base_field(internal, state)
        radius = self.settings.modulation_radius_lambda_over_d
        samples = self.settings.modulation_samples
        if radius == 0.0:
            return base[None, ...]
        angles = np.arange(samples, dtype=np.float64) * 2.0 * np.pi / samples
        diameter = self.config.telescope.pupil_diameter_m
        cosines = np.cos(angles)[:, None, None]
        sines = np.sin(angles)[:, None, None]
        tilts = np.exp(
            2j
            * np.pi
            * radius
            * (cosines * self.xx[None, ...] + sines * self.yy[None, ...])
            / diameter
        ).astype(self._complex_dtype, copy=False)
        return np.asarray(base[None, ...] * tilts, dtype=self._complex_dtype)

    def render(self, wavefront: NDArray[np.float64]) -> OpticalResult:
        internal = self.wavefront.opd(wavefront, target_shape=self.internal_shape)
        self.wavefront.validate_finite_inside(internal, self.pupil)
        photon_rate = np.zeros(self.output_shape, dtype=np.float64)
        total_field_flux: float | None = None
        captured = 0.0
        pixels = self.settings.pixels_across_pupil
        separation = self.settings.pupil_separation_pixels
        margin = self.settings.detector_margin_pixels
        for state in self.source_states:
            fields = self._fields(internal, state)
            padded = pad_center(fields, (self.nfft, self.nfft))
            focal = centered_fft2(padded, workers=self.config.numerics.fft_workers)
            exit_pupil = centered_ifft2(
                focal * self._mask[None, ...], workers=self.config.numerics.fft_workers
            )
            intensity = np.abs(exit_pupil) ** 2
            base_output_shape = (pixels + separation, pixels + separation)
            cropped = crop_center(intensity, base_output_shape)
            mosaic = np.asarray(np.mean(cropped, axis=0), dtype=np.float64)
            if margin:
                padded = np.zeros(self.output_shape, dtype=np.float64)
                padded[
                    margin : margin + base_output_shape[0], margin : margin + base_output_shape[1]
                ] = mosaic
                mosaic = padded
            if total_field_flux is None:
                total_field_flux = float(np.sum(np.abs(fields[0]) ** 2))
            cropped_flux = float(np.sum(mosaic))
            if cropped_flux < 0.0:
                raise ValueError("pyramid propagation produced negative flux")
            photon_rate += mosaic * (self.source_rate * state.weight / total_field_flux)
            captured += self.source_rate * state.weight * cropped_flux / total_field_flux
        if total_field_flux is None or total_field_flux <= 0.0:
            raise ValueError("pupil has no illuminated pixels")
        return OpticalResult(photon_rate, self.source_rate, captured, wavefront)


__all__ = ["PyramidEngine"]
