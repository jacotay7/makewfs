"""CPU Fourier-optics model of a four-face pyramid wavefront sensor."""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from ..backend import ArrayBackend, centered_fft2, centered_ifft2, complex_dtype, real_dtype
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

    def __init__(self, config: WFSConfig, *, backend: ArrayBackend | None = None) -> None:
        if config.pyramid is None:
            raise ValueError("pyramid configuration is missing")
        self.config = config
        self.backend = self.resolve_backend(backend)
        self._real_dtype = real_dtype(config.numerics.dtype)
        self._rate_dtype = real_dtype("float64")
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
        self.nfft = self.backend.next_fast_length(max(self.output_shape))
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
        self.source_rate = source_rate_per_s(config.source, config.telescope)
        self.source_states = iter_source_states(config)
        self.file_digests = referenced_file_digests(config)
        self._complex_dtype = complex_dtype(config.numerics.dtype)
        self._mask = self._make_pyramid_mask()

    def _make_pyramid_mask(self) -> NDArray[Any]:
        """Build four signed focal-plane ramps that separate the pupils."""
        frequencies = self.backend.fftshift(self.backend.fftfreq(self.nfft)) * self.nfft
        fy, fx = self.backend.meshgrid(frequencies, frequencies, indexing="ij")
        sign_x = self.backend.where(fx >= 0.0, 1.0, -1.0)
        sign_y = self.backend.where(fy >= 0.0, 1.0, -1.0)
        half_separation = self.settings.pupil_separation_pixels / 2.0
        phase = -2.0 * math.pi * half_separation * (sign_x * fx + sign_y * fy) / self.nfft
        return cast(
            NDArray[Any],
            self.backend.asarray(self.backend.exp(1j * phase), dtype=self._complex_dtype),
        )

    def _base_field(self, internal: NDArray[np.float64], state: SourceState) -> NDArray[Any]:
        self.wavefront.validate_finite_inside(internal, self.pupil)
        field_angle_opd = self.xx * state.angle_x_rad + self.yy * state.angle_y_rad
        total_opd = internal + field_angle_opd
        illuminated = self.pupil > 0
        illuminated_weight = self.backend.astype(illuminated, total_opd.dtype)
        illuminated_count = self.backend.sum(illuminated_weight)
        piston = self.backend.sum(total_opd * illuminated_weight) / self.backend.where(
            illuminated_count > 0,
            illuminated_count,
            1.0,
        )
        relative_opd = total_opd - piston
        phase = 2.0 * math.pi * relative_opd / state.wavelength_m
        return cast(
            NDArray[Any],
            self.backend.asarray(
                self.pupil * self.backend.exp(1j * phase),
                dtype=self._complex_dtype,
            ),
        )

    def _fields(self, internal: NDArray[np.float64], state: SourceState) -> NDArray[Any]:
        base = self._base_field(internal, state)
        radius = self.settings.modulation_radius_lambda_over_d
        samples = self.settings.modulation_samples
        if radius == 0.0:
            return base[None, ...]
        angles = self.backend.arange(samples, dtype=self._real_dtype) * 2.0 * math.pi / samples
        diameter = self.config.telescope.pupil_diameter_m
        cosines = self.backend.cos(angles)[:, None, None]
        sines = self.backend.sin(angles)[:, None, None]
        tilts = self.backend.exp(
            2j
            * math.pi
            * radius
            * (cosines * self.xx[None, ...] + sines * self.yy[None, ...])
            / diameter
        )
        return cast(
            NDArray[Any],
            self.backend.asarray(base[None, ...] * tilts, dtype=self._complex_dtype),
        )

    def render(self, wavefront: NDArray[np.float64]) -> OpticalResult:
        internal = self.wavefront.opd(wavefront, target_shape=self.internal_shape)
        self.wavefront.validate_finite_inside(internal, self.pupil)
        photon_rate = self.backend.zeros(self.output_shape, dtype=self._rate_dtype)
        wavelengths = tuple(dict.fromkeys(state.wavelength_m for state in self.source_states))
        wavelength_index = {wavelength: index for index, wavelength in enumerate(wavelengths)}
        spectral_photon_rate = self.backend.zeros(
            (len(wavelengths), *self.output_shape), dtype=self._rate_dtype
        )
        total_field_flux: Any | None = None
        captured: Any | None = None
        pixels = self.settings.pixels_across_pupil
        separation = self.settings.pupil_separation_pixels
        margin = self.settings.detector_margin_pixels
        for state in self.source_states:
            fields = self._fields(internal, state)
            padded = pad_center(fields, (self.nfft, self.nfft), backend=self.backend)
            focal = centered_fft2(
                padded,
                workers=self.config.numerics.fft_workers,
                backend=self.backend,
            )
            exit_pupil = centered_ifft2(
                focal * self._mask[None, ...],
                workers=self.config.numerics.fft_workers,
                backend=self.backend,
            )
            intensity = self.backend.abs(exit_pupil) ** 2
            base_output_shape = (pixels + separation, pixels + separation)
            cropped = crop_center(intensity, base_output_shape)
            mosaic = self.backend.mean(cropped, axis=0)
            if margin:
                padded = self.backend.zeros(self.output_shape, dtype=self._rate_dtype)
                padded[
                    margin : margin + base_output_shape[0], margin : margin + base_output_shape[1]
                ] = mosaic
                mosaic = padded
            if total_field_flux is None:
                total_field_flux = self.backend.asarray(
                    self.backend.sum(self.backend.abs(fields[0]) ** 2),
                    dtype=self._rate_dtype,
                )
            cropped_flux = self.backend.sum(mosaic)
            normalizer = self.backend.where(total_field_flux > 0, total_field_flux, 1.0)
            contribution = mosaic * (self.source_rate * state.weight / normalizer)
            photon_rate += contribution
            spectral_photon_rate[wavelength_index[state.wavelength_m]] += contribution
            captured = (0.0 if captured is None else captured) + (
                self.source_rate * state.weight * cropped_flux / normalizer
            )
        if total_field_flux is None or self.backend.scalar(total_field_flux) <= 0.0:
            raise ValueError("pupil has no illuminated pixels")
        assert captured is not None
        return OpticalResult(
            photon_rate,
            self.source_rate,
            self.backend.scalar(captured),
            wavefront,
            spectral_photon_rate,
            wavelengths,
        )


__all__ = ["PyramidEngine"]
