"""CPU Fourier-optics model of a four-face pyramid wavefront sensor."""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
from aocore import opd_to_phase
from numpy.typing import NDArray

from ..backend import ArrayBackend, complex_dtype, real_dtype
from ..config import WFSConfig
from ..provenance import referenced_file_digests
from ..pupil import make_pupil
from ..radiometry import clear_aperture_fraction, source_rate_per_s
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
        # Oversample the propagation grid so the diffraction halo lands outside
        # the detector crop instead of wrapping onto the pupil rims; cropped
        # flux is reported as captured rate, never renormalized.
        self.nfft = self.backend.next_fast_length(
            config.numerics.fft_oversampling * max(self.output_shape)
        )
        self.pupil = make_pupil(
            config.telescope,
            self.internal_shape,
            config.input.grid_extent_m,
            supersampling=config.numerics.pupil_supersampling,
            backend=self.backend,
            dtype=self._real_dtype,
        )
        self.configured_pupil = self.pupil
        self.wavefront = WavefrontInput(
            config,
            load_static_opd(config),
            backend=self.backend,
        )
        self.xx, self.yy = _coordinates(
            self.internal_shape,
            config.input.grid_extent_m,
            backend=self.backend,
            dtype=self._real_dtype,
        )
        self.source_rate = source_rate_per_s(config.source, config.telescope)
        self.clear_aperture_fraction = 1.0
        if config.source.normalization == "magnitude":
            # Spiders, segment gaps and custom masks block light the analytic
            # annulus of the magnitude normalization would otherwise count.
            self.clear_aperture_fraction = clear_aperture_fraction(
                self.pupil,
                config.telescope,
                self.internal_shape,
                config.input.grid_extent_m,
                supersampling=config.numerics.pupil_supersampling,
                backend=self.backend,
            )
            self.source_rate *= self.clear_aperture_fraction
        self.source_states = iter_source_states(config)
        self.file_digests = referenced_file_digests(config)
        self._complex_dtype = complex_dtype(config.numerics.dtype)
        self._mask = self._make_pyramid_mask()
        self._total_field_flux = self.backend.asarray(
            self.backend.sum(self.backend.abs(self.pupil) ** 2), dtype=self._rate_dtype
        )
        if self.backend.scalar(self._total_field_flux) <= 0.0:
            raise ValueError("pupil has no illuminated pixels")
        piston_score = self.pupil / (1.0 + self.xx**2 + self.yy**2)
        flat_piston_index = int(self.backend.scalar(self.backend.argmax(piston_score)))
        self._piston_index = divmod(flat_piston_index, self.internal_shape[1])
        self._field_angle_opd = tuple(
            self.xx * state.angle_x_rad + self.yy * state.angle_y_rad
            for state in self.source_states
        )
        self._modulation_tilts = self._make_modulation_tilts()
        self._modulated_pupil = (
            None
            if self._modulation_tilts is None
            else self.pupil[None, ...] * self._modulation_tilts
        )
        self._wavelengths = tuple(dict.fromkeys(state.wavelength_m for state in self.source_states))
        wavelength_index = {value: index for index, value in enumerate(self._wavelengths)}
        self._state_wavelength_indices = tuple(
            wavelength_index[state.wavelength_m] for state in self.source_states
        )
        self._base_output_shape = (pixels + separation, pixels + separation)
        self._build_unshifted_geometry()

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

    def _build_unshifted_geometry(self) -> None:
        """Cache the propagation geometry on the unshifted FFT grid.

        The reference chain is ``pad_center``, ``ifftshift``, ``fft2``,
        ``fftshift``, mask, ``ifftshift``, ``ifft2``, ``fftshift``, intensity
        and ``crop_center``. The shifts are pure permutations, so the two in the
        middle cancel once the mask is stored ``ifftshift``-ed, and the outer two
        become index lists: where each padded pupil row lands after the input
        shift, and which unshifted output rows the centred crop keeps. Values
        are unchanged; four full-grid copies per state are not made.
        """
        n = self.nfft
        pixels = self.internal_shape[0]
        crop = self._base_output_shape[0]
        # ``ifftshift`` sends padded index i to (i - n // 2) mod n, and the
        # output ``fftshift`` reads unshifted index (i - n // 2) mod n.
        self._pupil_start = ((n - pixels) // 2 - n // 2) % n
        self._crop_start = ((n - crop) // 2 - n // 2) % n
        self._unshifted_mask = self.backend.ifftshift(self._mask)

    def _make_modulation_tilts(self) -> NDArray[Any] | None:
        """Cache modulation phasors, which are immutable instrument geometry."""
        radius = self.settings.modulation_radius_lambda_over_d
        if radius == 0.0:
            return None
        samples = self.settings.modulation_samples
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
        return cast(NDArray[Any], self.backend.asarray(tilts, dtype=self._complex_dtype))

    def _base_phasor(
        self, internal: NDArray[np.float64], state: SourceState, state_index: int
    ) -> NDArray[Any]:
        total_opd = internal + self._field_angle_opd[state_index]
        piston = total_opd[self._piston_index]
        relative_opd = total_opd - piston
        phase = opd_to_phase(relative_opd, state.wavelength_m)
        return cast(
            NDArray[Any],
            self.backend.asarray(self.backend.exp(1j * phase), dtype=self._complex_dtype),
        )

    def _fields(
        self, internal: NDArray[np.float64], state: SourceState, state_index: int
    ) -> NDArray[Any]:
        phasor = self._base_phasor(internal, state, state_index)
        if self._modulated_pupil is None:
            return cast(NDArray[Any], (self.pupil * phasor)[None, ...])
        return cast(
            NDArray[Any],
            self.backend.asarray(
                phasor[None, ...] * self._modulated_pupil, dtype=self._complex_dtype
            ),
        )

    def render(self, wavefront: NDArray[np.float64]) -> OpticalResult:
        internal = self.wavefront.opd(wavefront, target_shape=self.internal_shape)
        outputs = self._propagate(internal)
        photon_rate, captured = outputs[0], outputs[1]
        spectral_photon_rate = outputs[2] if len(outputs) > 2 else photon_rate[None, ...]
        return OpticalResult(
            photon_rate,
            self.source_rate,
            captured,
            self.wavefront.input_opd(wavefront),
            spectral_photon_rate,
            self._wavelengths,
        )

    def _propagate(self, internal: NDArray[np.float64]) -> tuple[Any, ...]:
        """Return the photon rate, captured rate and, if polychromatic, the cube."""
        photon_rate = self.backend.zeros(self.output_shape, dtype=self._rate_dtype)
        spectral_photon_rate = (
            None
            if len(self._wavelengths) == 1
            else self.backend.zeros(
                (len(self._wavelengths), *self.output_shape), dtype=self._rate_dtype
            )
        )
        captured: Any = 0.0
        margin = self.settings.detector_margin_pixels
        for state_index, state in enumerate(self.source_states):
            fields = self._fields(internal, state, state_index)
            # Centred unitary FFT, pyramid mask, centred inverse FFT and crop,
            # evaluated on the unshifted grid (see ``_build_unshifted_geometry``).
            # The transforms skip the zero padding rows and the cropped-away
            # output lines.
            focal = self.backend.pruned_fft2(
                fields,
                size=self.nfft,
                input_start=self._pupil_start,
                workers=self.config.numerics.fft_workers,
            )
            focal *= self._unshifted_mask
            exit_pupil = self.backend.pruned_fft2(
                focal,
                size=self.nfft,
                output_start=self._crop_start,
                output_length=self._base_output_shape[0],
                inverse=True,
                workers=self.config.numerics.fft_workers,
            )
            cropped = self.backend.abs(exit_pupil) ** 2
            mosaic = self.backend.mean(cropped, axis=0)
            if margin:
                padded = self.backend.zeros(self.output_shape, dtype=self._rate_dtype)
                padded[
                    margin : margin + self._base_output_shape[0],
                    margin : margin + self._base_output_shape[1],
                ] = mosaic
                mosaic = padded
            cropped_flux = self.backend.sum(mosaic)
            contribution = mosaic * (self.source_rate * state.weight / self._total_field_flux)
            photon_rate += contribution
            if spectral_photon_rate is not None:
                spectral_photon_rate[self._state_wavelength_indices[state_index]] += contribution
            captured += self.source_rate * state.weight * cropped_flux / self._total_field_flux
        if spectral_photon_rate is None:
            return photon_rate, captured
        return photon_rate, captured, spectral_photon_rate


__all__ = ["PyramidEngine"]
