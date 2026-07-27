"""Flux-preserving sampling helpers shared by sensor engines."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from .backend import ArrayBackend, centered_fft_intensity, cpu_backend, next_fast_length


def load_blur_kernel(path: str) -> NDArray[np.float64]:
    """Load and normalize a finite, odd-sized measured optical blur kernel."""
    source = Path(path)
    if source.suffix.lower() == ".npy":
        array = np.load(source)
    elif source.suffix.lower() == ".npz":
        archive = np.load(source)
        if not archive.files:
            raise ValueError(f"optical blur archive {source} contains no arrays")
        array = archive[archive.files[0]]
    elif source.suffix.lower() in {".fits", ".fit"}:
        try:
            from astropy.io import fits  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover - optional file format
            raise ImportError("reading FITS blur kernels requires astropy") from exc
        array = fits.getdata(source)
    else:
        raise ValueError("optical blur kernels support .npy, .npz, or FITS")
    kernel = np.asarray(array, dtype=np.float64)
    if (
        kernel.ndim != 2
        or kernel.shape[0] % 2 == 0
        or kernel.shape[1] % 2 == 0
        or not np.all(np.isfinite(kernel))
        or np.any(kernel < 0)
        or np.sum(kernel) <= 0
    ):
        raise ValueError("optical blur kernel must be finite, non-negative, 2-D, and odd-sized")
    return np.asarray(kernel / np.sum(kernel), dtype=np.float64)


def pad_center(
    array: NDArray[Any], shape: tuple[int, int], *, backend: ArrayBackend | None = None
) -> NDArray[Any]:
    """Zero-pad a batch of 2-D arrays around their Fourier centre."""
    old_h, old_w = array.shape[-2:]
    new_h, new_w = shape
    if new_h < old_h or new_w < old_w:
        raise ValueError("pad_center cannot crop")
    resolved = backend or cpu_backend()
    result = resolved.zeros(array.shape[:-2] + shape, dtype=array.dtype)
    y0 = (new_h - old_h) // 2
    x0 = (new_w - old_w) // 2
    result[..., y0 : y0 + old_h, x0 : x0 + old_w] = array
    return cast(NDArray[Any], result)


def crop_center(array: NDArray[Any], shape: tuple[int, int]) -> NDArray[Any]:
    """Crop a batch of 2-D arrays around its Fourier centre."""
    old_h, old_w = array.shape[-2:]
    new_h, new_w = shape
    if new_h > old_h or new_w > old_w:
        raise ValueError("crop_center cannot enlarge")
    y0 = (old_h - new_h) // 2
    x0 = (old_w - new_w) // 2
    return array[..., y0 : y0 + new_h, x0 : x0 + new_w]


def block_sum(
    array: NDArray[Any], factor: int, *, backend: ArrayBackend | None = None
) -> NDArray[Any]:
    """Sum square pixel blocks while preserving total flux."""
    if factor < 1:
        raise ValueError("factor must be positive")
    if factor == 1:
        return array
    height, width = array.shape[-2:]
    if height % factor or width % factor:
        raise ValueError(f"shape {array.shape[-2:]} is not divisible by factor {factor}")
    if factor == 2:
        # Two-times oversampling is the common SH path. Direct strided sums
        # avoid NumPy's disproportionately expensive multi-axis reduction over
        # thousands of tiny spot images and work unchanged with CuPy arrays.
        return cast(
            NDArray[Any],
            array[..., 0::2, 0::2]
            + array[..., 0::2, 1::2]
            + array[..., 1::2, 0::2]
            + array[..., 1::2, 1::2],
        )
    resolved = backend or cpu_backend()
    reshaped = array.reshape((*array.shape[:-2], height // factor, factor, width // factor, factor))
    reduced_x = resolved.sum(reshaped, axis=-1)
    return cast(NDArray[Any], resolved.sum(reduced_x, axis=-2))


def spot_intensity(
    field: NDArray[Any],
    *,
    pixels: int,
    samples_per_lenslet: int,
    sampling: float,
    oversampling: int,
    workers: int,
    field_stop_radius_lambda_over_d: float | None = None,
    optical_blur_fwhm_pixels: float = 0.0,
    optical_blur_kernel: NDArray[np.float64] | None = None,
    backend: ArrayBackend | None = None,
) -> NDArray[Any]:
    """Propagate lenslet fields and integrate onto ``pixels`` detector pixels.

    The Fourier pixel scale is ``lambda / D_subap / sampling``.  Zero padding
    by ``oversampling`` improves pixel-area integration while the final block
    sum returns the configured native-pixel grid.
    """
    # The requested native-pixel window must fit even when a designer chooses
    # a large detector pixel scale.  The previous implementation only sized
    # the FFT from the optical sampling and could fail while cropping a valid
    # configuration.
    resolved = backend or cpu_backend()
    nfft = next_fast_length(
        max(
            samples_per_lenslet,
            math.ceil(samples_per_lenslet * sampling * oversampling),
            pixels * oversampling,
        )
    )
    padded = pad_center(field, (nfft, nfft), backend=resolved)
    high_resolution_pixels = pixels * oversampling
    if high_resolution_pixels % 2 == 0:
        # An even detector has its optical axis on the boundary shared by its
        # central four pixels. Evaluate the Fourier transform at half-integer
        # frequency samples so equal-area integration is exactly symmetric
        # around that boundary. The pupil-plane phase ramp performs the
        # half-sample Fourier shift without interpolating intensity or changing
        # flux. Its sign only selects the equivalent half-pixel sampling branch.
        coordinate = resolved.arange(nfft, dtype=np.float64)
        half_sample = resolved.exp(-1j * math.pi * coordinate / nfft)
        padded *= half_sample[None, :, None] * half_sample[None, None, :]
    intensity = centered_fft_intensity(
        padded,
        workers=workers,
        backend=resolved,
        overwrite_input=True,
    )
    # ``fftshift`` puts zero frequency at ``nfft // 2``. This start index is
    # symmetric for odd grids; even grids become symmetric after the half-sample
    # evaluation above.
    crop_start = nfft // 2 - high_resolution_pixels // 2
    crop_stop = crop_start + high_resolution_pixels
    cropped = intensity[..., crop_start:crop_stop, crop_start:crop_stop]
    if field_stop_radius_lambda_over_d is not None:
        coordinates = resolved.arange(high_resolution_pixels, dtype=np.float64)
        y, x = resolved.meshgrid(coordinates, coordinates, indexing="ij")
        radius_lambda_over_d = resolved.hypot(
            x - (high_resolution_pixels - 1) / 2.0,
            y - (high_resolution_pixels - 1) / 2.0,
        ) / (oversampling * sampling)
        cropped = cropped * (radius_lambda_over_d <= field_stop_radius_lambda_over_d)
    native = block_sum(cropped, oversampling, backend=resolved)
    if optical_blur_kernel is not None and optical_blur_fwhm_pixels > 0.0:
        raise ValueError("provide either optical_blur_fwhm_pixels or optical_blur_kernel")
    if optical_blur_kernel is not None:
        native = resolved.convolve(native, resolved.asarray(optical_blur_kernel)[None, ...])
    elif optical_blur_fwhm_pixels > 0.0:
        sigma = optical_blur_fwhm_pixels / 2.3548200450309493
        native = resolved.gaussian_filter(native, sigma=(0.0, sigma, sigma))
    # Keep the precision produced by the complex FFT through pixel integration.
    # Sensor-level accumulation intentionally converts to the configured photon
    # rate dtype, but this avoids promoting the large spot batch prematurely.
    return native


__all__ = ["block_sum", "crop_center", "load_blur_kernel", "pad_center", "spot_intensity"]
