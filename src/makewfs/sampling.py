"""Flux-preserving sampling helpers shared by sensor engines."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .backend import centered_fft2, next_fast_length


def pad_center(array: NDArray[Any], shape: tuple[int, int]) -> NDArray[Any]:
    """Zero-pad a batch of 2-D arrays around their Fourier centre."""
    old_h, old_w = array.shape[-2:]
    new_h, new_w = shape
    if new_h < old_h or new_w < old_w:
        raise ValueError("pad_center cannot crop")
    result = np.zeros(array.shape[:-2] + shape, dtype=array.dtype)
    y0 = (new_h - old_h) // 2
    x0 = (new_w - old_w) // 2
    result[..., y0 : y0 + old_h, x0 : x0 + old_w] = array
    return result


def crop_center(array: NDArray[Any], shape: tuple[int, int]) -> NDArray[Any]:
    """Crop a batch of 2-D arrays around its Fourier centre."""
    old_h, old_w = array.shape[-2:]
    new_h, new_w = shape
    if new_h > old_h or new_w > old_w:
        raise ValueError("crop_center cannot enlarge")
    y0 = (old_h - new_h) // 2
    x0 = (old_w - new_w) // 2
    return array[..., y0 : y0 + new_h, x0 : x0 + new_w]


def block_sum(array: NDArray[Any], factor: int) -> NDArray[Any]:
    """Sum square pixel blocks while preserving total flux."""
    if factor < 1:
        raise ValueError("factor must be positive")
    height, width = array.shape[-2:]
    if height % factor or width % factor:
        raise ValueError(f"shape {array.shape[-2:]} is not divisible by factor {factor}")
    reshaped = array.reshape((*array.shape[:-2], height // factor, factor, width // factor, factor))
    return np.asarray(reshaped.sum(axis=(-1, -3)))


def spot_intensity(
    field: NDArray[Any],
    *,
    pixels: int,
    samples_per_lenslet: int,
    sampling: float,
    oversampling: int,
    workers: int,
) -> NDArray[np.float64]:
    """Propagate lenslet fields and integrate onto ``pixels`` detector pixels.

    The Fourier pixel scale is ``lambda / D_subap / sampling``.  Zero padding
    by ``oversampling`` improves pixel-area integration while the final block
    sum returns the configured native-pixel grid.
    """
    # The requested native-pixel window must fit even when a designer chooses
    # a large detector pixel scale.  The previous implementation only sized
    # the FFT from the optical sampling and could fail while cropping a valid
    # configuration.
    nfft = next_fast_length(
        max(
            samples_per_lenslet,
            int(np.ceil(samples_per_lenslet * sampling * oversampling)),
            pixels * oversampling,
        )
    )
    transformed = centered_fft2(pad_center(field, (nfft, nfft)), workers=workers)
    intensity = np.abs(transformed) ** 2
    high_resolution_pixels = pixels * oversampling
    cropped = crop_center(intensity, (high_resolution_pixels, high_resolution_pixels))
    return np.asarray(block_sum(cropped, oversampling), dtype=np.float64)


__all__ = ["block_sum", "crop_center", "pad_center", "spot_intensity"]
