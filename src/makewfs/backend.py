"""CPU array and FFT primitives.

The module keeps backend-specific operations in one place.  The public release
currently uses NumPy; the small interface makes a CuPy implementation possible
without changing sensor mathematics.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def real_dtype(name: str) -> np.dtype[Any]:
    """Return the configured real dtype."""
    if name == "float32":
        return np.dtype(np.float32)
    if name == "float64":
        return np.dtype(np.float64)
    raise ValueError(f"unsupported dtype {name!r}")


def complex_dtype(name: str) -> np.dtype[Any]:
    """Return the complex dtype paired with a real dtype."""
    return np.dtype(np.complex64 if name == "float32" else np.complex128)


def centered_fft2(array: NDArray[Any], *, workers: int = 1) -> NDArray[Any]:
    """Centered two-dimensional FFT with unitary normalization."""
    # scipy.fft supports worker control and preserves single precision.  The
    # import is local so importing configuration remains lightweight.
    from scipy import fft

    transformed = fft.fftshift(
        fft.fft2(fft.ifftshift(array, axes=(-2, -1)), axes=(-2, -1), workers=workers), axes=(-2, -1)
    )
    height, width = array.shape[-2:]
    return np.asarray(transformed / np.sqrt(height * width))


def centered_ifft2(array: NDArray[Any], *, workers: int = 1) -> NDArray[Any]:
    """Centered two-dimensional inverse FFT with unitary normalization."""
    from scipy import fft

    transformed = fft.fftshift(
        fft.ifft2(fft.ifftshift(array, axes=(-2, -1)), axes=(-2, -1), workers=workers),
        axes=(-2, -1),
    )
    height, width = array.shape[-2:]
    return np.asarray(transformed * np.sqrt(height * width))


def next_fast_length(value: int) -> int:
    """Return a convenient FFT length."""
    from scipy.fft import next_fast_len

    return int(next_fast_len(value))


__all__ = ["centered_fft2", "centered_ifft2", "complex_dtype", "next_fast_length", "real_dtype"]
