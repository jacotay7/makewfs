"""Array and FFT primitives used by the portable optical kernels.

The CPU release uses :mod:`numpy` arrays and :mod:`scipy.fft`, but sensor
mathematics calls this small backend object rather than allocating through
NumPy directly.  That boundary is deliberately private today; it gives a
future CuPy implementation one place to provide array creation, reductions,
FFT, and interpolation semantics without changing the optical equations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ArrayBackend:
    """Numerical namespace for one optical array backend.

    ``xp`` is an Array API-compatible namespace.  The CPU instance is the only
    supported instance in the public release; a future device backend may
    provide the corresponding namespace and override the SciPy-only helpers.
    Methods named ``scalar`` and ``to_host`` are explicit host-boundary points
    for metadata and configuration diagnostics, rather than accidental scalar
    extraction in a sensor kernel.
    """

    xp: Any
    name: str = "cpu"

    @property
    def is_cpu(self) -> bool:
        """Whether this backend uses host NumPy/SciPy arrays."""
        return self.name == "cpu"

    def asarray(self, value: Any, *, dtype: Any | None = None) -> Any:
        """Convert a value using this backend's array namespace."""
        return self.xp.asarray(value, dtype=dtype)

    def zeros(self, shape: Any, *, dtype: Any) -> Any:
        """Allocate a zero-filled array on this backend."""
        return self.xp.zeros(shape, dtype=dtype)

    def zeros_like(self, value: Any) -> Any:
        """Allocate an array matching ``value`` on this backend."""
        return self.xp.zeros_like(value)

    def astype(self, value: Any, dtype: Any) -> Any:
        """Cast an array without changing its backend."""
        return value.astype(dtype)

    def full(self, shape: Any, value: Any, *, dtype: Any) -> Any:
        """Allocate a constant-filled array on this backend."""
        return self.xp.full(shape, value, dtype=dtype)

    def empty(self, shape: Any, *, dtype: Any) -> Any:
        """Allocate an uninitialized array on this backend."""
        return self.xp.empty(shape, dtype=dtype)

    def arange(self, *args: Any, **kwargs: Any) -> Any:
        """Create a backend array of evenly spaced values."""
        return self.xp.arange(*args, **kwargs)

    def meshgrid(self, *args: Any, **kwargs: Any) -> Any:
        """Create backend coordinate grids."""
        return self.xp.meshgrid(*args, **kwargs)

    def repeat(self, value: Any, repeats: Any, *, axis: int) -> Any:
        """Repeat values along one backend axis."""
        return self.xp.repeat(value, repeats, axis=axis)

    def tile(self, value: Any, reps: Any) -> Any:
        """Tile a backend array."""
        return self.xp.tile(value, reps)

    def stack(self, values: Any, *, axis: int = 0) -> Any:
        """Stack backend arrays."""
        return self.xp.stack(values, axis=axis)

    def sum(self, value: Any, *, axis: Any = None) -> Any:
        """Reduce a backend array by summation."""
        return self.xp.sum(value, axis=axis)

    def mean(self, value: Any, *, axis: Any = None) -> Any:
        """Reduce a backend array by mean."""
        return self.xp.mean(value, axis=axis)

    def average(self, value: Any, *, weights: Any) -> Any:
        """Compute a weighted backend average."""
        return self.xp.average(value, weights=weights)

    def any(self, value: Any) -> Any:
        """Backend reduction testing whether any element is true."""
        return self.xp.any(value)

    def all(self, value: Any) -> Any:
        """Backend reduction testing whether all elements are true."""
        return self.xp.all(value)

    def isfinite(self, value: Any) -> Any:
        """Elementwise finite-value test."""
        return self.xp.isfinite(value)

    def abs(self, value: Any) -> Any:
        """Elementwise absolute value."""
        return self.xp.abs(value)

    def exp(self, value: Any) -> Any:
        """Elementwise exponential."""
        return self.xp.exp(value)

    def sqrt(self, value: Any) -> Any:
        """Elementwise square root."""
        return self.xp.sqrt(value)

    def hypot(self, left: Any, right: Any) -> Any:
        """Elementwise Euclidean norm."""
        return self.xp.hypot(left, right)

    def cos(self, value: Any) -> Any:
        """Elementwise cosine."""
        return self.xp.cos(value)

    def sin(self, value: Any) -> Any:
        """Elementwise sine."""
        return self.xp.sin(value)

    def arctan2(self, left: Any, right: Any) -> Any:
        """Elementwise two-argument arctangent."""
        return self.xp.arctan2(left, right)

    def mod(self, value: Any, divisor: Any) -> Any:
        """Elementwise remainder."""
        return self.xp.mod(value, divisor)

    def where(self, condition: Any, left: Any, right: Any) -> Any:
        """Select values elementwise on the backend."""
        return self.xp.where(condition, left, right)

    def ptp(self, value: Any) -> Any:
        """Backend peak-to-peak reduction."""
        return self.xp.ptp(value)

    def fftfreq(self, value: int) -> Any:
        """Return backend FFT frequency bins."""
        return self.xp.fft.fftfreq(value)

    def fftshift(self, value: Any, *, axes: Any = None) -> Any:
        """Shift zero frequency to the center."""
        return self.xp.fft.fftshift(value, axes=axes)

    def ifftshift(self, value: Any, *, axes: Any = None) -> Any:
        """Undo a centered FFT shift."""
        return self.xp.fft.ifftshift(value, axes=axes)

    def centered_fft2(self, array: Any, *, workers: int = 1) -> Any:
        """Perform a centered, unitary two-dimensional FFT."""
        axes = (-2, -1)
        if self.is_cpu:
            from scipy import fft

            transformed = fft.fftshift(
                fft.fft2(
                    fft.ifftshift(array, axes=axes),
                    axes=axes,
                    workers=workers,
                ),
                axes=axes,
            )
        else:  # pragma: no cover - reserved for a future device backend
            transformed = self.fftshift(
                self.xp.fft.fft2(self.ifftshift(array, axes=axes), axes=axes),
                axes=axes,
            )
        height, width = array.shape[-2:]
        scale = self.sqrt(self.asarray(height * width, dtype=transformed.real.dtype))
        return transformed / scale

    def centered_ifft2(self, array: Any, *, workers: int = 1) -> Any:
        """Perform a centered, unitary two-dimensional inverse FFT."""
        axes = (-2, -1)
        if self.is_cpu:
            from scipy import fft

            transformed = fft.fftshift(
                fft.ifft2(
                    fft.ifftshift(array, axes=axes),
                    axes=axes,
                    workers=workers,
                ),
                axes=axes,
            )
        else:  # pragma: no cover - reserved for a future device backend
            transformed = self.fftshift(
                self.xp.fft.ifft2(self.ifftshift(array, axes=axes), axes=axes),
                axes=axes,
            )
        height, width = array.shape[-2:]
        scale = self.sqrt(self.asarray(height * width, dtype=transformed.real.dtype))
        return transformed * scale

    def map_coordinates(self, array: Any, coordinates: Any, *, order: int, mode: str) -> Any:
        """Interpolate coordinates, using SciPy only for the CPU backend."""
        if not self.is_cpu:  # pragma: no cover - reserved for a future device backend
            raise NotImplementedError("map_coordinates is not implemented for this backend")
        from scipy.ndimage import map_coordinates

        return map_coordinates(array, coordinates, order=order, mode=mode)

    def convolve(self, array: Any, kernel: Any) -> Any:
        """Convolve a batch of arrays with a backend-compatible kernel."""
        if not self.is_cpu:  # pragma: no cover - reserved for a future device backend
            raise NotImplementedError("measured blur is not implemented for this backend")
        from scipy.ndimage import convolve

        return convolve(array, kernel, mode="constant", cval=0.0)

    def gaussian_filter(self, array: Any, sigma: Any) -> Any:
        """Apply a Gaussian filter through the CPU numerical backend."""
        if not self.is_cpu:  # pragma: no cover - reserved for a future device backend
            raise NotImplementedError("Gaussian blur is not implemented for this backend")
        from scipy.ndimage import gaussian_filter

        return gaussian_filter(array, sigma=sigma, mode="constant")

    def next_fast_length(self, value: int) -> int:
        """Return an FFT-friendly length using this backend's implementation."""
        if not self.is_cpu:  # pragma: no cover - reserved for a future device backend
            raise NotImplementedError("next_fast_length is not implemented for this backend")
        from scipy.fft import next_fast_len

        return int(next_fast_len(value))

    def scalar(self, value: Any) -> float:
        """Extract one host scalar at an explicit metadata/geometry boundary."""
        item = value.item() if hasattr(value, "item") else value
        return float(item)

    def to_host(self, value: Any) -> NDArray[Any]:
        """Copy an array to host NumPy storage at an explicit boundary."""
        if self.is_cpu:
            return cast(NDArray[Any], value)
        return np.asarray(value)


_CPU_BACKEND = ArrayBackend(np, name="cpu")


def cpu_backend() -> ArrayBackend:
    """Return the shared CPU backend instance."""
    return _CPU_BACKEND


def real_dtype(name: str) -> np.dtype[Any]:
    """Return the configured real dtype."""
    if name == "float32":
        return np.dtype(np.float32)
    if name == "float64":
        return np.dtype(np.float64)
    raise ValueError(f"unsupported dtype {name!r}")


def complex_dtype(name: str) -> np.dtype[Any]:
    """Return the complex dtype paired with a real dtype."""
    if name == "float32":
        return np.dtype(np.complex64)
    if name == "float64":
        return np.dtype(np.complex128)
    raise ValueError(f"unsupported dtype {name!r}")


def centered_fft2(
    array: NDArray[Any], *, workers: int = 1, backend: ArrayBackend | None = None
) -> NDArray[Any]:
    """Centered two-dimensional FFT with unitary normalization."""
    return cast(NDArray[Any], (backend or cpu_backend()).centered_fft2(array, workers=workers))


def centered_ifft2(
    array: NDArray[Any], *, workers: int = 1, backend: ArrayBackend | None = None
) -> NDArray[Any]:
    """Centered two-dimensional inverse FFT with unitary normalization."""
    return cast(NDArray[Any], (backend or cpu_backend()).centered_ifft2(array, workers=workers))


def next_fast_length(value: int) -> int:
    """Return a convenient CPU FFT length."""
    return cpu_backend().next_fast_length(value)


__all__ = [
    "ArrayBackend",
    "centered_fft2",
    "centered_ifft2",
    "complex_dtype",
    "cpu_backend",
    "next_fast_length",
    "real_dtype",
]
