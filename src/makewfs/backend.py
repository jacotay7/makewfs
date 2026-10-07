"""Array and FFT primitives used by the portable optical kernels.

The CPU release uses :mod:`numpy` arrays and :mod:`scipy.fft`, but sensor
mathematics calls this small backend object rather than allocating through
NumPy directly.  That boundary is deliberately private today; it gives a
future CuPy implementation one place to provide array creation, reductions,
FFT, and interpolation semantics without changing the optical equations.

It is deliberately separate from ``aocore.Backend``. The sensors choose a dtype
per array rather than one precision per backend, pass explicit FFT worker
counts and ``overwrite_x``, and need SciPy/CuPy ``ndimage`` helpers. The
centred FFTs here use the ``fftshift`` convention (zero frequency on pixel
``n // 2``). The engines own the conversion to CONVENTIONS 1.3 centring: for
example ``sampling.spot_intensity`` applies a half-sample phase ramp for even
detectors, and ``tests/test_conformance.py`` checks both sensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from aocore import centered_coordinates as _aocore_centered_coordinates
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

    def centered_coordinates(self, n: int, *, dtype: Any) -> Any:
        """Unit-pitch pixel-centre coordinates (CONVENTIONS 1.2) on this device.

        ``aocore.centered_coordinates`` builds them in float64 and casts to
        ``dtype`` directly on the GPU, so no host array is copied over. The
        CPU path still passes through :meth:`asarray`, which keeps an injected
        namespace in the loop.
        """
        if self.is_cpu:
            return self.asarray(_aocore_centered_coordinates(n, dtype=dtype), dtype=dtype)
        return _aocore_centered_coordinates(n, dtype=dtype, backend="gpu")

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

    def matmul(self, left: Any, right: Any) -> Any:
        """Multiply backend arrays with batched broadcasting."""
        return self.xp.matmul(left, right)

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

    def argmax(self, value: Any) -> Any:
        """Return the flat index of a backend array maximum."""
        return self.xp.argmax(value)

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
                    norm="ortho",
                    overwrite_x=True,
                ),
                axes=axes,
            )
        else:  # pragma: no cover - reserved for a future device backend
            transformed = self.fftshift(
                self.xp.fft.fft2(self.ifftshift(array, axes=axes), axes=axes, norm="ortho"),
                axes=axes,
            )
        return transformed

    def centered_fft_intensity(
        self, array: Any, *, workers: int = 1, overwrite_input: bool = False
    ) -> Any:
        """Return centered unitary FFT intensity without an irrelevant input roll.

        Translating an entrance field changes only Fourier phase, so an input
        ``ifftshift`` cannot affect intensity. Shack-Hartmann propagation uses
        this identity to avoid one detector-batch-sized array permutation.
        """
        axes = (-2, -1)
        height, width = array.shape[-2:]
        if height % 2 == 0 and width % 2 == 0:
            working = array if overwrite_input else self.xp.array(array, copy=True)
            working[..., ::2, 1::2] *= -1
            working[..., 1::2, ::2] *= -1
            if self.is_cpu:
                from scipy import fft

                transformed = fft.fft2(
                    working,
                    axes=axes,
                    workers=workers,
                    norm="ortho",
                    overwrite_x=overwrite_input,
                )
            else:  # pragma: no cover - GPU optional
                transformed = self.xp.fft.fft2(working, axes=axes, norm="ortho")
            return self.abs(transformed) ** 2
        if self.is_cpu:
            from scipy import fft

            transformed = fft.fftshift(
                fft.fft2(array, axes=axes, workers=workers, norm="ortho"),
                axes=axes,
            )
        else:  # pragma: no cover - GPU optional
            transformed = self.fftshift(self.xp.fft.fft2(array, axes=axes, norm="ortho"), axes=axes)
        return self.abs(transformed) ** 2

    def fft_axis(
        self,
        array: Any,
        *,
        axis: int,
        norm: str,
        inverse: bool = False,
        workers: int = 1,
    ) -> Any:
        """One-dimensional FFT along ``axis`` of an array the caller owns.

        The pruned two-dimensional transforms are built from these passes. On
        the CPU the input may be overwritten, so pass only a temporary.
        """
        if self.is_cpu:
            from scipy import fft

            transform = fft.ifft if inverse else fft.fft
            return transform(array, axis=axis, norm=norm, workers=workers, overwrite_x=True)
        transform = self.xp.fft.ifft if inverse else self.xp.fft.fft
        return transform(array, axis=axis, norm=norm)

    def pruned_fft2(
        self,
        array: Any,
        *,
        size: int,
        input_start: int | None = None,
        output_start: int | None = None,
        output_length: int | None = None,
        inverse: bool = False,
        axes: tuple[int, int] = (-2, -1),
        workers: int = 1,
    ) -> Any:
        """Unitary square 2-D FFT that skips zero input and unwanted output lines.

        ``array`` holds the ``(..., k, k)`` block that occupies grid rows and
        columns ``input_start, ..., input_start + k - 1`` (modulo ``size``) of
        an otherwise zero ``(size, size)`` grid, or the whole grid when
        ``input_start`` is ``None`` (it may then be overwritten). Only the
        ``output_length`` rows and columns from ``output_start`` (again modulo
        ``size``) are returned, or all of them when ``output_start`` is
        ``None``. Positions are on the unshifted grid. The result is
        ``fft2(grid, norm="ortho")`` (``ifft2`` when ``inverse``) restricted to
        those lines.

        A 2-D FFT is a pass of 1-D transforms along ``axes[0]`` followed by a
        pass along ``axes[1]``. A line that is entirely zero transforms to zero,
        so the first pass runs only over the ``k`` lines that hold data; output
        lines that are cropped away are never needed, so the second pass runs
        only over the wanted ones. The whole ``1 / size`` unitary factor is
        applied in the first pass, as SciPy's own 2-D transform does. Each
        computed value is that of the same two passes over the whole grid; it is
        bit-identical whenever the FFT library treats each line alike (SciPy
        transforms lines in SIMD groups and a short remainder group can round
        differently, so a different line count may move a value by an ulp).

        Keep ``axes`` equal to the pass order of the transform being replaced,
        because rounding depends on it: SciPy's ``fft2`` runs the listed axes in
        order when ``overwrite_x=True`` but the last axis first when it
        allocates its output.
        """
        if (output_start is None) != (output_length is None):
            raise ValueError("output_start and output_length go together")
        first_axis, second_axis = axes
        first_norm, second_norm = ("backward", "forward") if inverse else ("forward", "backward")
        spectrum = array
        if input_start is not None:
            spectrum = self._embed(spectrum, input_start, size=size, axis=first_axis)
        spectrum = self.fft_axis(
            spectrum, axis=first_axis, norm=first_norm, inverse=inverse, workers=workers
        )
        if output_start is not None:
            assert output_length is not None
            spectrum = self._gather(spectrum, output_start, output_length, axis=first_axis)
        if input_start is not None:
            spectrum = self._embed(spectrum, input_start, size=size, axis=second_axis)
        spectrum = self.fft_axis(
            spectrum, axis=second_axis, norm=second_norm, inverse=inverse, workers=workers
        )
        if output_start is not None:
            assert output_length is not None
            spectrum = self._gather(spectrum, output_start, output_length, axis=second_axis)
        return spectrum

    def _embed(self, array: Any, start: int, *, size: int, axis: int) -> Any:
        """Zero-extend ``axis`` to ``size``, placing ``array`` from ``start`` with wrap."""
        axis = axis % array.ndim
        shape = list(array.shape)
        shape[axis] = size
        result = self.zeros(tuple(shape), dtype=array.dtype)
        lead = (slice(None),) * axis
        for grid, block in _wrapped_segments(start, array.shape[axis], size):
            result[(*lead, grid)] = array[(*lead, block)]
        return result

    def _gather(self, array: Any, start: int, length: int, *, axis: int) -> Any:
        """Select ``length`` entries of ``axis`` from ``start``, wrapping around."""
        axis = axis % array.ndim
        lead = (slice(None),) * axis
        parts = [
            array[(*lead, grid)] for grid, _ in _wrapped_segments(start, length, array.shape[axis])
        ]
        return parts[0] if len(parts) == 1 else self.xp.concatenate(parts, axis=axis)

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
                    norm="ortho",
                    overwrite_x=True,
                ),
                axes=axes,
            )
        else:  # pragma: no cover - reserved for a future device backend
            transformed = self.fftshift(
                self.xp.fft.ifft2(self.ifftshift(array, axes=axes), axes=axes, norm="ortho"),
                axes=axes,
            )
        return transformed

    def map_coordinates(self, array: Any, coordinates: Any, *, order: int, mode: str) -> Any:
        """Interpolate coordinates, using SciPy only for the CPU backend."""
        if self.is_cpu:
            from scipy.ndimage import map_coordinates

            return map_coordinates(array, coordinates, order=order, mode=mode)
        from cupyx.scipy.ndimage import map_coordinates  # pragma: no cover - GPU optional

        device_coordinates = (
            self.xp.stack(coordinates, axis=0)
            if isinstance(coordinates, (list, tuple))
            else coordinates
        )
        return map_coordinates(array, device_coordinates, order=order, mode=mode)

    def convolve(self, array: Any, kernel: Any) -> Any:
        """Convolve a batch of arrays with a backend-compatible kernel."""
        if self.is_cpu:
            from scipy.ndimage import convolve

            return convolve(array, kernel, mode="constant", cval=0.0)
        from cupyx.scipy.ndimage import convolve  # pragma: no cover - GPU optional

        return convolve(array, kernel, mode="constant", cval=0.0)

    def gaussian_filter(self, array: Any, sigma: Any) -> Any:
        """Apply a Gaussian filter through the CPU numerical backend."""
        if self.is_cpu:
            from scipy.ndimage import gaussian_filter

            return gaussian_filter(array, sigma=sigma, mode="constant")
        from cupyx.scipy.ndimage import gaussian_filter  # pragma: no cover - GPU optional

        return gaussian_filter(array, sigma=sigma, mode="constant")

    def next_fast_length(self, value: int) -> int:
        """Return an FFT-friendly length using this backend's implementation."""
        if self.is_cpu:
            from scipy.fft import next_fast_len
        else:  # pragma: no cover - GPU optional
            from cupyx.scipy.fft import next_fast_len

        return int(next_fast_len(value))

    def scalar(self, value: Any) -> float:
        """Extract one host scalar at an explicit metadata/geometry boundary."""
        item = value.item() if hasattr(value, "item") else value
        return float(item)

    def scalars(self, *values: Any) -> tuple[float, ...]:
        """Extract several scalars with one device synchronization."""
        if self.is_cpu:
            return tuple(self.scalar(value) for value in values)
        packed = self.xp.stack([self.xp.asarray(value) for value in values])
        return tuple(float(value) for value in self.xp.asnumpy(packed))

    def to_host(self, value: Any) -> NDArray[Any]:
        """Copy an array to host NumPy storage at an explicit boundary."""
        if self.is_cpu:
            return cast(NDArray[Any], value)
        return np.asarray(self.xp.asnumpy(value))


def _wrapped_segments(start: int, length: int, size: int) -> list[tuple[slice, slice]]:
    """Split ``length`` positions from ``start`` modulo ``size`` into slices.

    Returns ``(grid, block)`` pairs: grid positions and the matching positions
    in a block that lists the wrapped range in order. There are at most two.
    """
    start %= size
    head = min(length, size - start)
    segments = [(slice(start, start + head), slice(0, head))]
    if head < length:
        segments.append((slice(0, length - head), slice(head, length)))
    return segments


_CPU_BACKEND = ArrayBackend(np, name="cpu")


def cpu_backend() -> ArrayBackend:
    """Return the shared CPU backend instance."""
    return _CPU_BACKEND


def cupy_backend() -> ArrayBackend:
    """Return the private optional CuPy backend.

    CuPy is intentionally imported lazily so the core package remains usable
    without CUDA.  Callers should treat this as experimental and keep the
    explicit host transfer before the ``getframes`` detector adapter.
    """
    try:
        import cupy
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise ImportError("the private CuPy backend requires makewfs[gpu]") from exc
    return ArrayBackend(cupy, name="cupy")


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


def centered_fft_intensity(
    array: NDArray[Any],
    *,
    workers: int = 1,
    backend: ArrayBackend | None = None,
    overwrite_input: bool = False,
) -> NDArray[Any]:
    """Return centered unitary FFT intensity without shifting the input field."""
    return cast(
        NDArray[Any],
        (backend or cpu_backend()).centered_fft_intensity(
            array, workers=workers, overwrite_input=overwrite_input
        ),
    )


def next_fast_length(value: int) -> int:
    """Return a convenient CPU FFT length."""
    return cpu_backend().next_fast_length(value)


__all__ = [
    "ArrayBackend",
    "centered_fft2",
    "centered_fft_intensity",
    "centered_ifft2",
    "complex_dtype",
    "cpu_backend",
    "cupy_backend",
    "next_fast_length",
    "real_dtype",
]
