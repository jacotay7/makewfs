"""Fixed per-subaperture sodium-LGS convolution kernels (``lgs_model = "kernel"``).

The quadrature LGS model (:mod:`makewfs.source`) propagates one optical state
per source-angle sample and sodium range. Each state adds a tilt across every
subaperture,

```text
delta_theta_jk = (launch - c_j) * (1 / r_k - 1 / r_mean) + alpha_a,
```

which by the Fourier shift theorem only translates that subaperture's
diffraction spot by ``delta_theta_jk``. The incoherent sum over states is
therefore the zero-offset spot convolved with a fixed kernel per subaperture
``j``: the weighted set of translations, blurred by the Gaussian source extent.
For a fixed beacon geometry that kernel never changes, so the kernel mode
propagates only the wavelength states and applies the kernel as one batched
FFT convolution. The same convolution description of an elongated
subaperture spot underlies Schreiber et al. 2009, *MNRAS*
(doi:10.1111/j.1365-2966.2009.14797.x), and Thomas et al. 2008, *MNRAS* 387,
173.

Numerical choices, each documented in the Shack-Hartmann guide:

- The convolution acts on the oversampled focal-plane intensity, before the
  field stop, any focal blur, and pixel-area integration, exactly where the
  quadrature model's translations act.
- The spot is propagated over the detector window plus a margin of up to the
  kernel half-width on each side, limited to one alias-free period of the
  sampled lenslet field. Light translated *into* the window from that margin is
  kept; light translated *out of* the window is lost and reported, never
  renormalized.
- Each translation is deposited by linear (cloud-in-cell) interpolation on the
  oversampled grid, which conserves its weight and first moment exactly and
  keeps the kernel non-negative. The Gaussian extent is sampled on the same
  grid, truncated at four standard deviations and normalized.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from ..backend import ArrayBackend
from ..source import LGSKernelMorphology

_GAUSSIAN_TRUNCATION_SIGMA = 4.0


@dataclass(frozen=True)
class _KernelOffsets:
    """Kernel translations of every subaperture in oversampled detector samples."""

    x: tuple[Any, ...]
    y: tuple[Any, ...]
    weights: tuple[float, ...]


def kernel_offsets(
    centers_x: Any,
    centers_y: Any,
    *,
    launch_m: tuple[float, float],
    mean_range_m: float | None,
    morphology: LGSKernelMorphology,
    rotation_rad: float,
    sample_angle_rad: float,
    backend: ArrayBackend,
) -> _KernelOffsets:
    """Return each subaperture's kernel translations in focal-plane samples.

    Parameters
    ----------
    centers_x, centers_y
        Flattened ``(lenslet_y, lenslet_x)`` subaperture centres in entrance
        pupil metres, the positions the quadrature model uses.
    launch_m
        Launch position ``[x, y]`` in entrance-pupil metres.
    mean_range_m
        Weighted mean sodium range at which the OPD is interpreted, or
        ``None`` for a thin beacon.
    morphology
        Angular points, Gaussian width and normalized range quadrature.
    rotation_rad
        Lenslet-grid rotation. A sky angle in the pupil frame translates the
        spot along the lenslet (detector) axes rotated by minus this angle.
    sample_angle_rad
        On-sky angle of one oversampled focal-plane sample.
    backend
        Array backend holding the centres.

    Returns
    -------
    _KernelOffsets
        One ``x``/``y`` offset array (shape ``(n_subapertures,)``) and weight
        per translation, ordered by angular point then range.
    """
    ranges = morphology.ranges_m or (None,)
    range_weights = morphology.range_weights or (1.0,)
    cosine = math.cos(rotation_rad)
    sine = math.sin(rotation_rad)
    launch_x, launch_y = launch_m
    offsets_x: list[Any] = []
    offsets_y: list[Any] = []
    weights: list[float] = []
    for point_x, point_y, point_weight in morphology.points:
        for range_m, range_weight in zip(ranges, range_weights):
            if range_m is None or mean_range_m is None:
                angle_x = centers_x * 0.0 + point_x
                angle_y = centers_y * 0.0 + point_y
            else:
                range_delta = 1.0 / range_m - 1.0 / mean_range_m
                angle_x = (launch_x - centers_x) * range_delta + point_x
                angle_y = (launch_y - centers_y) * range_delta + point_y
            # A pupil-frame tilt theta over lenslet-frame coordinates (u, v)
            # is u (cos theta_x + sin theta_y) + v (-sin theta_x + cos theta_y).
            offsets_x.append((cosine * angle_x + sine * angle_y) / sample_angle_rad)
            offsets_y.append((-sine * angle_x + cosine * angle_y) / sample_angle_rad)
            weights.append(point_weight * range_weight)
    total = math.fsum(weights)
    return _KernelOffsets(
        tuple(offsets_x),
        tuple(offsets_y),
        tuple(weight / total for weight in weights),
    )


def gaussian_radius_samples(sigma_samples: float) -> int:
    """Return the truncation radius of the sampled Gaussian source extent."""
    if sigma_samples <= 0.0:
        return 0
    return math.ceil(_GAUSSIAN_TRUNCATION_SIGMA * sigma_samples)


def kernel_radius_samples(
    offsets: _KernelOffsets, sigma_samples: float, *, backend: ArrayBackend
) -> int:
    """Return the half-width that holds every translated, blurred kernel sample."""
    largest = 0.0
    for offset_x, offset_y in zip(offsets.x, offsets.y):
        largest = max(
            largest,
            *backend.scalars(backend.abs(offset_x).max(), backend.abs(offset_y).max()),
        )
    # One more sample holds the upper neighbour of the linear deposition.
    return math.ceil(largest) + gaussian_radius_samples(sigma_samples) + 1


def _interpolated_gaussian(
    position: Any, sigma_samples: float, *, backend: ArrayBackend, dtype: Any
) -> Any:
    """Evaluate the linearly interpolated, normalized sampled Gaussian.

    ``position`` holds offsets in samples. For ``sigma_samples == 0`` the
    sampled Gaussian is a unit impulse and this is the linear (cloud-in-cell)
    deposition weight ``max(0, 1 - |position|)``.
    """
    lower = backend.floor(position)
    fraction = position - lower
    if sigma_samples <= 0.0:
        return backend.maximum(1.0 - backend.abs(position), 0.0)
    radius = gaussian_radius_samples(sigma_samples)
    support = backend.arange(-radius, radius + 1, dtype=dtype)
    norm = backend.sum(backend.exp(-0.5 * (support / sigma_samples) ** 2))

    def sampled(index: Any) -> Any:
        value = backend.exp(-0.5 * (index / sigma_samples) ** 2) / norm
        return backend.where(backend.abs(index) <= radius, value, 0.0)

    return (1.0 - fraction) * sampled(lower) + fraction * sampled(lower + 1.0)


@dataclass(frozen=True)
class _LGSKernelPlan:
    """Device-resident per-subaperture kernel spectra for one lenslet geometry."""

    backend: ArrayBackend
    subapertures: int
    window: int
    """Detector-window width in oversampled focal-plane samples."""
    margin: int
    """Extra propagated samples on each side of the window."""
    radius: int
    """Kernel half-width; the stored kernel starts at index 0, not centred."""
    size: int
    """Square FFT length of the circular convolution."""
    kernel_spectrum: Any
    workers: int

    @classmethod
    def build(
        cls,
        offsets: _KernelOffsets,
        *,
        sigma_samples: float,
        radius: int,
        window: int,
        margin: int,
        workers: int,
        complex_dtype: Any,
        backend: ArrayBackend,
    ) -> _LGSKernelPlan:
        """Build the kernel spectra for one geometry.

        ``radius`` must contain every translated sample (see
        :func:`kernel_radius_samples`), so each kernel holds its full unit
        weight.
        """
        extended = window + 2 * margin
        # The kernel is stored from index 0 rather than centred, which delays
        # the convolution by ``radius``; the crop starts that much later. A
        # circular length of ``window + margin + radius`` then keeps every kept
        # output free of wrapped input, and the extended window plus one
        # radius is enough.
        size = backend.next_fast_length(extended + radius)
        grid = backend.arange(-radius, radius + 1, dtype=np.float64)
        subapertures = int(offsets.x[0].shape[0])
        kernel = backend.zeros((subapertures, 2 * radius + 1, 2 * radius + 1), dtype=np.float64)
        for offset_x, offset_y, weight in zip(offsets.x, offsets.y, offsets.weights):
            profile_x = _interpolated_gaussian(
                grid[None, :] - offset_x[:, None],
                sigma_samples,
                backend=backend,
                dtype=np.float64,
            )
            profile_y = _interpolated_gaussian(
                grid[None, :] - offset_y[:, None],
                sigma_samples,
                backend=backend,
                dtype=np.float64,
            )
            kernel += weight * profile_y[:, :, None] * profile_x[:, None, :]
        real = np.float32 if np.dtype(complex_dtype) == np.complex64 else np.float64
        spectrum = backend.asarray(
            backend.padded_convolution_spectrum(
                backend.asarray(kernel, dtype=real), size=size, workers=workers
            ),
            dtype=complex_dtype,
        )
        return cls(backend, subapertures, window, margin, radius, size, spectrum, workers)

    def apply(self, intensity: Any) -> Any:
        """Convolve a ``(batch * subapertures, extended, extended)`` stack, crop the window.

        The leading axis is ordered with the subaperture index fastest, as the
        Shack-Hartmann engine batches lenslets. Light convolved outside the
        detector window is discarded. FFT rounding can leave values of order
        ``1e-16`` of the peak below zero; they are clipped so the intensity
        stays non-negative.
        """
        backend = self.backend
        extended = self.window + 2 * self.margin
        batch = intensity.reshape(-1, self.subapertures, extended, extended)
        spectrum = backend.padded_convolution_spectrum(batch, size=self.size, workers=self.workers)
        window = backend.cropped_convolution(
            spectrum * self.kernel_spectrum[None, ...],
            size=self.size,
            start=self.margin + self.radius,
            length=self.window,
            workers=self.workers,
        )
        clipped = backend.maximum(window, 0.0)
        return cast(
            Any,
            backend.asarray(
                clipped.reshape(-1, self.window, self.window),
                dtype=intensity.dtype,
            ),
        )


__all__ = ["gaussian_radius_samples", "kernel_offsets", "kernel_radius_samples"]
