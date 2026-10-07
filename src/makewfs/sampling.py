"""Flux-preserving sampling helpers shared by sensor engines."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from aocore import block_sum as _aocore_block_sum
from numpy.typing import NDArray

from .backend import ArrayBackend, cpu_backend


def spot_sampling_geometry(
    *,
    pixels: int,
    samples_per_lenslet: int,
    sampling: float,
    oversampling: int,
) -> tuple[str, int | float]:
    """Return the exact propagation geometry for one detector sampling.

    An FFT is exact only when its integer grid simultaneously represents the
    requested ``pixels / (lambda / D)`` sampling and contains the requested
    detector window. Arbitrary or undersampled geometries use a sampled DFT at
    detector-cell quadrature points instead of rounding the physical scale.
    """
    high_resolution_pixels = pixels * oversampling
    ideal_nfft = samples_per_lenslet * sampling * oversampling
    integer_nfft = round(ideal_nfft)
    if (
        math.isclose(ideal_nfft, integer_nfft, rel_tol=0.0, abs_tol=1e-12)
        and integer_nfft >= samples_per_lenslet
        and integer_nfft >= high_resolution_pixels
    ):
        return ("fft", int(integer_nfft))
    return ("dft", float(sampling))


def lenslet_field_upsampling(
    *,
    pupil_samples_per_lenslet: int,
    window_lambda_over_d: float,
) -> int:
    """Return the smallest integer field refinement that avoids spot aliasing.

    A lenslet field sampled at ``s`` points across the lenslet pitch ``d`` has a
    Fraunhofer transform that is periodic every ``s`` lenslet ``lambda / d``:
    the sampled kernel ``exp(-2 pi i u m / s)`` repeats when ``u`` advances by
    ``s``. Integrating a detector window wider than one period therefore sums
    replicas of the spot as if they were real light, which creates flux, and a
    tilt beyond ``+-s/2`` aliases because the phase step between samples
    exceeds pi. The window, ``pixels / sampling`` lenslet ``lambda / d``, must
    not exceed the period. This returns the smallest integer ``k`` such that
    ``k * s >= window``; ``1`` leaves an already adequate grid untouched.
    Integer refinement keeps every configured pupil cell aligned with the
    refined grid, so the configured pupil and lenslet tiling are preserved.

    Parameters
    ----------
    pupil_samples_per_lenslet
        Samples across one lenslet pitch on the configured pupil grid.
    window_lambda_over_d
        Widest detector window that light can reach, in lenslet ``lambda / d``
        at the shortest propagated wavelength. Zero (a closed field stop) needs
        no refinement.

    Returns
    -------
    int
        Positive integer refinement factor.
    """
    if pupil_samples_per_lenslet < 1:
        raise ValueError("pupil_samples_per_lenslet must be positive")
    if not math.isfinite(window_lambda_over_d) or window_lambda_over_d < 0.0:
        raise ValueError("window_lambda_over_d must be finite and non-negative")
    # The tolerance absorbs rounding in wavelength-scaled sampling, so a window
    # that equals the period exactly does not trigger a needless refinement.
    return max(1, math.ceil(window_lambda_over_d / pupil_samples_per_lenslet - 1e-9))


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
    """Sum square pixel blocks while preserving total flux.

    A thin wrapper over ``aocore.block_sum``, which owns flux-conserving
    binning (CONVENTIONS 9); it keeps this module's error messages.
    ``backend`` is accepted for compatibility; the reduction runs on the
    array's own namespace. Since aocore 0.1.3 its strided CPU adds and
    single-kernel CuPy path are at least as fast as the factor-two shortcut
    this module used to keep, on the spot stacks the Shack-Hartmann bins.
    """
    if factor < 1:
        raise ValueError("factor must be positive")
    if factor == 1:
        return array
    height, width = array.shape[-2:]
    if height % factor or width % factor:
        raise ValueError(f"shape {array.shape[-2:]} is not divisible by factor {factor}")
    return cast(NDArray[Any], _aocore_block_sum(array, factor))


def _area_overlap(target: int, source: int, *, backend: ArrayBackend, dtype: Any) -> NDArray[Any]:
    """Return the ``(target, source)`` fractions of each target cell's length.

    Both grids tile the same interval. Edges are compared in integer units of
    ``1 / (target * source)``, so the overlaps are exact and every row sums to
    one.
    """
    rows = backend.arange(target)[:, None]
    columns = backend.arange(source)[None, :]
    upper = backend.where(
        (rows + 1) * source < (columns + 1) * target,
        (rows + 1) * source,
        (columns + 1) * target,
    )
    lower = backend.where(rows * source > columns * target, rows * source, columns * target)
    overlap = backend.where(upper > lower, upper - lower, 0)
    return cast(NDArray[Any], backend.asarray(overlap, dtype=dtype) / source)


def area_rebin(
    array: NDArray[Any],
    shape: tuple[int, int],
    *,
    backend: ArrayBackend | None = None,
) -> NDArray[Any]:
    """Area-average a map onto another grid spanning the same extent.

    Each target pixel takes the mean of the source map over its own area,
    weighting every source pixel by the exact fraction it overlaps, so any
    integer or non-integer ratio, up or down, is handled without
    interpolation. A uniform map stays uniform and the area integral of the map
    is preserved. Equal shapes return the map unchanged.

    Parameters
    ----------
    array
        Two-dimensional ``(y, x)`` map on the source grid.
    shape
        Target ``(height, width)``.
    backend
        Array backend holding ``array``.

    Returns
    -------
    numpy.ndarray or cupy.ndarray
        The area-averaged map on the target grid, in ``array``'s dtype.
    """
    resolved = backend or cpu_backend()
    height, width = array.shape
    if (height, width) == tuple(shape):
        return array
    rows = _area_overlap(shape[0], height, backend=resolved, dtype=array.dtype)
    columns = _area_overlap(shape[1], width, backend=resolved, dtype=array.dtype)
    return cast(NDArray[Any], resolved.matmul(resolved.matmul(rows, array), columns.T))


@dataclass
class _SpotPropagationPlan:
    """Cached backend-resident geometry for repeated spot propagation.

    The optical transform still receives a fresh field and output on every call,
    but all geometry-only arrays are constructed once for a persistent sensor.
    Keeping this private avoids making backend/device arrays part of the public
    configuration model.
    """

    backend: ArrayBackend
    pixels: int
    samples_per_lenslet: int
    sampling: float
    oversampling: int
    field_dtype: np.dtype[Any]
    geometry: str
    geometry_value: int | float
    high_resolution_pixels: int
    field_stop_radius_lambda_over_d: float | None
    half_sample: Any | None = None
    fft_window_start: int = 0
    fft_output_start: int = 0
    fft_weights: Any | None = None
    fft_axes: tuple[int, int] = (-2, -1)
    dft_kernel: Any | None = None
    field_stop_mask: Any | None = None
    optical_blur_kernel: Any | None = None
    charge_diffusion_kernel: Any | None = None

    @classmethod
    def build(
        cls,
        *,
        pixels: int,
        samples_per_lenslet: int,
        sampling: float,
        oversampling: int,
        field_dtype: Any,
        field_stop_radius_lambda_over_d: float | None,
        optical_blur_kernel: NDArray[np.float64] | None,
        charge_diffusion_kernel: NDArray[np.float64] | None,
        backend: ArrayBackend,
    ) -> _SpotPropagationPlan:
        """Build backend-resident geometry once for one optical state."""
        geometry, geometry_value = spot_sampling_geometry(
            pixels=pixels,
            samples_per_lenslet=samples_per_lenslet,
            sampling=sampling,
            oversampling=oversampling,
        )
        high_resolution_pixels = pixels * oversampling
        plan = cls(
            backend=backend,
            pixels=pixels,
            samples_per_lenslet=samples_per_lenslet,
            sampling=float(sampling),
            oversampling=oversampling,
            field_dtype=np.dtype(field_dtype),
            geometry=geometry,
            geometry_value=geometry_value,
            high_resolution_pixels=high_resolution_pixels,
            field_stop_radius_lambda_over_d=(
                None
                if field_stop_radius_lambda_over_d is None
                else float(field_stop_radius_lambda_over_d)
            ),
            optical_blur_kernel=(
                None if optical_blur_kernel is None else backend.asarray(optical_blur_kernel)
            ),
            charge_diffusion_kernel=(
                None
                if charge_diffusion_kernel is None
                else backend.asarray(charge_diffusion_kernel)
            ),
        )
        if geometry == "fft":
            nfft = int(geometry_value)
            if high_resolution_pixels % 2 == 0:
                coordinate = backend.arange(nfft, dtype=np.float64)
                plan.half_sample = backend.exp(-1j * math.pi * coordinate / nfft)
            plan._build_fft_window(nfft)
        else:
            detector_coordinate = backend.centered_coordinates(
                high_resolution_pixels, dtype=np.float64
            ) / (sampling * oversampling)
            pupil_coordinate = backend.arange(samples_per_lenslet, dtype=np.float64)
            kernel = backend.exp(
                -2j
                * math.pi
                * detector_coordinate[:, None]
                * pupil_coordinate[None, :]
                / samples_per_lenslet
            )
            plan.dft_kernel = backend.astype(kernel, plan.field_dtype)
        if field_stop_radius_lambda_over_d is not None:
            coordinates = backend.centered_coordinates(high_resolution_pixels, dtype=np.float64)
            y, x = backend.meshgrid(coordinates, coordinates, indexing="ij")
            radius_lambda_over_d = backend.hypot(x, y) / (oversampling * sampling)
            plan.field_stop_mask = radius_lambda_over_d <= field_stop_radius_lambda_over_d
        return plan

    def _build_fft_window(self, nfft: int) -> None:
        """Cache where the lenslet field and the detector crop sit on the FFT grid.

        ``spot_intensity`` once zero-padded every lenslet to ``nfft``, applied
        the half-sample ramp and the even-grid checkerboard to the whole grid,
        transformed it all and cropped the intensity. Only the ``s x s`` field
        window holds data and only ``high_resolution_pixels`` rows and columns
        survive the crop, so the plan keeps those positions and the window's
        combined weights for :meth:`ArrayBackend.pruned_fft2`.

        The weights are the cached ramp outer product, in the ramp's complex128
        precision as the old in-place product used, with the checkerboard sign
        folded in. Negation is exact and rounding is sign-symmetric, so the
        weighted field is bit-for-bit the one the full-grid path transformed.
        """
        backend = self.backend
        samples = self.samples_per_lenslet
        start = (nfft - samples) // 2
        crop_start = nfft // 2 - self.high_resolution_pixels // 2
        weights = None
        if self.half_sample is not None:
            ramp = self.half_sample[start : start + samples]
            weights = ramp[:, None] * ramp[None, :]
        if nfft % 2 == 0:
            # ``centered_fft_intensity`` negates odd ``y + x`` samples so the
            # transform lands already centred; the crop is then contiguous.
            window = backend.arange(start, start + samples)
            parity = backend.mod(window[:, None] + window[None, :], 2)
            sign = backend.where(parity == 0, 1.0, -1.0)
            weights = (
                backend.asarray(sign, dtype=self.field_dtype) if weights is None else weights * sign
            )
            self.fft_output_start = crop_start
            # That reference transformed in place, which runs axis -2 first.
            self.fft_axes = (-2, -1)
        else:
            # Odd grids are ``fftshift``-ed after a transform into a new array
            # (axis -1 first): read the crop from the unshifted frequencies.
            self.fft_output_start = (crop_start - nfft // 2) % nfft
            self.fft_axes = (-1, -2)
        self.fft_window_start = start
        self.fft_weights = weights

    def validate(
        self,
        field: NDArray[Any],
        *,
        pixels: int,
        samples_per_lenslet: int,
        sampling: float,
        oversampling: int,
        backend: ArrayBackend,
        field_stop_radius_lambda_over_d: float | None,
        optical_blur_kernel: NDArray[np.float64] | None,
        charge_diffusion_kernel: NDArray[np.float64] | None,
    ) -> None:
        """Reject accidental use of a plan for a different physical geometry."""
        if (
            self.backend is not backend
            or self.pixels != pixels
            or self.samples_per_lenslet != samples_per_lenslet
            or self.oversampling != oversampling
            or self.field_dtype != np.dtype(field.dtype)
            or (
                self.field_stop_radius_lambda_over_d
                != (
                    None
                    if field_stop_radius_lambda_over_d is None
                    else float(field_stop_radius_lambda_over_d)
                )
            )
            or ((self.optical_blur_kernel is None) != (optical_blur_kernel is None))
            or ((self.charge_diffusion_kernel is None) != (charge_diffusion_kernel is None))
            or (
                (self.geometry == "dft" or self.field_stop_mask is not None)
                and self.sampling != float(sampling)
            )
        ):
            raise ValueError("spot propagation plan does not match the requested geometry")


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
    charge_diffusion_kernel: NDArray[np.float64] | None = None,
    _plan: _SpotPropagationPlan | None = None,
    backend: ArrayBackend | None = None,
) -> NDArray[Any]:
    """Propagate lenslet fields and integrate onto ``pixels`` detector pixels.

    The Fourier pixel scale is ``lambda / D_subap / sampling``.  Zero padding
    by ``oversampling`` improves pixel-area integration while the final block
    sum returns the configured native-pixel grid.

    ``optical_blur_fwhm_pixels`` is a focal-plane optical width in native
    detector pixels and is applied on the oversampled grid before pixel
    integration, so sub-pixel widths remain physical. ``optical_blur_kernel`` is
    a measured native-pitch kernel and is applied after pixel integration.

    ``charge_diffusion_kernel`` is the sensor's lateral charge-diffusion kernel,
    owned and built by ``getframes`` for this oversampling. Detector physics
    belongs to ``getframes``; this function only applies the supplied operator at
    the one sampling where a sub-pixel width is representable, ahead of the
    pixel-area integration that collects the diffused charge.
    """
    resolved = backend or cpu_backend()
    plan = _plan or _SpotPropagationPlan.build(
        pixels=pixels,
        samples_per_lenslet=samples_per_lenslet,
        sampling=sampling,
        oversampling=oversampling,
        field_dtype=field.dtype,
        field_stop_radius_lambda_over_d=field_stop_radius_lambda_over_d,
        optical_blur_kernel=optical_blur_kernel,
        charge_diffusion_kernel=charge_diffusion_kernel,
        backend=resolved,
    )
    plan.validate(
        field,
        pixels=pixels,
        samples_per_lenslet=samples_per_lenslet,
        sampling=sampling,
        oversampling=oversampling,
        backend=resolved,
        field_stop_radius_lambda_over_d=field_stop_radius_lambda_over_d,
        optical_blur_kernel=optical_blur_kernel,
        charge_diffusion_kernel=charge_diffusion_kernel,
    )
    geometry = plan.geometry
    geometry_value = plan.geometry_value
    high_resolution_pixels = plan.high_resolution_pixels
    if geometry == "fft":
        nfft = int(geometry_value)
        # An even detector has its optical axis on the boundary shared by its
        # central four pixels. Evaluate the Fourier transform at half-integer
        # frequency samples so equal-area integration is exactly symmetric
        # around that boundary. The pupil-plane phase ramp in ``fft_weights``
        # performs the half-sample Fourier shift without interpolating intensity
        # or changing flux. Its sign only selects the equivalent half-pixel
        # sampling branch.
        weighted = (
            field
            if plan.fft_weights is None
            else resolved.asarray(field * plan.fft_weights, dtype=field.dtype)
        )
        # Zero padding to ``nfft`` sets the physical sampling; the transform
        # skips the all-zero padding rows and the frequencies cropped away.
        # ``fftshift`` puts zero frequency at ``nfft // 2``. The crop start is
        # symmetric for odd grids; even grids become symmetric after the
        # half-sample evaluation above.
        spectrum = resolved.pruned_fft2(
            weighted,
            size=nfft,
            input_start=plan.fft_window_start,
            output_start=plan.fft_output_start,
            output_length=high_resolution_pixels,
            axes=plan.fft_axes,
            workers=workers,
        )
        cropped = resolved.abs(spectrum) ** 2
    else:
        # Evaluate the Fraunhofer transform exactly at the detector quadrature
        # points. This preserves arbitrary normalized sampling, including
        # quadcell pixels wider than lambda/D, without silently snapping the
        # physical scale to a nearby integer FFT grid.
        assert plan.dft_kernel is not None
        transformed = resolved.matmul(resolved.matmul(plan.dft_kernel, field), plan.dft_kernel.T)
        ideal_nfft = samples_per_lenslet * sampling * oversampling
        cropped = resolved.abs(transformed / ideal_nfft) ** 2
    if plan.field_stop_mask is not None:
        cropped = cropped * plan.field_stop_mask
    if optical_blur_kernel is not None and optical_blur_fwhm_pixels > 0.0:
        raise ValueError("provide either optical_blur_fwhm_pixels or optical_blur_kernel")
    if optical_blur_fwhm_pixels > 0.0:
        # Residual focal-plane optical blur acts on the continuous irradiance
        # before each pixel integrates over its own area. Convolving the
        # already-summed native grid instead is unrepresentable for sub-pixel
        # widths: a sigma below about half a native pixel leaves a discrete
        # kernel indistinguishable from a delta function, so the configured
        # value would silently do nothing. Blur the oversampled grid, where the
        # same physical width is resolved, and let block_sum integrate after.
        sigma = optical_blur_fwhm_pixels * oversampling / 2.3548200450309493
        if sigma < 0.5:
            raise ValueError(
                "optical_blur_fwhm_pixels "
                f"{optical_blur_fwhm_pixels} is not representable at "
                f"numerics.fft_oversampling {oversampling}: it needs a "
                "focal-plane sigma of at least 0.5 oversampled samples. "
                "Raise fft_oversampling to at least "
                f"{math.ceil(0.5 * 2.3548200450309493 / optical_blur_fwhm_pixels)}."
            )
        cropped = resolved.gaussian_filter(cropped, sigma=(0.0, sigma, sigma))
    if plan.charge_diffusion_kernel is not None:
        # Detector-owned operator, applied last in the focal plane: charge
        # diffuses in the silicon and only then is collected per pixel below.
        cropped = resolved.convolve(cropped, plan.charge_diffusion_kernel[None, ...])
    native = block_sum(cropped, oversampling, backend=resolved)
    if plan.optical_blur_kernel is not None:
        # A measured kernel is supplied on the native pixel pitch, so it is the
        # one blur that belongs after pixel integration.
        native = resolved.convolve(native, plan.optical_blur_kernel[None, ...])
    # Keep the precision produced by the complex FFT through pixel integration.
    # Sensor-level accumulation intentionally converts to the configured photon
    # rate dtype, but this avoids promoting the large spot batch prematurely.
    return native


__all__ = [
    "area_rebin",
    "block_sum",
    "crop_center",
    "lenslet_field_upsampling",
    "load_blur_kernel",
    "pad_center",
    "spot_intensity",
    "spot_sampling_geometry",
]
