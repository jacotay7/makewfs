"""Shape-specialized CUDA execution for Shack--Hartmann DFT propagation.

The public numerical contract remains in :mod:`makewfs.sensors.shack_hartmann`.
This module is a private execution plan for compatible CuPy configurations: it
generates CUDA source whose loop bounds and shared-memory layout are constants,
then lets :class:`cupy.RawKernel` compile and cache the binary on first use.
Unsupported optical features continue through the readable array-operation
reference path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray


class _CompiledOpticalArrays(NamedTuple):
    """Device arrays produced by one compiled optical execution."""

    photon_rate: Any
    spectral_photon_rate: Any
    captured_rate_per_s: Any


@dataclass(frozen=True)
class _ExecutorSignature:  # pragma: no cover - optional CUDA execution
    """Compile-time geometry for one CUDA kernel variant."""

    dtype_name: str
    sample_count: int
    lenslets: int
    pupil_samples: int
    pixels: int
    oversampling: int
    margin: int
    wavelengths: int
    field_stop: bool
    focal_kernel_size: int

    @property
    def high_resolution_pixels(self) -> int:
        return self.pixels * self.oversampling

    @property
    def internal_width(self) -> int:
        return self.lenslets * self.pupil_samples

    @property
    def output_width(self) -> int:
        return self.lenslets * self.pixels + 2 * self.margin

    @property
    def block_threads(self) -> int:
        if self.focal_kernel_size:
            return self.pixels**2 * 32
        work = max(32, self.pupil_samples**2, self.pixels**2)
        return ((work + 31) // 32) * 32

    @property
    def shared_bytes(self) -> int:
        real_bytes = 4 if self.dtype_name == "float32" else 8
        complex_bytes = 2 * real_bytes
        high = self.high_resolution_pixels
        field = self.sample_count * self.pupil_samples**2 * complex_bytes
        intermediate = self.sample_count * high * self.pupil_samples * complex_bytes
        intensity = self.sample_count * high**2 * real_bytes if self.focal_kernel_size else 0
        pistons = self.sample_count * 8
        return field + intermediate + intensity + pistons


class _CompiledShackHartmannExecutor:  # pragma: no cover - optional CUDA execution
    """One first-use-JIT CUDA kernel and its immutable launch data."""

    def __init__(self, engine: Any, sample_count: int) -> None:
        charge_kernel = engine._charge_diffusion_kernel
        field_stop = engine.settings.field_stop_radius_lambda_over_d is not None
        focal_kernel_size = 0 if charge_kernel is None else int(charge_kernel.shape[0])
        self.signature = _ExecutorSignature(
            dtype_name=engine.config.numerics.dtype,
            sample_count=sample_count,
            lenslets=engine.n_lenslets,
            pupil_samples=engine.samples_per_lenslet,
            pixels=engine.settings.pixels_per_subaperture,
            oversampling=engine.config.numerics.fft_oversampling,
            margin=engine.settings.detector_margin_pixels,
            wavelengths=len(engine._wavelengths),
            field_stop=field_stop,
            focal_kernel_size=focal_kernel_size,
        )
        if self.signature.block_threads > 1024:
            raise ValueError("compiled Shack-Hartmann geometry needs more than 1024 threads")
        # Forty-eight KiB is the portable per-block floor across the supported
        # CUDA devices. Larger geometries retain the general implementation.
        if self.signature.shared_bytes > 48 * 1024:
            raise ValueError("compiled Shack-Hartmann geometry exceeds 48 KiB shared memory")

        self.backend = engine.backend
        self.xp = engine.backend.xp
        self._kernel = self.xp.RawKernel(
            _kernel_source(self.signature),
            "makewfs_compiled_shack_hartmann",
            options=("-std=c++11",),
        )
        self._grid = (self.signature.lenslets**2,)
        self._block = (self.signature.block_threads,)
        self._piston_index = int(engine._piston_index[0]) * self.signature.internal_width + int(
            engine._piston_index[1]
        )
        self._lenslet_mask = engine.lenslet_mask
        self._angles = engine._field_angle_opd
        self._dft_kernels = tuple(plan.dft_kernel for plan in engine._spot_plans)
        self._field_stops = tuple(
            self._lenslet_mask if plan.field_stop_mask is None else plan.field_stop_mask
            for plan in engine._spot_plans
        )
        real_dtype = np.dtype(self.signature.dtype_name)
        self._integration_weights = (
            self._lenslet_mask
            if charge_kernel is None
            else self.backend.asarray(
                _fused_integration_weights(
                    self.backend.to_host(charge_kernel),
                    pixels=self.signature.pixels,
                    oversampling=self.signature.oversampling,
                    dtype=real_dtype,
                ),
                dtype=real_dtype,
            )
        )
        total_field_flux = self.backend.scalar(engine._total_field_flux)
        self._state_arguments = tuple(
            (
                float(state.wavelength_m),
                float(
                    1.0
                    / (
                        engine.samples_per_lenslet
                        * sampling
                        * engine.config.numerics.fft_oversampling
                    )
                ),
                float(engine.source_rate * state.weight / total_field_flux),
                int(engine._state_wavelength_indices[index]),
            )
            for index, (state, sampling) in enumerate(
                zip(engine.source_states, engine._state_spot_sampling)
            )
        )

    def render(self, internal: Any) -> _CompiledOpticalArrays:
        """Execute every incoherent state into newly owned rate arrays."""
        shape = (self.signature.output_width, self.signature.output_width)
        spectral = self.xp.zeros((self.signature.wavelengths, *shape), dtype=np.float64)
        photon_rate = self.xp.zeros(shape, dtype=np.float64)
        for index, arguments in enumerate(self._state_arguments):
            wavelength_m, inverse_scale, contribution_scale, wavelength_index = arguments
            self._kernel(
                self._grid,
                self._block,
                (
                    internal,
                    self._angles[index],
                    self._lenslet_mask,
                    self._dft_kernels[index],
                    self._field_stops[index],
                    self._integration_weights,
                    np.float64(wavelength_m),
                    np.float64(inverse_scale),
                    np.float64(contribution_scale),
                    photon_rate,
                    spectral,
                    np.int32(wavelength_index),
                    np.int32(self._piston_index),
                ),
            )
        return _CompiledOpticalArrays(
            photon_rate,
            spectral,
            self.backend.sum(photon_rate),
        )


def _compiled_executor_rejection(  # pragma: no cover - optional CUDA execution
    engine: Any, sample_count: int
) -> str | None:
    """Return why the exact compiled path is unavailable, or ``None``."""
    if engine.backend.is_cpu:
        return "CPU backend"
    if sample_count < 1:
        return "empty temporal batch"
    if engine.settings.optical_blur_fwhm_pixels > 0.0:
        return "continuous optical blur"
    if engine._optical_blur_kernel is not None:
        return "measured native-pixel optical blur"
    if any(plan.geometry != "dft" or plan.dft_kernel is None for plan in engine._spot_plans):
        return "non-DFT spot geometry"
    charge_kernel = engine._charge_diffusion_kernel
    if charge_kernel is not None:
        shape = tuple(int(value) for value in charge_kernel.shape)
        if len(shape) != 2 or shape[0] != shape[1] or shape[0] % 2 != 1:
            return "non-square or even charge-diffusion kernel"
    signature = _ExecutorSignature(
        dtype_name=engine.config.numerics.dtype,
        sample_count=sample_count,
        lenslets=engine.n_lenslets,
        pupil_samples=engine.samples_per_lenslet,
        pixels=engine.settings.pixels_per_subaperture,
        oversampling=engine.config.numerics.fft_oversampling,
        margin=engine.settings.detector_margin_pixels,
        wavelengths=len(engine._wavelengths),
        field_stop=engine.settings.field_stop_radius_lambda_over_d is not None,
        focal_kernel_size=0 if charge_kernel is None else int(charge_kernel.shape[0]),
    )
    if signature.block_threads > 1024:
        return "more than 1024 threads per block"
    if signature.shared_bytes > 48 * 1024:
        return "more than 48 KiB shared memory per block"
    return None


def _fused_integration_weights(
    focal_kernel: NDArray[Any],
    *,
    pixels: int,
    oversampling: int,
    dtype: np.dtype[Any],
) -> NDArray[Any]:
    """Compose focal convolution and native-pixel integration exactly.

    The source intensity coefficient for each native pixel is immutable for a
    compiled geometry. Building it once removes the repeated kernel traversal
    from every lenslet, wavelength, and temporal sample.
    """
    high_pixels = pixels * oversampling
    native_pixels = pixels * pixels
    # This is host compiler data, built once before upload rather than a
    # portable sensor-runtime array. Keep that boundary visibly separate from
    # the ArrayBackend-owned frame path.
    weights: NDArray[Any] = np.ndarray((native_pixels, high_pixels * high_pixels), dtype=dtype)
    weights.fill(0)
    radius = focal_kernel.shape[0] // 2
    for native_y in range(pixels):
        for native_x in range(pixels):
            native_index = native_y * pixels + native_x
            for dy in range(oversampling):
                output_y = native_y * oversampling + dy
                for dx in range(oversampling):
                    output_x = native_x * oversampling + dx
                    for kernel_y in range(focal_kernel.shape[0]):
                        source_y = output_y + radius - kernel_y
                        if source_y < 0 or source_y >= high_pixels:
                            continue
                        for kernel_x in range(focal_kernel.shape[1]):
                            source_x = output_x + radius - kernel_x
                            if source_x < 0 or source_x >= high_pixels:
                                continue
                            source_index = source_y * high_pixels + source_x
                            weights[native_index, source_index] += focal_kernel[kernel_y, kernel_x]
    return weights


def _kernel_source(  # pragma: no cover - optional CUDA execution
    signature: _ExecutorSignature,
) -> str:
    """Generate one constant-bound CUDA C kernel."""
    if signature.dtype_name == "float32":
        real_type = "float"
        complex_type = "float2"
        sincos_call = "sincosf(phase, &phase_sin, &phase_cos);"
    else:
        real_type = "double"
        complex_type = "double2"
        sincos_call = "sincos(phase, &phase_sin, &phase_cos);"
    return f"""
#define SAMPLE_COUNT {signature.sample_count}
#define LENSLETS {signature.lenslets}
#define PUPIL_SAMPLES {signature.pupil_samples}
#define PIXELS {signature.pixels}
#define OVERSAMPLING {signature.oversampling}
#define HIGH_PIXELS {signature.high_resolution_pixels}
#define INTERNAL_WIDTH {signature.internal_width}
#define INTERNAL_PIXELS {signature.internal_width**2}
#define OUTPUT_WIDTH {signature.output_width}
#define OUTPUT_PIXELS {signature.output_width**2}
#define MARGIN {signature.margin}
#define HAS_FIELD_STOP {1 if signature.field_stop else 0}
#define FOCAL_KERNEL_SIZE {signature.focal_kernel_size}

typedef {real_type} real_t;
typedef {complex_type} complex_t;

__device__ __forceinline__ complex_t complex_make(real_t real, real_t imag) {{
    complex_t value;
    value.x = real;
    value.y = imag;
    return value;
}}

extern "C" __global__ void makewfs_compiled_shack_hartmann(
    const double* internal,
    const double* angle,
    const real_t* pupil,
    const complex_t* dft_kernel,
    const bool* field_stop,
    const real_t* integration_weights,
    const double wavelength_m,
    const double inverse_scale,
    const double contribution_scale,
    double* photon_rate,
    double* spectral_rate,
    const int wavelength_index,
    const int piston_index
) {{
    const int lenslet = blockIdx.x;
    const int thread = threadIdx.x;
    const int lenslet_y = lenslet / LENSLETS;
    const int lenslet_x = lenslet - lenslet_y * LENSLETS;
    __shared__ double pistons[SAMPLE_COUNT];
    __shared__ complex_t fields[SAMPLE_COUNT * PUPIL_SAMPLES * PUPIL_SAMPLES];
    __shared__ complex_t intermediate[SAMPLE_COUNT * HIGH_PIXELS * PUPIL_SAMPLES];
#if FOCAL_KERNEL_SIZE > 0
    __shared__ real_t intensity[SAMPLE_COUNT * HIGH_PIXELS * HIGH_PIXELS];
#endif

    for (int sample = thread; sample < SAMPLE_COUNT; sample += blockDim.x) {{
        pistons[sample] = internal[sample * INTERNAL_PIXELS + piston_index]
            + angle[piston_index];
    }}
    __syncthreads();

    for (
        int item = thread;
        item < SAMPLE_COUNT * PUPIL_SAMPLES * PUPIL_SAMPLES;
        item += blockDim.x
    ) {{
        const int sample = item / (PUPIL_SAMPLES * PUPIL_SAMPLES);
        const int local = item - sample * PUPIL_SAMPLES * PUPIL_SAMPLES;
        const int pupil_y = local / PUPIL_SAMPLES;
        const int pupil_x = local - pupil_y * PUPIL_SAMPLES;
        const int input_index = (lenslet_y * PUPIL_SAMPLES + pupil_y) * INTERNAL_WIDTH
            + lenslet_x * PUPIL_SAMPLES + pupil_x;
        const double relative_opd = internal[sample * INTERNAL_PIXELS + input_index]
            + angle[input_index] - pistons[sample];
        const real_t phase = (real_t)(6.283185307179586476925286766559
            * relative_opd / wavelength_m);
        real_t phase_sin;
        real_t phase_cos;
        {sincos_call}
        const real_t amplitude = pupil[input_index];
        fields[item] = complex_make(amplitude * phase_cos, amplitude * phase_sin);
    }}
    __syncthreads();

    for (
        int item = thread;
        item < SAMPLE_COUNT * HIGH_PIXELS * PUPIL_SAMPLES;
        item += blockDim.x
    ) {{
        const int sample = item / (HIGH_PIXELS * PUPIL_SAMPLES);
        const int local = item - sample * HIGH_PIXELS * PUPIL_SAMPLES;
        const int output_y = local / PUPIL_SAMPLES;
        const int pupil_x = local - output_y * PUPIL_SAMPLES;
        real_t real = (real_t)0;
        real_t imag = (real_t)0;
        for (int pupil_y = 0; pupil_y < PUPIL_SAMPLES; ++pupil_y) {{
            const complex_t kernel = dft_kernel[output_y * PUPIL_SAMPLES + pupil_y];
            const complex_t field = fields[
                sample * PUPIL_SAMPLES * PUPIL_SAMPLES
                + pupil_y * PUPIL_SAMPLES + pupil_x
            ];
            real += kernel.x * field.x - kernel.y * field.y;
            imag += kernel.x * field.y + kernel.y * field.x;
        }}
        intermediate[item] = complex_make(real, imag);
    }}
    __syncthreads();

#if FOCAL_KERNEL_SIZE > 0
    for (
        int item = thread;
        item < SAMPLE_COUNT * HIGH_PIXELS * HIGH_PIXELS;
        item += blockDim.x
    ) {{
        const int sample = item / (HIGH_PIXELS * HIGH_PIXELS);
        const int local = item - sample * HIGH_PIXELS * HIGH_PIXELS;
        const int output_y = local / HIGH_PIXELS;
        const int output_x = local - output_y * HIGH_PIXELS;
        real_t real = (real_t)0;
        real_t imag = (real_t)0;
        for (int pupil_x = 0; pupil_x < PUPIL_SAMPLES; ++pupil_x) {{
            const complex_t left = intermediate[
                sample * HIGH_PIXELS * PUPIL_SAMPLES
                + output_y * PUPIL_SAMPLES + pupil_x
            ];
            const complex_t right = dft_kernel[output_x * PUPIL_SAMPLES + pupil_x];
            real += left.x * right.x - left.y * right.y;
            imag += left.x * right.y + left.y * right.x;
        }}
        real *= (real_t)inverse_scale;
        imag *= (real_t)inverse_scale;
        real_t value = real * real + imag * imag;
#if HAS_FIELD_STOP
        value = field_stop[local] ? value : (real_t)0;
#endif
        intensity[item] = value;
    }}
    __syncthreads();
#endif

#if FOCAL_KERNEL_SIZE == 0
    for (int native_pixel = thread; native_pixel < PIXELS * PIXELS; native_pixel += blockDim.x) {{
        const int native_y = native_pixel / PIXELS;
        const int native_x = native_pixel - native_y * PIXELS;
        real_t temporal_total = (real_t)0;
        for (int sample = 0; sample < SAMPLE_COUNT; ++sample) {{
            real_t sample_total = (real_t)0;
            for (int dy = 0; dy < OVERSAMPLING; ++dy) {{
                const int output_y = native_y * OVERSAMPLING + dy;
                for (int dx = 0; dx < OVERSAMPLING; ++dx) {{
                    const int output_x = native_x * OVERSAMPLING + dx;
                    real_t real = (real_t)0;
                    real_t imag = (real_t)0;
                    for (int pupil_x = 0; pupil_x < PUPIL_SAMPLES; ++pupil_x) {{
                        const complex_t left = intermediate[
                            sample * HIGH_PIXELS * PUPIL_SAMPLES
                            + output_y * PUPIL_SAMPLES + pupil_x
                        ];
                        const complex_t right = dft_kernel[
                            output_x * PUPIL_SAMPLES + pupil_x
                        ];
                        real += left.x * right.x - left.y * right.y;
                        imag += left.x * right.y + left.y * right.x;
                    }}
                    real *= (real_t)inverse_scale;
                    imag *= (real_t)inverse_scale;
                    real_t value = real * real + imag * imag;
#if HAS_FIELD_STOP
                    value = field_stop[output_y * HIGH_PIXELS + output_x]
                        ? value : (real_t)0;
#endif
                    sample_total += value;
                }}
            }}
            temporal_total += sample_total;
        }}
        const double contribution = (double)temporal_total
            * contribution_scale / (double)SAMPLE_COUNT;
        const int output_y = MARGIN + lenslet_y * PIXELS + native_y;
        const int output_x = MARGIN + lenslet_x * PIXELS + native_x;
        const int output_index = output_y * OUTPUT_WIDTH + output_x;
        photon_rate[output_index] += contribution;
        spectral_rate[wavelength_index * OUTPUT_PIXELS + output_index] += contribution;
    }}
#endif

#if FOCAL_KERNEL_SIZE > 0
    const int native_pixel = thread / 32;
    const int lane = thread & 31;
    if (native_pixel < PIXELS * PIXELS) {{
        real_t temporal_total = (real_t)0;
        for (int sample = 0; sample < SAMPLE_COUNT; ++sample) {{
            real_t sample_total = (real_t)0;
            for (
                int source = lane;
                source < HIGH_PIXELS * HIGH_PIXELS;
                source += 32
            ) {{
                sample_total += intensity[
                    sample * HIGH_PIXELS * HIGH_PIXELS + source
                ] * integration_weights[
                    native_pixel * HIGH_PIXELS * HIGH_PIXELS + source
                ];
            }}
            for (int offset = 16; offset > 0; offset /= 2) {{
                sample_total += __shfl_down_sync(0xffffffff, sample_total, offset);
            }}
            temporal_total += sample_total;
        }}
        if (lane == 0) {{
            const int native_y = native_pixel / PIXELS;
            const int native_x = native_pixel - native_y * PIXELS;
            const double contribution = (double)temporal_total
                * contribution_scale / (double)SAMPLE_COUNT;
            const int output_y = MARGIN + lenslet_y * PIXELS + native_y;
            const int output_x = MARGIN + lenslet_x * PIXELS + native_x;
            const int output_index = output_y * OUTPUT_WIDTH + output_x;
            photon_rate[output_index] += contribution;
            spectral_rate[wavelength_index * OUTPUT_PIXELS + output_index] += contribution;
        }}
    }}
#endif
}}
"""


__all__ = [
    "_CompiledOpticalArrays",
    "_CompiledShackHartmannExecutor",
    "_compiled_executor_rejection",
    "_fused_integration_weights",
]
