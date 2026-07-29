# Performance and precision

Construct a `WavefrontSensor` once. Pupil masks, source quadrature, FFT geometry,
pyramid masks, and the `getframes.Camera` adapter are built at construction;
`photon_rate` and `expose` are the warm paths.

The CPU implementation uses SciPy batched FFTs and supports `fft_workers` plus
float32/complex64 or float64/complex128 optical arithmetic. Float32 is useful
for throughput and memory; float64 is the reference path. Benchmark cold
construction separately from warm frames and record Python, NumPy, SciPy, and
hardware versions.

The common two-times Shack--Hartmann oversampling path integrates detector
pixels with direct strided sums. Temporally integrated exposures accumulate
photon-rate, spectral-rate, and OPD arrays incrementally, avoiding a second
full-frame stack. These are mathematically equivalent allocation/reduction
optimizations; temporal samples are still rendered as separate intensities.

On GPU, Shack--Hartmann source states that resolve to the same FFT size are
propagated in one device batch. State accumulation order and the CPU reference
remain unchanged. A configured wavelength-scaled field stop keeps states
sequential because its mask differs with sampling. On a Quadro P620, the
eight-state, four-temporal-sample HAKA-class path groups the first five
wavelengths and leaves the other three single. A matched, alternating-order
64-frame benchmark measured 37.19 ms sequential versus 36.39 ms grouped median
optics time, a 2.16% reduction. The maximum relative ideal photon-rate
difference was 5.1e-8. Reproduce it with:

```bash
python benchmarks/benchmark_sh_state_batching.py path/to/haka-resolved-wfs.toml \
  --frames 64 --temporal-samples 4
```

The versioned Quadro P620 record is
`benchmarks/haka-sh-state-batching-quadro-p620.json`. It is a local matched
measurement, not a portable latency guarantee.

Install the optional CUDA extra, then set
`numerics.device = "gpu"` in the WFS TOML:

```bash
python -m pip install 'makewfs[gpu]'
python -m pytest -q -m gpu
```

Sensor allocations, reductions, FFTs, interpolation, blur hooks, photon-rate
maps, detector stochastic samples, truth, and ADU remain on the device.
`frame.data` is the zero-copy CuPy interface; `np.asarray(frame)`, FITS output,
plotting, or `getframes.to_numpy` intentionally transfer to the host. CPU and GPU
use independent random streams, so optical arrays are compared numerically while
detector parity is validated statistically and seeded repetition is checked on
each backend.

## CPU versus GPU reference throughput

The July 2026 development reference used an AMD Ryzen 9 9950X3D and NVIDIA RTX
5090 with Python 3.12, NumPy 2.2.6, SciPy 1.16.3, CuPy 13.6.0, `getframes`
2.1.0, and `pyturb` 1.0.0. Each row constructs one persistent sensor, performs
an untimed end-to-end warm-up, and records 100 frames. A zero-OPD array enters on
the selected device; optics, detector stochastic samples, truth, and ADU remain
there. Every frame receives a distinct deterministic detector-noise seed. CUDA
is synchronized around timed regions; construction and host transfers are
excluded from frames/s.

“Work samples” is source states multiplied by modulation points: it is 9 for the
three-wavelength by three-range LGS case and 8 or 32 for the modulated pyramid
cases.

| Configuration | Sensor | Output | Work samples | CPU (frames/s) | GPU (frames/s) | Speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `shack_hartmann_20x20_float32.toml` | SH | 160x160 | 1 | 143.4 | 2,011.2 | 14.03x |
| `shack_hartmann_60x60_float64.toml` | SH | 360x360 | 1 | 26.3 | 945.2 | 35.94x |
| `shack_hartmann_broadband_lgs.toml` | SH | 64x64 | 9 | 99.8 | 557.1 | 5.58x |
| `pyramid_40_float32.toml` | pyramid | 54x54 | 1 | 3,569.8 | 1,541.1 | 0.43x |
| `pyramid_60_mod8_float32.toml` | pyramid | 80x80 | 8 | 668.5 | 1,579.4 | 2.36x |
| `pyramid_80_mod32_float64.toml` | pyramid | 108x108 | 32 | 33.3 | 903.9 | 27.18x |

Higher frames/s is better. Relative to the first end-to-end GPU snapshot on the
same machine, these kernels are 1.19x–1.73x faster on CPU and 1.42x–2.58x faster
on GPU. Static source/range geometry, modulation phasors, interpolation grids,
normalizers, and monochromatic spectral views are cached. SH uses an
intensity-only centered FFT that removes an irrelevant input permutation; both
sensors use native orthonormal FFT scaling, a fixed illuminated piston reference,
and batched metadata scalar transfers. The GPU detector improvements described
in `getframes` are included in every end-to-end row.

Pyramid behavior still shows the GPU crossover clearly: launch overhead makes
the tiny unmodulated case slower, eight modulation points provide a 2.36x gain,
and the 32-point float64 case reaches 27.2x. The broadband LGS case has nine
incoherent source states but a small 64x64 detector and gains 5.58x.

The [rendered snapshot](https://github.com/jacotay7/makewfs/blob/main/benchmarks/device-results.md)
and [raw JSON](https://github.com/jacotay7/makewfs/blob/main/benchmarks/device-results.json)
record exact configuration paths, command, revision, dirty-checkout flag,
environment, timings, and detector-only rates. The checked-in snapshot records a
specific local environment and is not a cross-hardware performance guarantee.

The repository benchmark runner separates cold construction, warm optical
frames, end-to-end frames, and detector-only frames:

```bash
python benchmarks/run.py --frames 10 --output benchmark-results.json
python benchmarks/run.py --device both --frames 100 \
  --config benchmarks/configs/shack_hartmann_20x20_float32.toml \
  --config benchmarks/configs/shack_hartmann_60x60_float64.toml \
  --config benchmarks/configs/shack_hartmann_broadband_lgs.toml \
  --config benchmarks/configs/pyramid_40_float32.toml \
  --config benchmarks/configs/pyramid_60_mod8_float32.toml \
  --config benchmarks/configs/pyramid_80_mod32_float64.toml \
  --output benchmarks/device-results.json
python benchmarks/render_device_table.py benchmarks/device-results.json \
  --output benchmarks/device-results.md
```

Benchmark JSON includes source-state count, output shape, exact invocation,
revision state, environment, methodology, and per-frame optical, end-to-end, and
detector-only timings. CUDA is synchronized around timed regions.

Each returned detector frame also records `wfs_optical_render_s`,
`wfs_detector_expose_s`, and `wfs_total_expose_s`. These are lightweight
diagnostics for a closed-loop driver, not scheduling guarantees or controller
abstractions.

Representative Shack–Hartmann configurations are provided under
`benchmarks/configs/`:

```bash
python benchmarks/run.py --representative --frames 3 --output benchmark-results.json
```

They cover 20×20 float32 and 60×60 float64 lenslet grids, a broadband
range-resolved SH source, and 40/60/80-pixel pyramid cases with 1/8/32
modulation samples while keeping the minimal SH and pyramid smoke cases in the
default benchmark run.

The repository also keeps a dated reference snapshot with hardware, Python,
dependency, precision, source-state, construction, warm-optics, and detector
columns in
[`benchmarks/reference-table.md`](https://github.com/jacotay7/makewfs/blob/main/benchmarks/reference-table.md).
Regenerate it on a new machine with:

```bash
python benchmarks/run.py --representative --frames 3 --measure-memory \
  --output benchmarks/reference-results.json
python benchmarks/render_table.py benchmarks/reference-results.json --output benchmarks/reference-table.md
```

For call-level hotspot inspection, write a compact cumulative `cProfile` report:

```bash
python benchmarks/profile_warm.py --frames 3 --output profile-results.json
```

CI also runs `benchmarks/check_regression.py` on the representative report. It
checks generous same-run ratios for 60x60 versus 20x20 SH, the nine-state
broadband LGS case versus minimal SH, and 32-sample versus unmodulated pyramid
optics. These are order-of-magnitude regression alarms, not portable latency
promises; inspect the full JSON report for machine-specific performance work.
