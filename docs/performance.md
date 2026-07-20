# Performance and precision

Construct a `WavefrontSensor` once. Pupil masks, source quadrature, FFT geometry,
pyramid masks, and the `getframes.Camera` adapter are built at construction;
`photon_rate` and `expose` are the warm paths.

The CPU implementation uses SciPy batched FFTs and supports `fft_workers` plus
float32/complex64 or float64/complex128 optical arithmetic. Float32 is useful
for throughput and memory; float64 is the reference path. Benchmark cold
construction separately from warm frames and record Python, NumPy, SciPy, and
hardware versions.

Install the optional CUDA extra plus the GPU-capable sibling checkout, then set
`numerics.device = "gpu"` in the WFS TOML:

```bash
python -m pip install -e '../getframes[gpu]'
python -m pip install -e '.[gpu]'
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
| `shack_hartmann_20x20_float32.toml` | SH | 160x160 | 1 | 82.9 | 1,068.5 | 12.89x |
| `shack_hartmann_60x60_float64.toml` | SH | 360x360 | 1 | 16.7 | 665.6 | 39.90x |
| `shack_hartmann_broadband_lgs.toml` | SH | 64x64 | 9 | 74.0 | 215.6 | 2.91x |
| `pyramid_40_float32.toml` | pyramid | 54x54 | 1 | 2,872.0 | 874.8 | 0.30x |
| `pyramid_60_mod8_float32.toml` | pyramid | 80x80 | 8 | 441.5 | 815.4 | 1.85x |
| `pyramid_80_mod32_float64.toml` | pyramid | 108x108 | 32 | 28.1 | 603.5 | 21.47x |

Higher frames/s is better. SH gains grow from 12.9x to 39.9x as the lenslet grid
grows. Pyramid behavior shows the GPU crossover particularly clearly: launch
overhead makes the tiny unmodulated case slower, eight modulation points provide
a 1.85x gain, and the 32-point float64 case reaches 21.5x. The broadband LGS
case has nine incoherent source states but a small 64x64 detector and gains 2.91x.

The [rendered snapshot](https://github.com/jacotay7/makewfs/blob/main/benchmarks/device-results.md)
and [raw JSON](https://github.com/jacotay7/makewfs/blob/main/benchmarks/device-results.json)
record exact configuration paths, command, revision, dirty-checkout flag,
environment, timings, and detector-only rates. The checked-in snapshot was
recorded from the GPU development checkout; it is local evidence, not a release
or CI performance guarantee.

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
