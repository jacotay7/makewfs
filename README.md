# makewfs

[![CI](https://github.com/jacotay7/makewfs/actions/workflows/ci.yml/badge.svg)](https://github.com/jacotay7/makewfs/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/makewfs.svg)](https://pypi.org/project/makewfs/)
[![Python](https://img.shields.io/pypi/pyversions/makewfs.svg)](https://pypi.org/project/makewfs/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

`makewfs` turns a configured pupil-plane phase/OPD map into a realistic
adaptive-optics wavefront-sensor image. The supported sensors are
Shack–Hartmann and four-face pyramid sensors.

The package owns the wavefront-sensor optics. It deliberately reuses
[`pyturb`](https://github.com/jacotay7/pyturb) for atmospheric OPD and
[`getframes`](https://github.com/jacotay7/getframes) for detector response and
noise; neither model will be reimplemented here.

> **Status:** stable. `makewfs` 1.0 freezes the configuration-driven
> Shack–Hartmann and four-face pyramid API under
> [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Broadband source
> morphology, wavelength-resolved detector QE, Shack–Hartmann sodium-layer
> elongation, and optional end-to-end CuPy execution are supported. The
> documented mean-altitude LGS OPD approximation remains intentionally distinct
> from future range-resolved turbulence.

- [Complete implementation roadmap](ROADMAP.md)
- [Instructions for implementation agents](AGENTS.md)

The minimal API is:

```python
import makewfs

wfs = makewfs.WavefrontSensor.from_toml("wfs.toml")
frame = wfs.expose(opd_m, seed=0)  # getframes.Frame, data in ADU
```

Install the CPU package from PyPI with:

```bash
python -m pip install makewfs
```

For the optional CUDA 12.x path:

```bash
python -m pip install "makewfs[gpu]"
```

Set `device = "gpu"` under `[numerics]` and pass a CuPy OPD array for the
device-resident path. `frame.data` is then a CuPy ADU array; use
`getframes.to_numpy(frame.data)` only at an intentional host boundary.
Compatible sampled-DFT Shack--Hartmann geometries are specialized and compiled
on their first use, then reuse CuPy's process and disk kernel caches. The first
render can therefore be much slower than steady state; construct and warm each
fixed sensor before measuring or entering a real-time loop. Unsupported optical
features automatically retain the array-reference implementation.

The repository includes a runnable starter configuration at
[`examples/configs/shack_hartmann_minimal.toml`](examples/configs/shack_hartmann_minimal.toml).
The corresponding pyramid starter is at
[`examples/configs/pyramid_minimal.toml`](examples/configs/pyramid_minimal.toml).
For an ideal photon-rate map, use `wfs.photon_rate(opd_m)`; for one detector
exposure containing several temporal OPD samples, use
`wfs.expose_integrated(samples)`.
High-rate sequential owners may pass a backend-native `uint32` `out=` array to
`expose()` or `expose_integrated()`; the returned frame then aliases that explicit
caller-owned destination.

For closed-loop work, construct the sensor once and call `expose()` with each
new residual wavefront. The only per-frame inputs are the wavefront and,
optionally, a detector-noise seed; telescope, source, sensor, sampling, and
camera choices live in the configuration.

## CPU/GPU throughput

Warm end-to-end frame throughput on an AMD Ryzen 9 9950X3D and NVIDIA RTX 5090
includes optics, `getframes` detector noise, truth, and ADU, with device-resident
input/output and no host transfers:

| Workflow | Output | Work samples | CPU (frames/s) | GPU (frames/s) | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20x20 SH, float32 | 160x160 | 1 | 143.4 | 2,011.2 | 14.03x |
| 60x60 SH, float64 | 360x360 | 1 | 26.3 | 945.2 | 35.94x |
| Broadband/range-sampled SH | 64x64 | 9 | 99.8 | 557.1 | 5.58x |
| Pyramid, 8-point modulation, float32 | 80x80 | 8 | 668.5 | 1,579.4 | 2.36x |
| Pyramid, 32-point modulation, float64 | 108x108 | 32 | 33.3 | 903.9 | 27.18x |

Higher is better. A tiny unmodulated 54x54 pyramid case remains CPU-faster
(3,570 versus 1,541 frames/s) because GPU launch overhead dominates its small
optical workload. See the [full snapshot](benchmarks/device-results.md),
[raw JSON](benchmarks/device-results.json), and
[benchmark methodology](docs/performance.md#cpu-versus-gpu-reference-throughput).
