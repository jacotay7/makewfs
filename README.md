# makewfs

[![CI](https://github.com/jacotay7/makewfs/actions/workflows/ci.yml/badge.svg)](https://github.com/jacotay7/makewfs/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/makewfs.svg)](https://pypi.org/project/makewfs/)
[![Python](https://img.shields.io/pypi/pyversions/makewfs.svg)](https://pypi.org/project/makewfs/)
[![Docs](https://img.shields.io/badge/docs-jacotay7.github.io%2Fmakewfs-teal.svg)](https://jacotay7.github.io/makewfs/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Documentation: [jacotay7.github.io/makewfs](https://jacotay7.github.io/makewfs/)**

**Configuration-driven adaptive-optics wavefront-sensor images.**

<p align="center">
  <img src="examples/makewfs_showcase.webp" width="503" alt="Animated wavefront sensor showcase: Shack-Hartmann, high-order Shack-Hartmann, modulated pyramid and broadband LGS sensors watching one wind-blown atmosphere, with live throughput.">
</p>

`makewfs` turns a pupil-plane phase/OPD map into the image an adaptive-optics
wavefront sensor would actually record. It supports **Shack–Hartmann** and
**four-face pyramid** sensors, with modulation, broadband and finite-source
quadrature, and sodium-layer LGS geometry — everything from the pupil to the ADU.

The package owns the wavefront-sensor optics and nothing else. It deliberately
reuses [`pyturb`](https://github.com/jacotay7/pyturb) for atmospheric OPD and
[`getframes`](https://github.com/jacotay7/getframes) for detector response and
noise; neither model is reimplemented here. It runs on NumPy by default and
switches to CUDA (via CuPy) with a single configuration field.

## Install

```bash
python -m pip install makewfs            # CPU
python -m pip install "makewfs[gpu]"     # + CuPy for CUDA 12.x
```

## Quickstart

```python
import makewfs

wfs = makewfs.WavefrontSensor.from_toml("wfs.toml")
frame = wfs.expose(opd_m, seed=0)  # getframes.Frame, data in ADU

rate = wfs.photon_rate(opd_m)  # ideal photons/s/native detector pixel
frame = wfs.expose_integrated(samples)  # several temporal OPD samples, one exposure
```

Construct the sensor once and call `expose()` with each new residual wavefront:
the only per-frame inputs are the wavefront and an optional noise seed, because
telescope, source, sensor, sampling and camera choices all live in versioned
configuration. Set `device = "gpu"` under `[numerics]` and pass a CuPy OPD array
for the fully device-resident path; `frame.data` is then a CuPy ADU array, and
`getframes.to_numpy(frame.data)` marks an intentional host boundary.

Runnable starter configurations live at
[`examples/configs/shack_hartmann_minimal.toml`](examples/configs/shack_hartmann_minimal.toml)
and [`examples/configs/pyramid_minimal.toml`](examples/configs/pyramid_minimal.toml).

See **[Quickstart](https://jacotay7.github.io/makewfs/quickstart/)** for the full
walkthrough, **[Configuration](https://jacotay7.github.io/makewfs/configuration/)**
for every field with its units and range, and
**[Concepts](https://jacotay7.github.io/makewfs/concepts/)** plus
**[Units and coordinates](https://jacotay7.github.io/makewfs/units-and-coordinates/)**
if you are new to wavefront sensing.

## Benchmarks

Warm end-to-end frame throughput on an AMD Ryzen 9 9950X3D and an NVIDIA RTX
5090, including optics, `getframes` detector noise, truth and ADU, with
device-resident input/output and no host transfers. The raw artifact and its
invocation are [versioned with the benchmarks](benchmarks/device-results.json):

| Workflow | Output | Work samples | CPU (frames/s) | GPU (frames/s) | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20×20 SH, float32 | 160×160 | 1 | 125 | 1,949 | 15.60× |
| 60×60 SH, float64 | 360×360 | 1 | 18 | 893 | 48.51× |
| Broadband/range-sampled LGS SH | 64×64 | 9 | 186 | 778 | 4.19× |
| Pyramid, 8-point modulation, float32 | 80×80 | 8 | 668 | 1,635 | 2.45× |
| Pyramid, 32-point modulation, float64 | 108×108 | 32 | 25 | 918 | 36.68× |

Higher is better. A tiny unmodulated 54×54 pyramid case remains CPU-faster
(3,541 versus 1,583 frames/s) because GPU launch overhead dominates its small
optical workload. Compatible sampled-DFT Shack–Hartmann geometries are compiled
on first use, so warm each fixed sensor before measuring or entering a real-time
loop. Reproduce the table with

```bash
python benchmarks/run.py --device both --frames 100 \
  $(printf -- '--config %s ' benchmarks/configs/*.toml)
```

See the [full snapshot](benchmarks/device-results.md) and the
**[performance guide](https://jacotay7.github.io/makewfs/performance/#cpu-versus-gpu-reference-throughput)**
for the methodology.

## Features

- **Shack–Hartmann sensors** — lenslet geometry with physical sampling controls
  (`spot_sampling_pixels_per_lambda_over_d`, `minimum_illuminated_fraction`),
  flux-conserving pixel integration, and explicit crop accounting; see
  **[Shack-Hartmann](https://jacotay7.github.io/makewfs/shack-hartmann/)**.
- **Four-face pyramid sensors** — fixed-mask pyramid with circular modulation
  (radius and sample count) and configurable pupil separation; see
  **[Pyramid](https://jacotay7.github.io/makewfs/pyramid/)**.
- **Broadband and finite sources** — deterministic spectral and angular
  quadrature, measured source curves, and user-supplied angular kernels, so
  extended and polychromatic beacons are integrated as intensities, not fields.
- **Laser guide stars** — sodium-range sampling with launch-position geometry
  and the resulting spot elongation; see
  **[Guide stars](https://jacotay7.github.io/makewfs/guide-stars/)**. The
  mean-altitude LGS OPD approximation is documented and deliberately distinct
  from range-resolved turbulence.
- **Analytic pupils** — central obscuration, segmented and rotated apertures,
  static OPD maps and user masks, hashed for provenance.
- **Configuration as the contract** — immutable TOML with units in every key
  name, unknown-key and range rejection with path-specific messages, a versioned
  schema, and digests of every referenced file; see
  **[Configuration](https://jacotay7.github.io/makewfs/configuration/)**.
- **Precision you choose** — matched `float32/complex64` and `float64/complex128`
  paths, tested on both, with no silent promotion in the hot path.
- **GPU-optional, end to end** — `numerics.device = "gpu"` runs optics *and* the
  `getframes` detector device-resident, with CPU parity tests; compatible
  Shack–Hartmann geometries use a specialized CUDA execution plan and fall back
  automatically to the readable reference implementation for any unsupported
  optical feature.
- **Real-time friendly** — construct once and reuse; static grids, masks and
  normalization are cached on the sensor, and `expose()`/`expose_integrated()`
  accept a caller-owned `out=` array so high-rate loops control storage lifetime.
- **Built to interoperate** — `pyturb` supplies the atmosphere and `getframes`
  the detector, including wavelength-resolved QE; see
  **[Interoperability](https://jacotay7.github.io/makewfs/interop/)**.
- **Validated against theory** — piston invariance, non-negative intensity, flux
  accounting, photon-rate linearity, stable sign/axis conventions and sampling
  convergence are asserted quantitatively; see
  **[Validation](https://jacotay7.github.io/makewfs/validation/)**.
- **Stable surface** — `load_config`, `WavefrontSensor` and `simulate` are frozen
  under [SemVer](https://jacotay7.github.io/makewfs/stability/) as of 1.0.

See the **[API reference](https://jacotay7.github.io/makewfs/api/)** for every
public function and class, the
**[documentation gallery](https://jacotay7.github.io/makewfs/gallery/)** and
**[runnable examples](examples/)** for sensor comparisons, modulation trades, LGS
elongation and closed-loop injection, and
**[Troubleshooting](https://jacotay7.github.io/makewfs/troubleshooting/)** when a
frame does not look the way you expect.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and
[AGENTS.md](AGENTS.md). Run the checks locally with:

```bash
ruff check . && ruff format --check . && python -m mypy && pytest
```

## License

MIT — see [LICENSE](LICENSE).
