# Performance and precision

Construct a `WavefrontSensor` once. Pupil masks, source quadrature, FFT geometry,
pyramid masks, and the `getframes.Camera` adapter are built at construction;
`photon_rate` and `expose` are the warm paths.

The CPU implementation uses SciPy batched FFTs and supports `fft_workers` plus
float32/complex64 or float64/complex128 optical arithmetic. Float32 is useful
for throughput and memory; float64 is the reference path. Benchmark cold
construction separately from warm frames and record Python, NumPy, SciPy, and
hardware versions.

No public GPU option exists yet. The backend helpers are the intended boundary
for a future CuPy optical implementation; the detector handoff is an explicit
host-side `getframes` boundary until that package supports device-resident
signal generation.

The repository benchmark runner separates cold construction, warm optical
frames, and warm detector frames:

```bash
python benchmarks/run.py --frames 10 --output benchmark-results.json
```

Benchmark JSON includes source-state count, output shape, environment, and
per-frame timings. Wall-clock values are evidence for local comparisons, not
portable CI performance promises.

Representative Shack–Hartmann configurations are provided under
`benchmarks/configs/`:

```bash
python benchmarks/run.py --representative --frames 3 --output benchmark-results.json
```

They cover 20×20 float32 and 60×60 float64 lenslet grids while keeping the
minimal SH and pyramid smoke cases in the default benchmark run.
