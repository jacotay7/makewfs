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
columns in [`benchmarks/reference-table.md`](../benchmarks/reference-table.md).
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
