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
