# CPU/GPU end-to-end WFS throughput on an Arm host

Frames/s, higher is better (warm detector frames: optics plus detector).
These come from `benchmarks/device-results-neoverse-n1-rtx4060.json` and
`benchmarks/device-results-neoverse-n1-rtxa400.json`. They use the same
method and configurations as [device-results.md](device-results.md), whose
x86 + RTX 5090 numbers are repeated here for reference.

- Host: cfl-test-bench, an 80-core Ampere Neoverse-N1 (aarch64), shared.
  Every run was pinned to 16 cores (`taskset -c 16-31`) with 16 BLAS
  threads, so CPU rows can vary with the frequency governor.
- GPUs: NVIDIA GeForce RTX 4060 (8 GB) and NVIDIA RTX A400 (4 GB), driver 580.
- Dependencies: makewfs 2.0.0, NumPy 2.5.3, SciPy 1.18.1, CuPy 14.2.0,
  getframes 2.4.0, pyturb 2.2.0.
- Method: one persistent sensor, warm device-resident OPD and output,
  detector truth enabled, a distinct seed per frame, construction and host
  transfers excluded, CUDA synchronized.

| Configuration | Output | Neoverse-N1 CPU (16 cores) | RTX A400 | RTX 4060 | Ryzen 9 9950X3D CPU | RTX 5090 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pyramid_40_float32.toml | 54x54 | 944.4 | 367.5 | 409.1 | 3,526.4 | 1,604.7 |
| pyramid_60_mod8_float32.toml | 80x80 | 178.4 | 377.1 | 418.1 | 659.0 | 1,645.9 |
| pyramid_80_mod32_float64.toml | 108x108 | 11.6 | 50.6 | 206.3 | 24.3 | 923.6 |
| shack_hartmann_20x20_float32.toml | 160x160 | 40.8 | 242.7 | 513.4 | 126.5 | 2,041.8 |
| shack_hartmann_60x60_float64.toml | 360x360 | 10.5 | 50.3 | 200.6 | 17.9 | 899.3 |
| shack_hartmann_quadrature_9sample.toml | 64x64 | 53.8 | 183.8 | 194.8 | 186.0 | 780.4 |

Reading this:

- **The GPU pays off for every configuration except the smallest pyramid.**
  `pyramid_40` runs about 2x faster on the CPU, because at that size each
  frame is a few kernel launches.
- **The cards separate on the big, double-precision configurations.** On
  `shack_hartmann_60x60_float64` and `pyramid_80_mod32_float64`, the RTX 4060
  runs about 4x the RTX A400. On the small ones they are within about 10%,
  because both are launch-bound there.
- **16 Neoverse-N1 cores reach 0.27–0.59x a 16-core Ryzen 9 9950X3D** on
  these sensors, typically about 0.3x. The gap is smallest on the large
  double-precision configurations. The Ryzen numbers were measured with
  makewfs 1.0.
