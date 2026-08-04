# CPU/GPU end-to-end WFS throughput

Generated from `benchmarks/device-results.json`. Higher frames/s is better; timings are local evidence.

- Platform: `Linux-7.0.0-28-generic-x86_64-with-glibc2.39`
- CPU: `AMD Ryzen 9 9950X3D 16-Core Processor`
- GPU: `NVIDIA GeForce RTX 5090`
- Dependencies: makewfs 1.0.0, numpy 2.2.6, scipy 1.16.3, cupy 14.1.1, getframes 2.1.1, pyturb 1.0.0
- Method: one persistent sensor, warm device-resident OPD and output, detector truth enabled, a distinct seed per frame, construction and host transfers excluded, CUDA synchronized.

| Configuration | Sensor | Output | Work samples | CPU (frames/s) | GPU (frames/s) | Speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| shack_hartmann_20x20_float32.toml | shack_hartmann | 160x160 | 1 | 126.5 | 2,041.8 | 16.13x |
| shack_hartmann_60x60_float64.toml | shack_hartmann | 360x360 | 1 | 17.9 | 899.3 | 50.11x |
| shack_hartmann_quadrature_9sample.toml | shack_hartmann | 64x64 | 9 | 186.0 | 780.4 | 4.20x |
| pyramid_40_float32.toml | pyramid | 54x54 | 1 | 3,526.4 | 1,604.7 | 0.46x |
| pyramid_60_mod8_float32.toml | pyramid | 80x80 | 8 | 659.0 | 1,645.9 | 2.50x |
| pyramid_80_mod32_float64.toml | pyramid | 108x108 | 32 | 24.3 | 923.6 | 37.94x |
