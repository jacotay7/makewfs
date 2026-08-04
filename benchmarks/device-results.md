# CPU/GPU end-to-end WFS throughput

Generated from `benchmarks/device-results.json`. Higher frames/s is better; timings are local evidence.

- Platform: `Linux-7.0.0-28-generic-x86_64-with-glibc2.39`
- CPU: `AMD Ryzen 9 9950X3D 16-Core Processor`
- GPU: `NVIDIA GeForce RTX 5090`
- Dependencies: makewfs 1.0.0, numpy 2.2.6, scipy 1.16.3, cupy 14.1.1, getframes 2.1.1, pyturb 1.0.0
- Method: one persistent sensor, warm device-resident OPD and output, detector truth enabled, a distinct seed per frame, construction and host transfers excluded, CUDA synchronized.

| Configuration | Sensor | Output | Work samples | CPU (frames/s) | GPU (frames/s) | Speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| shack_hartmann_20x20_float32.toml | shack_hartmann | 160x160 | 1 | 124.9 | 1,949.2 | 15.60x |
| shack_hartmann_60x60_float64.toml | shack_hartmann | 360x360 | 1 | 18.4 | 893.3 | 48.51x |
| shack_hartmann_broadband_lgs.toml | shack_hartmann | 64x64 | 9 | 185.8 | 778.4 | 4.19x |
| pyramid_40_float32.toml | pyramid | 54x54 | 1 | 3,540.5 | 1,582.6 | 0.45x |
| pyramid_60_mod8_float32.toml | pyramid | 80x80 | 8 | 667.5 | 1,635.1 | 2.45x |
| pyramid_80_mod32_float64.toml | pyramid | 108x108 | 32 | 25.0 | 918.0 | 36.68x |
