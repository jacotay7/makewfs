# CPU/GPU end-to-end WFS throughput

Generated from `benchmarks/device-results.json`. Higher frames/s is better; timings are local evidence.

- Platform: `Linux-6.17.0-35-generic-x86_64-with-glibc2.39`
- CPU: `AMD Ryzen 9 9950X3D 16-Core Processor`
- GPU: `NVIDIA GeForce RTX 5090`
- Dependencies: makewfs 0.1.0.dev0, numpy 2.2.6, scipy 1.16.3, cupy 13.6.0, getframes 2.1.0, pyturb 1.0.0
- Method: one persistent sensor, warm device-resident OPD and output, detector truth enabled, a distinct seed per frame, construction and host transfers excluded, CUDA synchronized.

| Configuration | Sensor | Output | Work samples | CPU (frames/s) | GPU (frames/s) | Speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| shack_hartmann_20x20_float32.toml | shack_hartmann | 160x160 | 1 | 82.9 | 1,068.5 | 12.89x |
| shack_hartmann_60x60_float64.toml | shack_hartmann | 360x360 | 1 | 16.7 | 665.6 | 39.90x |
| shack_hartmann_broadband_lgs.toml | shack_hartmann | 64x64 | 9 | 74.0 | 215.6 | 2.91x |
| pyramid_40_float32.toml | pyramid | 54x54 | 1 | 2,872.0 | 874.8 | 0.30x |
| pyramid_60_mod8_float32.toml | pyramid | 80x80 | 8 | 441.5 | 815.4 | 1.85x |
| pyramid_80_mod32_float64.toml | pyramid | 108x108 | 32 | 28.1 | 603.5 | 21.47x |
