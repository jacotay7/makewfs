# CPU/GPU end-to-end WFS throughput

Generated from `benchmarks/device-results.json`. Higher frames/s is better; timings are local evidence.

- Platform: `Linux-6.17.0-35-generic-x86_64-with-glibc2.39`
- CPU: `AMD Ryzen 9 9950X3D 16-Core Processor`
- GPU: `NVIDIA GeForce RTX 5090`
- Dependencies: makewfs 0.1.0.dev0, numpy 2.2.6, scipy 1.16.3, cupy 13.6.0, getframes 2.1.0, pyturb 1.0.0
- Method: one persistent sensor, warm device-resident OPD and output, detector truth enabled, a distinct seed per frame, construction and host transfers excluded, CUDA synchronized.

| Configuration | Sensor | Output | Work samples | CPU (frames/s) | GPU (frames/s) | Speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| shack_hartmann_20x20_float32.toml | shack_hartmann | 160x160 | 1 | 143.4 | 2,011.2 | 14.03x |
| shack_hartmann_60x60_float64.toml | shack_hartmann | 360x360 | 1 | 26.3 | 945.2 | 35.94x |
| shack_hartmann_broadband_lgs.toml | shack_hartmann | 64x64 | 9 | 99.8 | 557.1 | 5.58x |
| pyramid_40_float32.toml | pyramid | 54x54 | 1 | 3,569.8 | 1,541.1 | 0.43x |
| pyramid_60_mod8_float32.toml | pyramid | 80x80 | 8 | 668.5 | 1,579.4 | 2.36x |
| pyramid_80_mod32_float64.toml | pyramid | 108x108 | 32 | 33.3 | 903.9 | 27.18x |
