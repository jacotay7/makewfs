# Reference benchmark snapshot

Generated from `benchmarks/reference-results.json`. Timings are local evidence, not CI promises.

- Python: `3.13.13 | packaged by Anaconda, Inc. | (main, Apr 14 2026, 06:19:41) [GCC 14.3.0]`
- Platform: `Linux-6.8.0-136-generic-x86_64-with-glibc2.39`
- Dependencies: makewfs 0.1.0.dev0, numpy 2.2.6, scipy 1.17.1, getframes 2.0.0, pyturb 1.0.0

| Configuration | Sensor | Shape | Dtype | States | Construction (ms) | Wavelengths | Ranges | Modulation | Warm optics (ms/frame) | Warm detector (ms/frame) | Optics (frames/s) | Detector (frames/s) | Python peak (MiB) |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| shack_hartmann_minimal.toml | shack_hartmann | 64x64 | float32 | 1 | 1743.357 | 1 | 1 | 1 @ 0.0 λ/D | 5.611 | 9.783 | 178.2 | 102.2 | 37.4 |
| pyramid_minimal.toml | pyramid | 84x84 | float32 | 1 | 2.764 | 1 | 1 | 1 @ 0.0 λ/D | 1.217 | 4.399 | 821.9 | 227.3 | 1.0 |
| shack_hartmann_20x20_float32.toml | shack_hartmann | 160x160 | float32 | 1 | 5.044 | 1 | 1 | 1 @ 0.0 λ/D | 23.962 | 27.877 | 41.7 | 35.9 | 45.5 |
| shack_hartmann_60x60_float64.toml | shack_hartmann | 360x360 | float64 | 1 | 10.814 | 1 | 1 | 1 @ 0.0 λ/D | 104.134 | 115.704 | 9.6 | 8.6 | 190.9 |
| shack_hartmann_broadband_lgs.toml | shack_hartmann | 64x64 | float32 | 9 | 2.931 | 3 | 3 | 1 @ 0.0 λ/D | 31.650 | 34.329 | 31.6 | 29.1 | 7.8 |
| pyramid_40_float32.toml | pyramid | 54x54 | float32 | 1 | 2.316 | 1 | 1 | 1 @ 0.0 λ/D | 0.991 | 4.010 | 1008.8 | 249.4 | 0.7 |
| pyramid_60_mod8_float32.toml | pyramid | 80x80 | float32 | 1 | 2.243 | 1 | 1 | 8 @ 2.0 λ/D | 2.613 | 5.939 | 382.7 | 168.4 | 2.8 |
| pyramid_80_mod32_float64.toml | pyramid | 108x108 | float64 | 1 | 2.382 | 1 | 1 | 32 @ 3.0 λ/D | 18.931 | 22.582 | 52.8 | 44.3 | 32.5 |
