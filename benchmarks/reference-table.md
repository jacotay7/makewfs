# Reference benchmark snapshot

Generated from `benchmarks/reference-results.json`. Timings are local evidence, not CI promises.

- Python: `3.12.0 | packaged by conda-forge | (main, Oct  3 2023, 08:43:22) [GCC 12.3.0]`
- Platform: `Linux-6.17.0-35-generic-x86_64-with-glibc2.39`
- Dependencies: makewfs 0.1.0.dev0, numpy 2.2.6, scipy 1.16.3, getframes 2.1.0, pyturb 1.0.0

| Configuration | Sensor | Shape | Dtype | States | Construction (ms) | Wavelengths | Ranges | Modulation | Warm optics (ms/frame) | Warm detector (ms/frame) | Optics (frames/s) | Detector (frames/s) | Python peak (MiB) |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| shack_hartmann_minimal.toml | shack_hartmann | 64x64 | float32 | 1 | 848.956 | 1 | 1 | 1 @ 0.0 λ/D | 2.025 | 3.852 | 493.9 | 259.6 | 41.6 |
| pyramid_minimal.toml | pyramid | 144x144 | float32 | 1 | 3.742 | 1 | 1 | 1 @ 0.0 λ/D | 1.238 | 3.931 | 807.7 | 254.4 | 6.0 |
| shack_hartmann_20x20_float32.toml | shack_hartmann | 160x160 | float32 | 1 | 2.018 | 1 | 1 | 1 @ 0.0 λ/D | 14.910 | 17.965 | 67.1 | 55.7 | 45.3 |
| shack_hartmann_60x60_float64.toml | shack_hartmann | 360x360 | float64 | 1 | 3.899 | 1 | 1 | 1 @ 0.0 λ/D | 92.769 | 99.810 | 10.8 | 10.0 | 191.9 |
| shack_hartmann_broadband_lgs.toml | shack_hartmann | 64x64 | float32 | 9 | 1.960 | 3 | 3 | 1 @ 0.0 λ/D | 15.202 | 16.981 | 65.8 | 58.9 | 7.8 |
| pyramid_40_float32.toml | pyramid | 54x54 | float32 | 1 | 1.423 | 1 | 1 | 1 @ 0.0 λ/D | 0.615 | 2.503 | 1627.1 | 399.6 | 1.2 |
| pyramid_60_mod8_float32.toml | pyramid | 80x80 | float32 | 1 | 1.532 | 1 | 1 | 8 @ 2.0 λ/D | 2.469 | 4.219 | 405.0 | 237.0 | 8.8 |
| pyramid_80_mod32_float64.toml | pyramid | 108x108 | float64 | 1 | 1.848 | 1 | 1 | 32 @ 3.0 λ/D | 47.016 | 49.918 | 21.3 | 20.0 | 118.5 |
