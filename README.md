# makewfs

`makewfs` will turn a configured pupil-plane phase/OPD map into a realistic
adaptive-optics wavefront-sensor image. The first supported sensors will be
Shack–Hartmann and four-face pyramid sensors.

The package owns the wavefront-sensor optics. It deliberately reuses
[`pyturb`](https://github.com/jacotay7/pyturb) for atmospheric OPD and
[`getframes`](https://github.com/jacotay7/getframes) for detector response and
noise; neither model will be reimplemented here.

> **Status:** early development. The CPU Shack–Hartmann and four-face pyramid
> paths, configuration API, and detector handoff are implemented. Broadband
> source morphology and Shack–Hartmann LGS elongation are implemented. An
> optional wavelength-resolved detector-QE prototype is available with the
> unreleased `getframes` cube API; full range-resolved turbulence and a released
> spectral-QE contract remain roadmap work. A private optional CuPy optical path
> is available for parity experiments, but public GPU detector support is not
> advertised.

- [Complete implementation roadmap](ROADMAP.md)
- [Instructions for implementation agents](AGENTS.md)

The minimal API is:

```python
import makewfs

wfs = makewfs.WavefrontSensor.from_toml("wfs.toml")
frame = wfs.expose(opd_m, seed=0)  # getframes.Frame, data in ADU
```

The repository includes a runnable starter configuration at
[`examples/configs/shack_hartmann_minimal.toml`](examples/configs/shack_hartmann_minimal.toml).
The corresponding pyramid starter is at
[`examples/configs/pyramid_minimal.toml`](examples/configs/pyramid_minimal.toml).
For an ideal photon-rate map, use `wfs.photon_rate(opd_m)`; for one detector
exposure containing several temporal OPD samples, use
`wfs.expose_integrated(samples)`.

For closed-loop work, construct the sensor once and call `expose()` with each
new residual wavefront. The only per-frame inputs are the wavefront and,
optionally, a detector-noise seed; telescope, source, sensor, sampling, and
camera choices live in the configuration.
