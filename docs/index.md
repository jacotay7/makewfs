# makewfs

`makewfs` converts a pupil-plane phase or OPD map into a realistic adaptive-
optics wavefront-sensor image. The first engines are Shack-Hartmann and
four-face pyramid sensors.

The ownership boundary is:

```text
pyturb OPD -> makewfs optical photon-rate map -> getframes detector Frame
```

`makewfs` does not generate atmosphere, detector noise, wavefront
reconstructions, deformable-mirror commands, or controllers. See the
[roadmap](https://github.com/jacotay7/makewfs/blob/main/ROADMAP.md) for the
implementation plan and the [agent guide](https://github.com/jacotay7/makewfs/blob/main/AGENTS.md)
for repository conventions.

## Current status

The initial CPU Shack–Hartmann and four-face pyramid paths are available,
including deterministic broadband/finite-source quadrature and the documented
Shack–Hartmann sodium-range elongation model. A private optional CuPy optical
path has CPU parity tests; public GPU detector support, broader independent
validation, and release work remain staged in the roadmap.

See the [worked examples](https://github.com/jacotay7/makewfs/tree/main/examples)
for atmosphere handoff, magnitude series, SH-versus-pyramid comparison, and
the explicitly thin-beacon LGS contract.
