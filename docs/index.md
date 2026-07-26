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

Version 1.0 provides stable CPU and optional public CuPy Shack–Hartmann and
four-face pyramid paths, deterministic broadband/finite-source quadrature,
wavelength-resolved detector QE, and the documented Shack–Hartmann sodium-range
elongation model. Analytic, direct-DFT, HCIPy, and interoperability checks are
described in the validation guide.

See the [worked examples](https://github.com/jacotay7/makewfs/tree/main/examples)
for atmosphere handoff, magnitude series, SH-versus-pyramid comparison, and
the explicitly thin-beacon LGS contract.
