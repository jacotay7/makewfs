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

The initial CPU Shack-Hartmann path is available. Pyramid support, broader
source models, validation gallery, and GPU work are staged in the roadmap.
