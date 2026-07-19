# Concepts and conventions

## Wavefront units

OPD is the canonical internal quantity and is measured in metres. A phase input
must declare `quantity = "phase"`, `unit = "rad"`, and
`reference_wavelength_m`; it is converted to OPD before propagation. Units are
never inferred from array magnitude.

The input array uses `(y, x)` order. Its physical extent and shape come from the
`[input]` table. The pupil amplitude is configured separately, so a per-frame
input contains only the phase/OPD map.

## Image domains

The optical engines are deterministic and return an incident photon-rate map in
photons/s/native detector pixel. `getframes.Camera.expose()` performs the
scalar photon-to-electron-to-ADU chain, while the optional
`Camera.expose_spectral()` path applies wavelength-dependent QE exactly once
and preserves the incident cube in detector truth. Optical intensities are
summed over incoherent wavelength, source, modulation, and sodium-range samples;
complex fields are never added across incoherent states.

## Piston and sampling

A constant piston changes only the global complex phase and therefore cannot
change intensity. The numerical implementation removes the weighted global
piston before evaluating the complex exponential to keep this invariant stable
in single precision.

When an input grid does not divide into the configured lenslets, OPD is
resampled on physical coordinates. Wrapped phase is never interpolated.

## Closed-loop use

The package intentionally stops at the detector image. A downstream controller
may turn that image into slopes, a reconstruction, and a deformable-mirror
command, then feed the resulting residual OPD back into `expose()`.
