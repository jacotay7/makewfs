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

## Input wavefront RMS in frame metadata

Every frame records two RMS values of the input wavefront, both in OPD metres
(phase input is converted at `input.reference_wavelength_m` first). Both
describe the input array as given, without `input.static_opd_path`; for
`expose_integrated` they describe the mean of the temporal samples.

| Key | Definition |
| --- | --- |
| `wfs_input_opd_rms_m` | Pupil-weighted, piston-removed RMS, the `rms` of aocore CONVENTIONS 4.1: `sqrt(sum a^2 (opd - <opd>_a)^2 / sum a^2)`, where `<opd>_a` is the intensity-weighted mean. |
| `wfs_input_opd_rms_unweighted_m` | `sqrt(mean(opd^2))` over every pixel of the input grid, with piston included and pixels outside the pupil counted. This was `wfs_input_opd_rms_m` before 2.0. |

The weight `a^2` is the intensity of the pupil the optics use, on the input
grid. The engines use the configured pupil as a field amplitude, so the
intensity is its square. An analytic pupil (diameter, obscuration, spiders,
segment gaps, rotation) is evaluated on `input.shape` with the same
`numerics.pupil_supersampling`; this is the map
`WavefrontSensor.pupil_illumination()` returns, computed in float64. A
`telescope.custom_mask_path` mask exists only on the engine's own pupil grid
(`lenslets_across_pupil` times the pupil samples per lenslet for a
Shack-Hartmann, `pixels_across_pupil` for a pyramid), so its intensity is
area-averaged onto the input grid: each input
pixel takes the mean over the mask cells it overlaps, which is exact when the
two shapes match. A piston, or OPD outside the pupil, therefore changes only
the unweighted value.

Both values are reduced on the selected device and cross to the host together
with the captured photon rate in one transfer.

## Closed-loop use

The package intentionally stops at the detector image. A downstream controller
may turn that image into slopes, a reconstruction, and a deformable-mirror
command, then feed the resulting residual OPD back into `expose()`.
