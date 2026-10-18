# ADR 0006: per-frame pupil amplitude

Status: accepted (2.3.0)

## Context

ADR 0003 makes the wavefront the only physical per-frame input, and until 2.3
the wavefront was a phase/OPD map: the pupil amplitude was the configured,
static pupil. A field propagated through high turbulent layers also changes in
amplitude from frame to frame (scintillation), and a Shack-Hartmann subaperture
flux then measures that intensity. Studies of scintillation as a wavefront-
sensing signal, and of extreme-AO Shack-Hartmann sensors with a spatial filter,
could not be simulated. The roadmap allowed amplitude input only as a versioned
extension of the input contract.

Two forms were considered: a complex field, or an amplitude next to the OPD.
A complex field is monochromatic, so it does not define the field at the other
wavelengths of a broadband source, and its phase is wrapped, so it cannot be
interpolated onto another propagation grid; OPD metres are achromatic and are
the canonical wavefront quantity (ADR 0001).

## Decision

- `photon_rate`, `expose`, `expose_integrated`, `subaperture_flux`, and
  `simulate` accept an optional keyword `amplitude`; `expose_many` accepts an
  iterable `amplitudes`. It is the dimensionless field modulus relative to the
  unscintillated field, and multiplies the configured pupil amplitude of both
  sensors. The OPD remains the only phase input.
- The amplitude is one `input.shape` map for every wavelength or one map per
  wavelength node (`WavefrontSensor.wavelengths_m`), so chromatic
  scintillation is representable.
- Its intensity is area-averaged when the propagation grid differs, so the
  light each cell receives is conserved. An amplitude of exactly one reproduces
  the OPD-only result bit for bit on the CPU.
- The configured source rate describes amplitude one; a frame's flux scales
  with the pupil-weighted mean of the squared amplitude. Frames record
  `wfs_input_amplitude`.
- The spatial filter (`spatial_filter_width_lambda_over_d`) and the
  inter-lenslet cross-talk (`crosstalk_reach_subapertures`) are instrument
  geometry, so they are configuration keys, not per-call options.
- `WavefrontSensor.subaperture_flux` returns the ideal per-subaperture photon
  rates (detector pixel blocks and pupil-plane lenslet apertures) as a
  `SubapertureFlux` value in `makewfs.api`; the top-level exports are
  unchanged.

The amplitude is part of the wavefront, not instrument intent: like the OPD it
comes from the atmosphere and the caller's propagation, so it is a per-frame
input. The configuration and its digest still record every instrument choice.

## Consequences

- Existing calls and configurations are unchanged; the new configuration keys
  are omitted from the normalized configuration at their defaults, so digests
  do not move.
- makewfs still does not propagate fields between layers or model the
  atmosphere; callers (or `pyturb` and `aocore` propagators) produce the pupil
  field.
- Per-wavelength phases are not representable; the OPD is shared by every
  wavelength.
- On a GPU, frames with an amplitude use the array-operation Shack-Hartmann
  path rather than the compiled kernel; the pyramid replays a second CUDA graph.
