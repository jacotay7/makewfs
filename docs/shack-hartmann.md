# Shack-Hartmann sensor

The Shack-Hartmann engine partitions the configured pupil into a square lenslet
grid. Each subaperture is propagated by a batched Fraunhofer FFT, integrated onto
its native detector pixels, and assembled into a `(lenslet_y, lenslet_x)` mosaic.

`pixels_per_subaperture` and
`spot_sampling_pixels_per_lambda_over_d` determine the detector sampling. Partial
subapertures are retained and their illumination is recorded rather than
discarded. Finite detector windows can crop diffraction wings; the lost flux is
reported in metadata and is not silently renormalized.

The phase-ramp sign and spot displacement are fixed by analytic tests. This
repository produces the image; centroiding, slope extraction, and reconstruction
belong downstream.

## Current limitations

The CPU path supports deterministic wavelength quadrature, finite NGS angular
extent, static path OPD, and a configurable sodium-range elongation model for
Shack–Hartmann. Physical relay parameters, field stops, and measured optical
blur remain staged in the roadmap. LGS return flux is always supplied by the
user; `makewfs` does not model laser propagation or sodium physics.
