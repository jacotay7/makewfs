# Shack-Hartmann sensor

The Shack-Hartmann engine partitions the configured pupil into a square lenslet
grid. Each subaperture is propagated by a batched Fraunhofer transform, integrated
onto its native detector pixels, and assembled into a
`(lenslet_y, lenslet_x)` mosaic. Integer-compatible focal grids use an FFT;
arbitrary normalized sampling uses a directly sampled DFT so the requested
plate scale is represented exactly.

`pixels_per_subaperture` and
`spot_sampling_pixels_per_lambda_over_d` determine the detector sampling. Partial
subapertures are retained and their illumination is recorded rather than
discarded. Finite detector windows can crop diffraction wings; the lost flux is
reported in metadata and is not silently renormalized.

Detector pixels use centered cell coordinates. For an even-sized subaperture
window, zero wavefront slope lies on the boundary shared by the central four
pixels. The Fourier propagation is evaluated at half-integer samples before
flux-conserving pixel-area integration, so a symmetric flat-wavefront spot gives
equal signal in those four pixels rather than being assigned to one of them.
Positive normalized sampling below one is supported for undersampled quadcell
modes whose native detector pixels are wider than `lambda/D`.

Set `numerics.pupil_supersampling` above one when edge-area convergence matters;
this averages analytic sub-pixels rather than interpolating a binary mask.

Instead of normalized spot sampling, a configuration may provide lenslet pitch,
lenslet focal length, detector pixel pitch, and relay magnification. The engine
derives the same normalized sampling from those physical fields and rejects
mixed modes. Supplying the measured lenslet pitch explicitly is important when
the telescope pupil is reduced onto a millimetre-scale hardware array.
The optional field stop, Gaussian or measured-kernel blur, and detector-margin
controls are applied to the ideal photon-rate mosaic before the camera adapter.

`lenslet_grid_rotation_deg` and `lenslet_grid_offset_fraction` describe a
rotated or decentered lenslet array relative to the entrance pupil. They use
physical-coordinate interpolation and are intended for instrument registration
studies; the zero-valued defaults retain the faster aligned-grid path.

The phase-ramp sign and spot displacement are fixed by analytic tests. This
repository produces the image; centroiding, slope extraction, and reconstruction
belong downstream.

Verification uses centroids only as a test observable: makewfs does not expose a
centroiding or reconstruction API. A direct-summation DFT checks random spot
mosaics, the analytic OPD-ramp test fixes displacement in detector pixels, and
optional HCIPy/OOPAO comparisons check multi-amplitude response gain, sign, and
cross-axis leakage on both axes. See [Validation](validation.md) for the numerical
tolerances and why the external-package gain is not expected to match machine
precision.

## Lenslet-field sampling

Each lenslet field is represented by `s` samples across the lenslet pitch `d`.
The Fraunhofer transform of a field sampled at pitch `d / s` is periodic, with
period `s` lenslet `lambda/d`, while the detector window spans

```text
W = pixels_per_subaperture / spot_sampling_pixels_per_lambda_over_d   [lambda/d]
```

If `W > s`, the window integrates more than one period: every replica of the
spot is summed as light, so the photon rate can exceed the configured source
rate, and a spot tilted beyond `+-s/2 lambda/d` aliases into the wrong position
because the phase step between samples exceeds pi. The continuous lenslet field
has neither artefact. Spot sampling scales with wavelength, so `W` is largest at
the shortest wavelength of a spectral quadrature, and a field stop of radius `r`
limits it to `2 r`.

The engine therefore propagates each lenslet at

```text
samples_per_lenslet = k * pupil_samples_per_lenslet,
k = smallest integer with k * pupil_samples_per_lenslet >= W_max
```

The configured pupil grid is kept: each pupil cell's area-weighted transmission
is held constant over its `k x k` refined sub-cells, so lenslet illumination,
validity, and a custom mask (supplied on the configured grid) are unchanged. The
OPD is interpolated linearly from the input grid onto the refined grid, which
reproduces a tilt exactly. Configurations that already satisfy `s >= W_max`
keep `k = 1` and are bit-for-bit unchanged. The resolved values are available as
`sensor.engine.pupil_samples_per_lenslet`, `field_upsampling`,
`samples_per_lenslet`, and `detector_window_lambda_over_d`.

`s >= W` is the minimum for an unaliased model, not a convergence guarantee. A
sampled field still carries a small amount of wing light from neighbouring
periods into the window; for fully illuminated square lenslets the captured
fraction is a few percent above the continuous `sinc^2` value when `s` is close
to `W` (up to about 7% at `s / W = 1.1`) and about 1% or less at `s = 4 W` in
the test cases (`tests/test_shack_hartmann_sampling.py`).
Raise `numerics.pupil_samples_per_lenslet` when wing flux matters, for example
for wide-field or quad-cell geometries; the cost grows with `s^2`.

## Current limitations

The CPU path supports deterministic wavelength quadrature, finite NGS angular
extent and user kernels, rotated analytic pupils with square segment gaps,
static path OPD, measured blur kernels, and a configurable sodium-range
elongation model for Shack–Hartmann. Lenslet-grid rotation/offset is supported
through the explicit resampling path. LGS return flux is always supplied by the
user; `makewfs` does not model laser propagation or sodium physics.
