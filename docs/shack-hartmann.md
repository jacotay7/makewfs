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

## Lenslet-grid offset states

`shack_hartmann.lenslet_grid_offset_states_fraction` pre-declares positions of
the lenslet grid on the pupil, as `[x, y]` offsets in lenslet pitches added to
`lenslet_grid_offset_fraction`. Dithered, modulated, or otherwise time-varying
Shack-Hartmann schemes declare the positions they visit; the sensor builds each
state's geometry once and selects it per call:

```python
wfs = makewfs.WavefrontSensor.from_toml("examples/configs/shack_hartmann_offset_states.toml")
wfs.lenslet_grid_offsets_fraction()  # ((0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5))
frame = wfs.expose(opd_m, seed=0, offset_state=1)
valid = wfs.valid_subapertures(offset_state=1)
frames = wfs.expose_many(stream, offset_states=itertools.cycle(range(4)))
```

Each state caches its translated pupil illumination, lenslet mask,
valid-subaperture map, source and sodium-range tilts, flux normalization, LGS
kernel, and (on a GPU) compiled executor, so switching state costs a lookup.
Frames record `wfs_lenslet_offset_state` and the effective
`wfs_lenslet_grid_offset_fraction`. The index only selects geometry already in
the configuration, so the configuration and its digest still reproduce every
frame ([ADR 0005](adr/0005-lenslet-offset-states.md)). Memory grows with the
number of states times the number of propagated source states.

Moving the lenslet grid by `+f` pitches is equivalent to translating the pupil
illumination and the OPD by `-f` under a fixed grid. Both are sampled at the
transformed coordinates by linear interpolation. For the area-weighted pupil
this is its exact area-weighted translation on the sampled grid, so a lenslet's
illumination changes continuously with sub-sample offsets. Since 2.2 the pupil
is zero beyond the grid with interpolation across its last half sample, and
the OPD keeps its edge value there; previously each edge column of samples
dropped to zero at once, so illumination near a grid-filling pupil's edge moved
in steps of a whole sample column. Use `numerics.pupil_supersampling > 1` for
area-weighted pupil edges; with 8 sub-samples the translated lenslet
illumination matches the analytic area to better than 0.5% in the tests.

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

## Scintillation, flux, spatial filter, and cross-talk

A frame may carry a pupil amplitude next to the OPD, so the spots, the image,
and the subaperture fluxes include scintillation, and
`WavefrontSensor.subaperture_flux` returns the ideal photon rate of each
subaperture's pixels and of each lenslet's aperture. Two configuration keys
model extreme-AO designs: `spatial_filter_width_lambda_over_d`, a square
focal-plane stop in front of the lenslets, and `crosstalk_reach_subapertures`,
which removes the field stop at each window so light falls on the neighbouring
subapertures' pixels. All are described in
[Scintillation and subaperture flux](scintillation.md).

## Centroider choice for closed-loop use

makewfs forms images; it has no centroider, slope, or reconstruction API.
Those belong to downstream AO software (for example
[shmpipeline-ao](https://github.com/jacotay7/shmpipeline-ao)). When choosing a
centroider for a closed loop that consumes these images, these properties are
standard and worth checking against the measurement model the controller
assumes:

- A **linear centre of gravity over the full subaperture window with a fixed
  denominator** (the expected reference flux rather than the measured flux) is
  linear in the pixel values, so its mean under zero-mean detector noise equals
  its noise-free response. By the centroid theorem, the centroid of a fully
  captured spot equals the subaperture-averaged wavefront gradient (weighted by
  the pupil intensity in a partially illuminated subaperture), scaled by the
  plate scale.
- A **thresholded** centre of gravity is non-linear in the pixels. Its gain
  depends on flux and noise, so a gain calibrated on noise-free images is biased
  for noisy frames.
- A **weighted** (for example Gaussian-weighted) centre of gravity is not the
  subaperture-averaged gradient: its response depends on spot shape, and it
  differs from the average gradient when the wavefront varies within the
  subaperture, for instance when one subaperture spans several actuators.
- Dividing by the measured flux makes any centre of gravity non-linear, and
  window truncation (the field stop each subaperture window imposes, see
  `subaperture_field_of_view_arcsec()`) breaks the centroid theorem for spots
  that extend past the window, such as elongated LGS spots. With
  `crosstalk_reach_subapertures > 0` the window also collects light from
  neighbouring spots.

## Current limitations

The CPU path supports deterministic wavelength quadrature, finite NGS angular
extent and user kernels, rotated analytic pupils with square segment gaps,
static path OPD, measured blur kernels, and a configurable sodium-range
elongation model for Shack–Hartmann, with an equivalent fast
[kernel mode](guide-stars.md#fast-lgs-kernel-mode). Lenslet-grid rotation/offset,
including pre-declared offset states, is supported through the explicit
resampling path. A per-frame pupil amplitude, a square focal-plane spatial
filter, and inter-lenslet cross-talk are supported with the limitations listed
in [Scintillation and subaperture flux](scintillation.md#limitations). LGS
return flux is always supplied by the user; `makewfs` does not model laser
propagation, sodium physics, or the propagation that produces scintillation.
