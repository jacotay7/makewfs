# Troubleshooting

## Unexpectedly cropped flux

Increase `pixels_per_subaperture`, `fft_oversampling`, or the pyramid pupil
separation/output margin. Cropping is reported in `wfs_captured_photons_s` and
is never silently renormalized.

## Shack–Hartmann flux exceeds the source rate or spots appear in the wrong place

Versions up to 1.1.0 aliased the lenslet field when the subaperture window in
`lambda/d` (`pixels_per_subaperture / spot_sampling_pixels_per_lambda_over_d`,
at the shortest wavelength) exceeded `numerics.pupil_samples_per_lenslet`.
Current versions refine the lenslet field automatically; see
[Lenslet-field sampling](shack-hartmann.md#lenslet-field-sampling).

## Spots have the wrong scale

Check `spot_sampling_pixels_per_lambda_over_d`, `sensor.wavelength_m`, and the
physical `input.grid_extent_m`. OPD is metres, not phase radians; phase inputs
must declare their reference wavelength.

## Noisy frames differ between runs

Pass an integer `seed` to `expose`. Unseeded calls intentionally advance the
`getframes` camera RNG. Inspect `frame.truth` to separate optical photon-rate
changes from detector noise.

## LGS image is not elongated

A thin LGS is intentionally point-like. Add `lgs_ranges_m` and normalized
`lgs_range_weights` to a Shack–Hartmann source. The current approximation uses
one mean-range OPD plus geometric elongation; it does not synthesize
range-resolved turbulence.
