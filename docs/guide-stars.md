# Guide stars and source morphology

`makewfs` consumes a source photon budget; it does not predict laser return
power, sodium excitation, sky background, or atmospheric scintillation. This
keeps source/radiometry assumptions visible to an AO designer and leaves
detector effects to `getframes`.

## Natural guide stars

Use `normalization = "magnitude"` for an NGS when a bandpass and telescope
throughput should determine the photon budget. `getframes.Bandpass` and
`getframes.Telescope` provide the zero point, collecting area, obstruction, and
throughput calculation. Use direct detector-surface photons for a laboratory
calibration or a separately modeled source.

`field_angle_arcsec` applies a deterministic angular tilt. For an extended NGS,
`angular_fwhm_arcsec` and `angular_quadrature_order` form a two-dimensional
Gaussian quadrature around that centroid. Every angular state is propagated
independently and summed in intensity, preserving incoherence and total source
flux. A measured or otherwise user-defined morphology can instead be supplied
with `angular_kernel_path`, a three-column `x_arcsec y_arcsec weight` table;
kernel offsets are relative to `field_angle_arcsec` and are mutually exclusive
with Gaussian FWHM mode.

## Wavelength states

`wavelengths_m` and optional `wavelength_weights` form a normalized photon
quadrature. The Shack–Hartmann spot sampling scales with wavelength; the ideal
pyramid mask retains its configured fixed pupil separation. Without
`detector.qe_curve_path`, the resulting photon-rate maps are summed before one
scalar-QE `getframes` exposure. With that curve, the spectral-QE path passes the
cube to the released `getframes>=2.2.0` API. Detector QE is applied once per
wavelength and the incident spectral cube is retained in
`FrameTruth.spectral_photon_rate`.

For measured relative curves, `sed_path` and `transmission_path` point to
two-column text files with `wavelength_nm value`. If explicit wavelengths are
omitted, the curve knots become the quadrature grid; if they are supplied, the
curves are interpolated there and multiplied into the weights.

## Sodium LGS approximation

For `kind = "lgs"`, configure `detector_photon_rate_per_s`; magnitude
normalization is intentionally rejected. `lgs_ranges_m` and
`lgs_range_weights` describe a normalized sodium density quadrature, and
`lgs_launch_position_m` gives the launch point in pupil-plane metres. The current
Shack–Hartmann model treats the input OPD as the phase at the weighted mean
range and adds the geometric angular offset

```text
delta_theta = (launch_position - subaperture_position)
              * (1/range - 1/mean_range)
```

to each range state. A zero-thickness profile therefore reduces exactly to a
thin beacon, while a thicker profile elongates edge subaperture spots. The model
does not claim same-realization turbulent OPD at every sodium range; that would
require the conditional `pyturb` extension described in the roadmap.

## Fast LGS kernel mode

The quadrature above propagates one optical state per source-angle sample and
sodium range: with a 3 x 3 source rule and five ranges, 45 propagations per
wavelength per frame, which dominates a long closed loop. Each of those states
only adds a tilt across every subaperture, and by the Fourier shift theorem a
tilt only translates that subaperture's diffraction spot. For a fixed beacon
geometry the incoherent sum is therefore the zero-offset spot convolved with a
fixed kernel per subaperture `j`:

```text
K_j(theta) = sum_a sum_k w_a w_k G_sigma(theta - alpha_a - delta_theta_jk)
delta_theta_jk = (launch_position - subaperture_position_j)
                 * (1/range_k - 1/mean_range)
```

where `w_k` are the sodium range weights, `alpha_a, w_a` the optional
`angular_kernel_path` points, and `G_sigma` the Gaussian of `angular_fwhm_arcsec`.
This is the convolution description of elongated subaperture spots used by, for
example, Schreiber et al. (2009,
[doi:10.1111/j.1365-2966.2009.14797.x](https://doi.org/10.1111/j.1365-2966.2009.14797.x)).

`source.lgs_model = "kernel"` uses it. Only the wavelength states are
propagated, at `field_angle_arcsec`, and the kernel is applied as one batched
FFT convolution. It needs no new physical input: the kernel is built from the
same `lgs_ranges_m`, `lgs_range_weights`, `lgs_launch_position_m`,
`angular_fwhm_arcsec` or `angular_kernel_path`, and lenslet geometry as the
quadrature, once per sensor (and once per
[lenslet-grid offset state](shack-hartmann.md#lenslet-grid-offset-states)).
The mean-range OPD approximation is unchanged.

Numerical choices and their consequences:

- **Where the convolution acts.** On the oversampled focal-plane intensity,
  before the field stop, any focal-plane blur, charge diffusion, and pixel-area
  integration, exactly where the quadrature's translations act.
- **Truncation.** Light the kernel moves out of a subaperture's window is lost
  and reported in `wfs_captured_photons_s`; it is never renormalized. Light it
  moves *into* the window comes from a margin propagated around the window: up
  to the kernel half-width, but no further than one alias-free period of the
  sampled lenslet field (`samples_per_lenslet` lenslet `lambda/d`). Frames record
  both as `wfs_lgs_kernel_radius_px` and `wfs_lgs_kernel_margin_px`. When the
  margin is smaller than the radius, light from farther away is not moved in.
  The quadrature has the matching limit: a tilt beyond half that period aliases.
  For strongly elongated spots, raise `numerics.pupil_samples_per_lenslet` until
  `samples_per_lenslet` exceeds the window plus twice the streak half-length (in
  `lambda/d`); both models then agree.
- **Sub-sample translations** are deposited by linear (cloud-in-cell)
  interpolation on the focal grid, which conserves each translation's weight and
  first moment exactly and keeps the kernel non-negative, at the cost of a
  slight extra smoothing that shrinks with `numerics.fft_oversampling`.
- **Source extent** is the continuous Gaussian, sampled on the focal grid and
  truncated at four standard deviations, rather than the quadrature's discrete
  rule; `angular_quadrature_order` is unused.

Validation (`validation/lgs_kernel.py`, `tests/test_lgs_kernel.py`, and
[Validation](validation.md#fast-lgs-kernel-mode)): against a quadrature with a
densely sampled Gaussian source the kernel image agrees to 0.14% (L1) and
centroids to 0.002 pixel on a side-launched 20 x 20 sensor. Without source
extent the remaining difference is the linear deposition: 3.2% of the image at
`fft_oversampling = 2`, 0.7% at 4, and 0.2% at 8, and below 0.01 pixel in
centroid throughout. The default
three-point quadrature itself departs from the continuous Gaussian when the
source spans several pixels, so compare against a dense reference before
reading kernel-versus-quadrature differences as kernel error.

The shipped `examples/lgs_thin_beacon.py` intentionally demonstrates the thin
beacon/cone-effect boundary. A range-profile SH example is covered by the
configuration and regression tests; a full gallery remains a roadmap item.
