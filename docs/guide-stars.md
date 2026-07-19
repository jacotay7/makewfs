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
scalar-QE `getframes` exposure. With that curve, the optional spectral-QE path
passes the cube to `getframes` 2.1's development API (or applies an explicitly
documented integrated fallback on 2.0); the release gate remains in
`ROADMAP.md`.

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

The shipped `examples/lgs_thin_beacon.py` intentionally demonstrates the thin
beacon/cone-effect boundary. A range-profile SH example is covered by the
configuration and regression tests; a full gallery remains a roadmap item.
