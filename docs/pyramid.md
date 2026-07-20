# Pyramid wavefront sensor

The four-face pyramid path propagates each source wavelength through a
piecewise-linear focal-plane phase mask and returns the four re-imaged pupils
as one photon-rate mosaic. Circular modulation is represented by several source
tilts whose intensities are averaged before the detector exposure. Wavelength,
finite NGS angular, and source-spectrum quadrature are incoherent intensity
sums; sodium range elongation remains SH-specific.

```toml
[sensor]
kind = "pyramid"
wavelength_m = 700e-9

[pyramid]
pixels_across_pupil = 64
pupil_separation_pixels = 80
modulation_radius_lambda_over_d = 2.0
modulation_samples = 16
```

The shipped `examples/configs/pyramid_minimal.toml` is a complete detector
configuration. The output shape is
`pixels_across_pupil + pupil_separation_pixels` in each dimension, plus any
configured detector margins. `pupil_separation_pixels` is the centre-to-centre
spacing of adjacent pupil images, so keep it larger than
`pixels_across_pupil` for distinct pupils. A separation smaller than the pupil
diameter is allowed and intentionally produces overlapping pupil images; the
propagation does not assign pixels to a single pupil. Pyramid reconstruction remains
outside this repository.

The propagation is verified internally against a direct-summation forward and
inverse DFT on a random small grid. Optional HCIPy and OOPAO comparisons exercise
the flat reference plus tip, tilt, and focus push/pull response maps. HCIPy's
detector plane is rotated by 180 degrees for the comparison because its named
viewing convention is opposite to makewfs's fixed face order; this is an explicit
coordinate transform, not a fitted image registration. See [Validation](validation.md)
for quantitative tolerances and reproduction commands.
