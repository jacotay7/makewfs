# Pyramid wavefront sensor

The four-face pyramid path propagates a monochromatic pupil through a
piecewise-linear focal-plane phase mask and returns the four re-imaged pupils
as one photon-rate mosaic. Circular modulation is represented by several source
tilts whose intensities are averaged before the detector exposure.

```toml
[sensor]
kind = "pyramid"
wavelength_m = 700e-9

[pyramid]
pixels_across_pupil = 64
pupil_separation_pixels = 20
modulation_radius_lambda_over_d = 2.0
modulation_samples = 16
```

The shipped `examples/configs/pyramid_minimal.toml` is a complete detector
configuration. The output shape is
`pixels_across_pupil + pupil_separation_pixels` in each dimension. Broadband
propagation, finite-source convolution, LGS range structure, and pyramid
reconstruction are deliberately later roadmap items.
