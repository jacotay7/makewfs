# Units and coordinates

`makewfs` follows the AO stack conventions defined by
[`aocore`](https://github.com/jacotay7/aocore) (`CONVENTIONS.md`), and
`tests/test_conformance.py` runs aocore's checks against both sensors. Pixel
centres, arcsecond/radian constants, OPD/phase conversion and flux-conserving
pixel binning come from `aocore` rather than local copies.

- Input arrays are `(y, x)` NumPy arrays.
- OPD is metres. Phase is radians and requires
  `input.reference_wavelength_m`.
- `input.grid_extent_m` is the physical width/height represented by the input
  array; coordinates are centred on pixel centres.
- `source.field_angle_arcsec` is `[x, y]` on-sky angle, converted with
  `aocore.ARCSEC_TO_RAD`.
- `lgs_launch_position_m` is `[x, y]` in the entrance-pupil plane.
- Optical output is photons/s/native detector pixel. Detector output is ADU.

The centred FFT helpers use unitary normalization. They keep the `fftshift`
convention (zero frequency on pixel `n // 2`) internally; the detector images
the sensors produce follow the stack's centring, with the optical axis at pixel
`(n - 1) / 2`, which the conformance tests check. Piston removal is a numerical
stabilization only; it does not alter intensity. Detector cropping reports lost
flux rather than renormalizing the image.
