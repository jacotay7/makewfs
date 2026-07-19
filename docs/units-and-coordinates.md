# Units and coordinates

- Input arrays are `(y, x)` NumPy arrays.
- OPD is metres. Phase is radians and requires
  `input.reference_wavelength_m`.
- `input.grid_extent_m` is the physical width/height represented by the input
  array; coordinates are centred on pixel centres.
- `source.field_angle_arcsec` is `[x, y]` on-sky angle.
- `lgs_launch_position_m` is `[x, y]` in the entrance-pupil plane.
- Optical output is photons/s/native detector pixel. Detector output is ADU.

The centred FFT helpers use unitary normalization. Piston removal is a numerical
stabilization only; it does not alter intensity. Detector cropping reports lost
flux rather than renormalizing the image.
