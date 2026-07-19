# Interoperability

## pyturb

`pyturb.Atmosphere.frames()` yields OPD arrays in metres. Construct a
`WavefrontSensor` once and pass each OPD to `expose()` or use
`expose_integrated()` for multiple atmosphere samples within one detector
integration. `makewfs` does not import or simulate the atmosphere internally.

## getframes

`wfs.expose()` returns the existing array-like `getframes.Frame`. Use
`np.asarray(frame)` for ADU data, `frame.truth` for noise-free detector truth,
and `frame.metadata` for sensor/config provenance. Use `frame.to_fits()` for FITS
output when Astropy is installed.

## Closed-loop injection

The intended loop boundary is:

```python
residual_opd_m = controller_step(previous_frame)
frame = wfs.expose(residual_opd_m)
```

The controller and reconstructor in this sketch are deliberately external to
`makewfs`.
