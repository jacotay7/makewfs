# Interoperability

## pyturb

`pyturb.Atmosphere.frames()` yields OPD arrays in metres. Construct a
`WavefrontSensor` once and pass each OPD to `expose()` or use
`expose_integrated()` for multiple atmosphere samples within one detector
integration. `makewfs` does not import or simulate the atmosphere internally.

The runnable `examples/moving_atmosphere.py` script demonstrates this boundary,
including the explicit `pyturb.to_numpy` device-to-host conversion before the
CPU optical path. `examples/lgs_thin_beacon.py` uses the same pattern with
`lgs_altitude`; range-resolved sodium elongation is not silently inferred from
that single OPD map.

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

The optional `interop` test marker exercises one real `pyturb.Atmosphere.frames`
OPD sample when pyturb is installed:

```bash
python -m pip install -e '.[interop]'
python -m pytest -m interop
```

The clean-room smoke check has also been run from non-editable wheels for
`getframes 2.0.0`, `pyturb 1.0.0`, and `makewfs 0.1.0.dev0` in an isolated
Python 3.13 environment. It imports the installed packages, builds a small SH
configuration, renders an ideal rate map, and exposes one seeded detector frame.
The package constraints therefore currently record `getframes>=2.0` and
`pyturb>=1.0` as the minimum compatible sibling versions.
