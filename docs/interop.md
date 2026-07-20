# Interoperability

## pyturb

`pyturb.Atmosphere.frames()` yields OPD arrays in metres. Construct a
`WavefrontSensor` once and pass each OPD to `expose()` or use
`expose_integrated()` for multiple atmosphere samples within one detector
integration. `makewfs` does not import or simulate the atmosphere internally.

The runnable `examples/moving_atmosphere.py` script demonstrates the default CPU
boundary. `examples/lgs_thin_beacon.py` uses the same pattern with
`lgs_altitude`; range-resolved sodium elongation is not silently inferred from
that single OPD map.

For the public CUDA path, construct `pyturb` with `device="gpu"` and set
`numerics.device = "gpu"` in the makewfs TOML:

```python
import pyturb
import makewfs

atmosphere = pyturb.Atmosphere.from_profile(
    "mauna-kea", seeing=0.7, diameter=8.0, n=512, device="gpu", seed=1
)
wfs = makewfs.WavefrontSensor.from_toml("gpu_wfs.toml")

for _, opd_m_gpu in atmosphere.frames(dt=1e-3, steps=1000):
    frame = wfs.expose(opd_m_gpu)
    adu_gpu = frame.data
```

No array crosses the host boundary in that loop. GPU-capable `getframes` is
required; an older detector package produces an actionable construction error.

## getframes

`wfs.expose()` returns the existing array-like `getframes.Frame`. On CPU,
`frame.data` is NumPy; on GPU it is CuPy. `np.asarray(frame)`,
`getframes.to_numpy(frame.data)`, and `frame.to_fits()` are explicit host-facing
operations. `frame.truth` follows the selected device.

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
`getframes 2.1.0.dev0`, `pyturb 1.0.0`, and `makewfs 0.1.0.dev0` in an isolated
Python 3.13 environment. It imports the installed packages, builds a small SH
configuration, renders an ideal rate map, and exposes one seeded detector frame.
The scalar package constraint remains `getframes>=2.0` and `pyturb>=1.0`. Full
spectral cube truth needs the `Camera.expose_spectral` API arriving in
getframes 2.1.1 (already on `main`); released `getframes 2.1.0` does not provide
it, so those releases use the documented QE-weighted integrated-signal fallback
until 2.1.1 ships and can be pinned.
