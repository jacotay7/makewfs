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

The release clean-room check installs non-editable `makewfs` distributions with
released `getframes>=2.1.1` and `pyturb>=1.0` in an isolated environment. It
imports the installed packages, builds a small SH configuration, renders an
ideal rate map, exposes one seeded detector frame, and verifies
wavelength-resolved detector truth.
