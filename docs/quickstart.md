# Quickstart

Install `makewfs` together with the detector package:

```bash
python -m pip install makewfs
```

Build one configured sensor and feed it a pupil OPD in metres:

```python
from pathlib import Path

import numpy as np
import makewfs

config = makewfs.load_config(Path("examples/configs/shack_hartmann_minimal.toml"))
wfs = makewfs.WavefrontSensor(config)
opd_m = np.zeros(config.input.shape)

ideal_rate = wfs.photon_rate(opd_m)  # photons/s/native detector pixel
frame = wfs.expose(opd_m, seed=0)    # getframes.Frame; data are ADU
```

The same minimal API accepts the shipped four-face pyramid configuration:

```python
pyramid = makewfs.WavefrontSensor.from_toml("examples/configs/pyramid_minimal.toml")
pyramid_frame = pyramid.expose(opd_m, seed=0)
```

The sensor object caches pupil masks and FFT geometry. In a closed-loop driver,
keep it alive and call `expose()` once per residual OPD. For a single exposure
that contains several temporal phase samples, call `expose_integrated()`; it
averages ideal intensities and makes one detector read.

`pyturb` already returns OPD in metres, so the integration is direct:

```python
import pyturb

atmosphere = pyturb.Atmosphere.from_profile(
    "paranal-median", seeing=0.8, diameter=8.0, n=128, seed=1
)
for _, opd_m in atmosphere.frames(dt=config.detector.exposure_s, steps=10):
    frame = wfs.expose(opd_m)
```

All detector QE, shot noise, read noise, gain, saturation, and digitization are
provided by `getframes`.
