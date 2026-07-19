# ADR 0004: backend boundary

Status: accepted

CPU NumPy/SciPy is the reference implementation. Centred FFTs, dtype pairing,
padding/cropping, and sampling live behind small numerical helpers. Sensors
operate on those helpers rather than scattering backend-specific transforms.
There is no pretend GPU option: a future CuPy backend must first reproduce the
CPU validation suite and document the explicit device-to-host boundary before
being exposed to users.
