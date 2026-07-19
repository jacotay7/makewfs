# Detectors and radiometry

`makewfs` stops at incident photons. The detector adapter calls
`getframes.Camera.expose()` with the optical photon-rate map, exposure, camera
preset, temperature, binning, precision, and seed from configuration.

This preserves the existing detector model: QE, photon shot noise, dark current,
read noise, EM/eAPD gain, fixed-pattern effects, saturation, digitization,
persistence, and truth metadata are not duplicated here.

Magnitude normalization uses public `getframes.Bandpass` and `getframes.Telescope`
radiometry. Direct detector-surface photon rates are the preferred way to isolate
WFS optical behavior in a trade study.
