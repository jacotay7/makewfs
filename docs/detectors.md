# Detectors and radiometry

`makewfs` stops at incident photons. The detector adapter calls
`getframes.Camera.expose()` for scalar scenes, or the optional
`getframes.Camera.expose_spectral()` cube API when a QE curve is configured.
Exposure, camera preset, temperature, binning, precision, and seed come from
configuration.

This preserves the existing detector model: QE, photon shot noise, dark current,
read noise, EM/eAPD gain, fixed-pattern effects, saturation, digitization,
persistence, and truth metadata are not duplicated here.

Magnitude normalization uses public `getframes.Bandpass` and `getframes.Telescope`
radiometry. Direct detector-surface photon rates are the preferred way to isolate
WFS optical behavior in a trade study.

For broadband scenes whose spatial spectrum varies across the detector, set
`detector.qe_curve_path` to a two-column `wavelength_nm qe` curve. `makewfs`
keeps one optical photon-rate map per wavelength and calls
`getframes.Camera.expose_spectral` once. QE is applied exactly once inside
`getframes`; `FrameTruth.photon_rate` remains the integrated incident map while
`FrameTruth.spectral_photon_rate` and `wavelengths_nm` preserve the cube. Without
a QE curve, the existing scalar path is retained for compatibility. With a
released getframes camera that lacks `Camera.expose_spectral` (including 2.1.0),
makewfs applies the same QE-weighted electron map through a compatibility
fallback and records that integrated-only mode in metadata; full cube truth
requires the `Camera.expose_spectral` API landing in getframes 2.1.1.
