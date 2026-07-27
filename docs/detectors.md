# Detectors and radiometry

`makewfs` stops at incident photons. The detector adapter calls
`getframes.Camera.expose()` for scalar scenes, or the
`getframes.Camera.expose_spectral()` cube API when a QE curve is configured.
Exposure, camera preset, temperature, binning, precision, and seed come from
configuration.

This preserves the existing detector model: QE, photon shot noise, dark current,
read noise, EM/eAPD gain, fixed-pattern effects, saturation, digitization,
persistence, and truth metadata are not duplicated here.

For a physical detector subarray, configure `[detector.roi]` with `left_px`,
`top_px`, `width_px`, and `height_px` in full-sensor native pixels. `makewfs`
passes the ROI-shaped photon-rate map and its full-detector origin to
`getframes`; amplifier seams and fixed detector structure therefore remain
registered to the camera preset rather than being reconstructed in the optical
model.

Magnitude normalization uses public `getframes.Bandpass` and `getframes.Telescope`
radiometry. Direct detector-surface photon rates are the preferred way to isolate
WFS optical behavior in a trade study.

For broadband scenes whose spatial spectrum varies across the detector, set
`detector.qe_curve_path` to a two-column `wavelength_nm qe` curve. `makewfs`
keeps one optical photon-rate map per wavelength and calls
`getframes.Camera.expose_spectral` once. QE is applied exactly once inside
`getframes`; `FrameTruth.photon_rate` remains the integrated incident map while
`FrameTruth.spectral_photon_rate` and `wavelengths_nm` preserve the cube. Without
a QE curve, the scalar path is retained. `makewfs>=1.0` requires
`getframes>=2.1.1`, the first released detector version with this spectral cube
and truth contract.
