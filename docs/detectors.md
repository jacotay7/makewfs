# Detectors and radiometry

`makewfs` stops at incident photons. The detector adapter calls
`getframes.Camera.expose()` for scalar scenes, or the
`getframes.Camera.expose_spectral()` cube API when a QE curve is configured.
Exposure, camera preset, temperature, binning, precision, readout mode, and seed
come from configuration.

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

## Correlated double sampling

Nondestructive-readout IR arrays — SAPHIRA in a C-RED One, and the hybrid arrays
pyramid sensors are usually built around — are normally operated in correlated
double sampling rather than as simple integrators. Set
`detector.readout_mode = "cds"` to select it:

```toml
[detector]
preset = "first_light_imaging_cred_one"
# Read-to-read integration, NOT the frame period. The C-RED One reads at up to
# 3500 full frames/s, and CDS spends two of those reads per delivered frame, so
# the fastest CDS frame rate is 1750 Hz with a 1/3500 s integration between the
# pedestal and signal reads. The remaining half of the 571 us period is the
# reset and pedestal read, which collect no signal -- the 50% duty cycle is a
# real photon cost of CDS and must not be modelled away by writing 1/1750 here.
exposure_s = 0.000285714  # 1/3500 s
temperature_c = -188.55
binning = 1
readout_mode = "cds"
```

The adapter then calls `getframes.Camera.correlated_double_sample()` — or
`correlated_double_sample_spectral()` when a QE curve is configured — which
resets the array, reads the pedestal, integrates for `exposure_s`, reads again,
and returns the difference. Ownership is unchanged: `makewfs` still supplies only
a photon-rate map, and every noise term stays in `getframes`.

Two consequences matter for downstream AO software:

- **The frame is signed.** CDS data is `int32` and bias-subtracted by
  construction, so a dark pixel may be negative. Slope kernels that assume
  unsigned ADU, or that clip at zero, need to know this.
- **A small pedestal survives.** Differencing removes kTC noise and fixed bias
  structure, but the interval-proportional bias rate scales with integration
  time rather than with the read, so it does not cancel. Subtract a dark CDS
  frame at the same exposure and gain. See the `getframes` noise-model guide for
  the full term-by-term accounting.

CDS is incompatible with `detector.binning > 1` (there is no charge-domain
binning stage in that readout path) and with caller-owned `out` storage (the
difference is freshly allocated); both are rejected with an explicit message
rather than silently ignored.
