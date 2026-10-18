# Scintillation and subaperture flux

Since 2.3 a frame can carry a pupil-plane **amplitude** as well as the OPD, so
the Shack-Hartmann and pyramid images include scintillation: an intensity
pattern that changes from frame to frame. The Shack-Hartmann sensor also
returns the **flux of each subaperture**, so the flux can be used as a
measurement, and has two optional optical controls used in extreme-AO
Shack-Hartmann designs: a **focal-plane spatial filter** in front of the
lenslets and **inter-lenslet cross-talk** (no field stop between
subapertures). [ADR 0006](adr/0006-pupil-amplitude-input.md) records why the
amplitude is a per-frame input.

## Example

```python
import numpy as np
import makewfs

wfs = makewfs.WavefrontSensor.from_toml("examples/configs/shack_hartmann_scintillation.toml")
# U: complex pupil field at the sensing wavelength (from your propagation code)
# opd_m: the OPD your propagation tracks for that field (metres, input.shape)
amplitude = np.abs(U)  # relative to the unscintillated field
departure = np.angle(U * np.exp(-2j * np.pi * opd_m / wfs.config.sensor.wavelength_m))
opd_m = opd_m + departure * wfs.config.sensor.wavelength_m / (2 * np.pi)

frame = wfs.expose(opd_m, amplitude=amplitude, seed=0)  # noisy ADU frame
flux = wfs.subaperture_flux(opd_m, amplitude=amplitude)  # ideal, per lenslet
flux.detector_photons_s  # (n, n) photons/s in each subaperture's pixels
flux.pupil_photons_s  # (n, n) photons/s entering each lenslet
flux.valid  # (n, n) static valid-subaperture map
```

`examples/scintillation.py` runs the whole chain: a `pyturb` phase screen at
12 km, Fresnel propagation to the pupil with `aocore.AngularSpectrumPropagator`,
a deformable mirror emulated by removing the OPD below the lenslet Nyquist
frequency, then the image, the subaperture fluxes, and illustrative
centre-of-gravity slopes with and without the amplitude and behind a
`lambda / d` spatial filter. With the default 16 x 16 configuration
(scintillation index 0.36 at 700 nm) the pupil-plane subaperture flux
correlates with the pupil intensity at 1.00, the flux in the 4 `lambda / d`
detector windows at 0.60 (the window loses light that the 149 nm residual
spreads beyond it), and the flux behind the spatial filter at 0.02.

## Pupil amplitude

`photon_rate`, `expose`, `expose_many` (`amplitudes=`), `expose_integrated`,
`subaperture_flux`, and `simulate` accept an optional `amplitude`. It is the
dimensionless modulus of the incoming pupil field relative to the
unscintillated field, `A = |U| / |U_0|`, and it multiplies the configured pupil
amplitude. At wavelength `lambda` the field entering the sensor is

```text
pupil(x, y) * A_lambda(x, y) * exp(2 pi i OPD(x, y) / lambda)
```

- **Units and sign.** `A` has no units and must be finite and non-negative; a
  complex array is rejected. The pupil intensity scales with `A^2`. Arrays
  are `(y, x)` like the OPD: a bright feature at `+x` lands at larger column
  index on the lenslet grid and in each of the four pyramid pupil images.
- **Normalization.** The configured source rate (direct or from a magnitude)
  describes `A = 1`. A frame's photon rate is that rate times the
  pupil-weighted mean of `A^2` (times the captured fraction), so scintillation
  moves the total flux as it does on the sky. Normalize a simulated field by
  its long-term mean intensity, not frame by frame, to keep that fluctuation.
- **Shapes.** An `input.shape` map applies to every wavelength. A stack of
  shape `(len(wfs.wavelengths_m), *input.shape)` gives one map per wavelength
  node, in the order of `WavefrontSensor.wavelengths_m` (the spectral cube's
  order), for chromatic scintillation. `expose_integrated` takes one entry per
  temporal sample (a sequence, or an array whose leading axis is the sample
  axis).
- **Resampling.** The map is defined on the input grid. When the propagation
  grid differs, its intensity is area-averaged onto the new cells with exact
  overlaps and the square root taken, so each cell receives the integral of
  the input intensity over it. A rotated or offset lenslet grid then samples it
  at the transformed coordinates by linear interpolation, as it does the OPD.
  Supplying the amplitude on the propagation grid itself
  (`input.shape = lenslets_across_pupil * pupil_samples_per_lenslet`) avoids
  any resampling.
- **Exactness.** An amplitude of exactly one gives the OPD-only result bit for
  bit on the CPU, including resampled, rotated and offset grids, temporal
  integration, broadband and LGS quadrature, and the pyramid. On a GPU an
  amplitude routes a Shack-Hartmann frame to the array-operation reference
  path, because the compiled CUDA kernel takes the OPD only; that path agrees
  with the compiled one to rounding (2e-12 relative in float64 in the tests).
  A GPU pyramid replays a second CUDA graph for frames with an amplitude.
- **Metadata.** Frames record `wfs_input_amplitude` (true when an amplitude was
  given). `wfs_input_opd_rms_m` and `wfs_input_opd_rms_unweighted_m` keep
  their definitions and use the configured pupil, not the amplitude.

The OPD remains the only phase input because it is achromatic and can be
interpolated without wrapping. A complex field at the sensing wavelength is
represented exactly by `A = |U|` and an OPD whose phase equals `arg U` there,
as in the example above: adding the small angle between `U` and the field of
the OPD your propagation already tracks needs no phase unwrapping. Supply such
an OPD on the propagation grid, since linearly interpolating it across a
`2 pi` jump of that angle would be wrong. At other wavelengths of a broadband
source the phase is then `2 pi OPD / lambda`, which neglects the chromatic part
of the diffractive phase.

## Subaperture flux

`WavefrontSensor.subaperture_flux(wavefront, amplitude=None, offset_state=0)`
renders one frame and returns a `SubapertureFlux` with `(lenslet_y,
lenslet_x)` arrays on the sensor's device:

| Field | Meaning (photons/s, ideal) |
| --- | --- |
| `detector_photons_s` | Photon rate collected by each subaperture's own `pixels_per_subaperture` square block of pixels: the block sum of `photon_rate` without the detector margin. It loses what the spot sends beyond the block and, with cross-talk, gains what neighbours send in. |
| `pupil_photons_s` | Photon rate entering each lenslet's clear aperture: the source rate times the integral of the entrance-field intensity over the lenslet (after the spatial filter and the lenslet fill factor), divided by the unscintillated pupil's integral. It does not depend on the detector. |
| `valid` | The static valid-subaperture map of the offset state. |
| `photon_rate` | The ideal image the detector flux was summed from. |

For a uniform pupil `A = 1` the `pupil_photons_s` sum to the source rate.
Both fluxes are noise-free. A measured flux is the same block sum of an
exposed frame (`m` is `detector_margin_pixels`, `p` the pixels per
subaperture, `n` the lenslets across):

```python
data = np.asarray(frame.data, dtype=float)[m : m + n * p, m : m + n * p]
measured = data.reshape(n, p, n, p).sum(axis=(1, 3))
```

Dividing the measured flux by that of a flat-field (amplitude-one) frame
removes the static pattern of partly illuminated and edge subapertures.

## Focal-plane spatial filter

```toml
[shack_hartmann]
spatial_filter_width_lambda_over_d = 1.0
```

A square stop in a focal plane in front of the lenslet array, as in
extreme-AO Shack-Hartmann sensors (L. A. Poyneer and B. Macintosh,
*Spatially filtered wave-front sensor for high-order adaptive optics*, JOSA A
21, 810, 2004), low-passes the whole pupil field before the lenslets sample
it. Its full width is `c lambda_ref / d`: `c` is the key, `lambda_ref` is
`sensor.wavelength_m`, and `d` is the lenslet pitch on the pupil
(`input.grid_extent_m / lenslets_across_pupil`). `c = 1` passes exactly the
spatial frequencies the lenslets can sample without aliasing.

- The stop passes the pupil-plane spatial frequencies
  `|f_x|, |f_y| < c / (2 d) * lambda_ref / lambda`: it is a fixed angle, so a
  longer wavelength passes fewer cycles across the pupil. A frequency exactly
  on the edge is blocked.
- The entrance field (pupil times amplitude times phase, in the lenslet
  frame, so the square is aligned with the lenslets) is zero-padded to twice
  its size, Fourier transformed, restricted to the passband, transformed back,
  and cropped to the lenslet array. Only the passband lines are computed. The
  lenslet fill-factor mask applies after the filter.
- The stop is centred on the optical axis. An off-axis or finite source is
  filtered off-centre, as it would be by a physical stop.
- Light the stop removes, and light it diffracts beyond the lenslet array, is
  lost and reported in `wfs_captured_photons_s` and `pupil_photons_s`; nothing
  is renormalized.
- The filter turns residual phase into intensity: a phase ripple beyond the
  cutoff loses its diffracted orders, so the lenslet flux drops by
  `1 - J0(a)^2` for a ripple of `a` radians (checked in the tests). Behind the
  filter the subaperture flux therefore measures the residual phase as much as
  the scintillation.
- It is not available with range-resolved LGS states (`source.lgs_ranges_m`),
  whose per-subaperture tilts are not one pupil field a stop could act on.

## Inter-lenslet cross-talk

Without cross-talk each lenslet's spot is propagated, integrated, and cropped
to its own `pixels_per_subaperture` window, which acts as a field stop at the
window: light beyond it is lost (`subaperture_field_of_view_arcsec()` reports
the window's field). An instrument without such a stop lets that light fall on
the neighbouring subapertures' pixels:

```toml
[shack_hartmann]
crosstalk_reach_subapertures = 1
detector_margin_pixels = 8      # optional: keep the spill of edge lenslets
```

With a reach `k` every lenslet is propagated over a window `2k + 1`
subaperture windows wide on each axis, centred on its own, and every part of it
is added into the pixels it falls on. The image is the overlap-add of those
windows at the lenslet pitch (checked against independently propagated wide
windows in the tests). The total detected light equals the light inside each
lenslet's widened window while it stays on the detector; what falls beyond the
reach or off the detector (including any margin) is lost and reported. A
flat-wavefront square lenslet keeps `(2/pi Si(2 pi a))^2` of its light within
`+-a lambda/d`: 0.902 in a 4 `lambda/d` window, 0.967 with `k = 1` around it.

The lenslet field is sampled finely enough for the widened window not to
alias (`samples_per_lenslet >= (2k + 1) * pixels / sampling`), so a reach
raises the propagation cost. A configured circular field stop is applied over
the widened window. Cross-talk is not available with
`source.lgs_model = "kernel"`; the range quadrature model supports it.

## Limitations

- makewfs does not propagate the field: the amplitude and the OPD describe the
  field at the pupil, computed by the caller's atmosphere and propagation code.
- One OPD serves every wavelength: per-wavelength amplitudes are supported,
  per-wavelength phases are not.
- Overlapping light from neighbouring lenslets is added as intensity. The
  coherent interference between neighbouring spots, which depends on their
  relative phase and washes out over a spectral band or a finite source, is
  not modelled.
- The spatial filter is square, centred on the axis, and uses a fixed two-fold
  zero padding.
- On a GPU, frames with an amplitude, a spatial filter, or cross-talk use the
  array-operation Shack-Hartmann path rather than the compiled kernel (about
  3 ms instead of 0.7 ms per frame for a 20 x 20 float32 sensor on a Quadro
  P620; see [Performance](performance.md)).
- `subaperture_flux` is Shack-Hartmann only. The pyramid takes an amplitude
  but has no subapertures.
