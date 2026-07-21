# Keck II HAKA open-loop simulation

This example animates uncorrected Maunakea turbulence through a Keck II HAKA
natural-guide-star Shack-Hartmann model. It uses the repository boundaries
directly:

```text
pyturb Maunakea OPD -> makewfs HAKA optics -> getframes OCAM2K ADU
```

The exact eight-output geometry, measured amplifier responses, and separate
image/output-well fields currently require the sibling `getframes` checkout;
they are not part of the published 2.1 package yet. The normal makewfs CI gate
remains conditional on those fields until that sibling change is released.

The geometry is 57 x 57 subapertures with 4 x 4 native detector pixels per
subaperture, giving the requested 228 x 228 OCAM2K ROI. The generated pupil has
the 36 physical primary segments (the central hexagon is absent) and is scaled
to the official 10.95 m maximum diameter. The real RTC cube shows that this
diameter spans about 54 of the 57 lenslets, leaving edge subapertures partially
or wholly dark; the simulated OPD grid is padded to reproduce that support. The
official Keck guide describes the 36 segments, their 1.8 m corner-to-corner size,
and the 10.95 m maximum primary diameter: [Keck Telescope
and Facility Instrument Guide](https://www2.keck.hawaii.edu/observing/kecktelgde/ktelinstupdate.pdf).
The 3 mm segment-gap value follows the [NASA Exoplanet Exploration Program
segmented-aperture reference](https://exoplanets.nasa.gov/exep/files/exep/SCDAApertureDocument050416.pdf).
The 2.6 m secondary scale and six 26 mm support-arm widths begin from HCIPy's
maintained `make_keck_aperture`, whose documentation cites validation against
Keck internal simulation efforts. The supplied RTC pupil further constrains the
secondary shadow. After correcting the eight relative amplifier responses, a
fit to the 4x4 core-minus-border subaperture illumination favors the union of a
2.567 m diameter circle and a segment-aligned regular hexagon measuring 2.529 m
flat-to-flat (2.920 m corner-to-corner), offset by (+0.035, -0.025) m in pupil
(x, y). This captures the circular secondary mirror between the hexagonal
segment/baffle limits. The support pattern is rotated by 30 degrees so it has a
vertical rather than horizontal arm, as in the RTC image. Reproduce the fit and
its residual image with:

```bash
python examples/keck_haka/fit_secondary.py
```

The detector starts from the public `getframes` `andor_ocam2k` preset and changes
the ROI and lookup-table EM gain. OCAM2K's eight outputs are modeled as four rows
by two columns. The full 240-pixel detector has 60-row by 120-column regions; the
centered 228-pixel RTC crop therefore splits at y=[54, 114, 174] and x=[114].
Per-output bias offsets and relative conversion gains are measured directly from
the supplied eng519 V=10.16 open-loop cube. The gain fit uses the median signal in
subapertures that are at least 98% illuminated and do not cross an output seam.
Relative ADU/electron responses are normalized to arithmetic mean one, so this
models the visible seams without applying a global flux correction.

Every science frame is reduced with a reproducible, exposure-matched 32-frame
median master dark for its WATAO mode. The resulting panels are bias/dark-subtracted
floating-point counts, so read noise can produce negative pixels. The manufacturer
specifies 2067 fps full-frame maximum speed and about 0.4 input-referred electrons
of mean read noise at 2000 fps and multiplication gain near 600: [Andor OCAM2K
specifications](https://andor.oxinst.com/products/ocam-emccd-camera-series/ocam).
The preset keeps the detector's two saturation domains separate: Andor's
published 270,000-electron image-area well is applied before EM multiplication,
and the Keck-observed 10,000 dark-subtracted count ceiling is represented as a
100,000-electron output-register limit at 10 electrons/count. This avoids the
unphysical 25-count post-EM clip that results from treating a single small
input-referred number as the output well.

The JSON manifest contains a `frame_flux_audit` for every rendered exposure. It
records launched and optically captured photons, expected photoelectrons,
expected pre-saturation counts, and the noisy measured dark-subtracted count sum.
The spot sampling is 0.91 pixel per lambda/D, corresponding to the HAKA
57x0.75-arcsec/pixel scale at 673 nm and a 10.95/54 m illuminated subaperture.
This keeps most of the diffraction core in the central four quadcell pixels, as
seen in the reference cube. Any remaining finite-window crop loss is reported in
the manifest and is never renormalized away.

Run the full 5--15 magnitude sweep:

```bash
python examples/keck_haka/simulate.py
```

This writes `examples/keck_haka/keck_haka.gif` and a JSON provenance manifest.
Displayed frames advance the generated frozen-flow atmosphere by at least
1/30 s even when the physical bright-star exposure is only 0.5 ms. The physical
exposure is unchanged; this display cadence makes every magnitude in the sweep
show atmospheric evolution. Change it with `--atmosphere-step-s`.
MP4 is also supported when FFmpeg is installed:

```bash
python examples/keck_haka/simulate.py --output /tmp/keck_haka.mp4
```

A short smoke render is useful while iterating:

```bash
python examples/keck_haka/simulate.py \
  --magnitudes 5 10 15 --frames-per-magnitude 1 \
  --samples-per-exposure 1 --output /tmp/keck_haka.gif
```

Compare the supplied 750-frame RTC cube to an unscaled simulation of eng519
(ICRS 21:27:41.910 +15:18:23.00, epoch 2000.0, V=10.16, B-V=0.46) at its
EM x600, 750 Hz operating point:

```bash
python examples/keck_haka/compare_real.py
```

This writes `real_vs_simulation.gif`, `real_vs_simulation.png`, and a JSON report.
Both GIF panels use the same unscaled color normalization. The simulation runs
every 1/750 s exposure and evolves the generated `pyturb` phase screen through
the nine intervening exposures before retaining every tenth frame, exactly like
the RTC telemetry decimation. The default GIF shows 150 paired retained frames
(2 seconds of telescope time); `--animation-frames 750` renders the entire cube.
Metrics always use every simulated and reference frame. The report contains the raw-cube hash, target,
generated-phase-screen settings, per-output pedestal/noise measurements,
relative-gain derivation, real and simulated 4x4 spot morphology, and the
unmodified real/simulated total signal ratio. Because no matched real dark was
supplied, the real pedestal is estimated from pixels outside the pupil
independently for each output and each global 4x4 lenslet phase. The simulation
uses its normal exposure-matched master dark.

The catalog V magnitude sets the comparison photon budget. B-V is preserved in
the manifest but is not used to synthesize an unknown HAKA open-filter throughput
curve; image morphology remains monochromatic at the measured 673 nm guide
wavelength. Source throughput is one, so the simulation is expected to be
brighter than the instrument. The measured real/simulation ratio is reported and
never fed back as a scale factor.

For the checked-in deterministic run, the real and simulated central-2x2 spot
fractions are 94.0% and 92.9%. Mean signed signal is 1.015 million count/frame
in the real cube and 5.642 million count/frame in the throughput-unity
simulation: a real/simulation ratio of 0.180, reported but not applied. The
simulation launches 71.19 million photons/s. Thus no missing-flux correction is
hidden here—the no-throughput simulation is 5.56 times brighter than the real
instrument at this catalog V magnitude.

`WSFRRT1` is the default rate column; select `--frame-rate-column WSFRRT2` to
render the alternate rates. Magnitude bins are lower-inclusive and
upper-exclusive, so magnitude 10.0 selects WATAO row 10 and magnitude 15.0 row
1. The animation playback rate is only for viewing; each simulated exposure
uses the physical inverse lookup-table frame rate.

## Explicit assumptions

- The user-supplied magnitudes are interpreted as Vega R magnitudes because the
  lookup did not identify a photometric system. The sensing morphology is
  monochromatic at 673 nm (matching `GUIDWAVE=0.673 um` in the supplied Keck II
  header). Source throughput is exactly 1.0: no instrument transmission loss is
  modeled, so the predicted flux should exceed a real OCAM2K frame.
- Seeing defaults to 0.65 arcsec at 500 nm. `pyturb` supplies its traceable
  `mauna-kea` profile and all wavefronts are open loop: no residual scaling,
  reconstructor, DM, or controller is applied.
- Each exposure averages three atmospheric OPD samples by default. Exposure is
  assumed to be exactly `1 / frame_rate`, with no dead time.
- Each WATAO mode gets an exposure-matched master dark made by median-combining
  32 simulated OCAM2K dark frames. Change this with `--master-dark-frames`.
- The 3 mm primary gaps, fitted circle-plus-hexagon central shadow, and six 26 mm support arms
  are included. The gaps and arms are finer than a 228-sample pupil and therefore
  enter as supersampled fractional pixels. The 54-lenslet pupil diameter and
  0.91-pixel-per-lambda/D spot sampling are constrained by the supplied RTC cube.
  The secondary fit uses only relative live subaperture illumination and does
  not alter global flux. The support arms are in the transposed orientation
  visible in that cube.
  HAKA relay pupil rotation/distortion, lenslet defects, and measured
  non-common-path aberrations are not included.
- The eight relative amplifier responses are inferred from equal-flux, fully
  illuminated subapertures in the supplied star cube. This is a detector flat
  model even though the operational system does not apply a flat. Their global
  response normalization is fixed to one; it is not fitted to the simulation's
  total flux. A matched dark and uniform flat would improve the absolute detector
  calibration.
- The OCAM2K preset is calibrated around its 2000 fps, gain-near-600 operating
  point. Its QE, output read noise, CIC, bias, ADC, image-area well, and output
  register limit are held fixed while the supplied EM gain changes; this is a
  mode-comparison simulation, not a per-mode detector calibration.
- `OBWNNAME` and `bkgnd` are retained in the table and manifest. `bkgnd=1` is not
  converted into a photon background because its physical units/meaning were not
  supplied. Magnitudes 5--15 all use the `open` filter rows.
