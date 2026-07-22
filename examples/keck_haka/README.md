# Keck II HAKA open-loop simulation

This example animates uncorrected Maunakea turbulence through a Keck II HAKA
natural-guide-star Shack-Hartmann model. It uses the repository boundaries
directly:

```text
V-normalized F6 V spectrum x Maunakea extinction x 3 aluminum mirrors
                       + pyturb Maunakea OPD
                                      |
                                      v
                        makewfs HAKA broadband optics
                                      |
                                      v
                         getframes spectral OCAM2K ADU
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
2.567 m diameter circle and a segment-aligned regular hexagon measuring 2.532 m
flat-to-flat (2.924 m corner-to-corner), offset by (+0.038, -0.025) m in pupil
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
Per-output dark/bias levels and relative conversion gains are inferred directly
from the supplied eng519 V=10.16 open-loop cube. The gain fit uses the total
estimated-dark-subtracted signal in subapertures that are at least 98%
illuminated and do not cross an output seam. Relative ADU/electron responses are
normalized to arithmetic mean one, so this models the visible seams without
applying a global flux correction.

No matched dark cube accompanies the RTC data. Its best available dark/bias
estimate is built exclusively from pixels outside the fitted pupil: an integer
histogram-mode template for each of the eight outputs and each phase of the
repeated 4x4 pattern, followed by a robust per-frame median drift correction for
each output. Every simulated science frame instead uses a reproducible,
exposure-matched 32-frame median master dark for its WATAO mode. Both panels are
therefore dark-subtracted floating-point counts and can contain negative values.
The manufacturer specifies 2067 fps full-frame maximum speed and about 0.4
input-referred electrons of mean read noise at 2000 fps and multiplication gain
near 600: [Andor OCAM2K
specifications](https://andor.oxinst.com/products/ocam-emccd-camera-series/ocam).
The preset keeps the detector's two saturation domains separate: Andor's
published 270,000-electron image-area well is applied before EM multiplication,
and the Keck-observed 10,000 dark-subtracted count ceiling is represented as a
280,000-electron output-register limit at the approximately 28 output
electrons/count system conversion. At EM gain 600, one input electron therefore
produces about 21.43 counts. This avoids the unphysical 25-count post-EM clip
that results from treating a single small input-referred number as the output
well.

The source is a 6600 K Planck photon spectrum normalized to the catalog Johnson
V magnitude and integrated from 400 to 950 nm with eight-point Gauss-Legendre
quadrature. Atmospheric attenuation uses the mean Mauna Kea extinction curve
in magnitudes per airmass published by [Gemini
Observatory](https://www.gemini.edu/observing/telescopes-and-sites/sites#MK%20optical%20extinction%20curve).
The eng519 header supplies `AIRMASS=1.01`; the same value is the showcase default.
Gemini's table ends at 900 nm, so its 0.05 mag/airmass value is conservatively
held constant from 900 to the 950 nm band edge.
The curve is applied before the telescope and the absolute photon budget uses
the 72.04 m² clear area measured from the same sampled segmented pupil used by
the optical propagation.

The primary, secondary, and tertiary are each assigned 0.88 scalar reflectivity,
for a combined telescope-mirror throughput of `0.88^3 = 0.6815`. Keck documents
that its telescope mirrors use [bare aluminum
coatings](https://keckobservatory.org/a_mirrors_perfect_reflection/), and
[maintained aluminum coating data](https://www.layertec.de/en/coatings/metallic-coatings/aluminum-coatings/)
place broadband visible/NIR reflectivity in roughly the 85--92% range. Thus
0.88 is a defensible conservative band average,
but not a measured reflectivity curve for these three mirrors. Downstream HAKA
throughput remains exactly one and is never fitted to the RTC data.

The JSON manifest contains a `frame_flux_audit` for every rendered exposure. It
records launched and optically captured photons, expected photoelectrons,
expected pre-saturation counts, and the noisy measured dark-subtracted count sum.
The spot sampling is 0.91 pixel per lambda/D, corresponding to the HAKA
57x0.75-arcsec/pixel scale at the 673 nm reference wavelength and a 10.95/54 m
illuminated subaperture. Each wavelength's spot then scales physically across
400--950 nm before incoherent intensity summation. Any remaining finite-window
crop loss is reported in the manifest and is never renormalized away.

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
unmodified real/simulated lenslet-signal ratio. Source signal is measured
identically in both dark-subtracted cubes as the central 2x2 sum minus four
times the surrounding 12-pixel border mean in every lenslet, making the flux
comparison insensitive to small residual offsets. The simulation uses its
normal exposure-matched master dark.

The catalog V magnitude sets the absolute normalization of the 6600 K spectrum;
B-V=0.46 is retained as supporting evidence for the F5--F7 classification. The
400--950 nm passband is currently a top hat because a measured HAKA instrumental
curve was not supplied. Atmospheric transmission and detector QE are both
wavelength resolved. Downstream HAKA throughput is one, so the simulation is
expected to be brighter than the instrument after the three modeled telescope
reflections. The measured real/simulation ratio is reported and never fed back
as a scale factor.

For the checked-in deterministic run, the photon-weighted atmospheric
transmission is 90.53%. The 72.04 m² clear pupil receives 270.16 million
photons/s after atmospheric extinction; the three mirror reflections reduce
this to 184.11 million photons/s, and the finite SH windows capture 181.77
million photons/s. The real and simulated central-2x2 spot fractions are 94.1%
and 92.4%, respectively. The pedestal-insensitive lenslet signal is 0.848
million count/frame in the real cube and 2.956 million count/frame in the
simulation: a real/simulation ratio of 0.2870, reported but not applied. Because
atmospheric extinction and telescope-mirror losses are already included, this
is an approximate 28.70% remaining downstream HAKA-throughput diagnostic. With
the three modeled mirror reflections included, the inferred post-atmosphere
telescope-plus-HAKA throughput is 19.56%; including atmospheric extinction gives
17.71% from above the atmosphere to the detector entrance. These estimates are
still convolved with the estimated real dark, absolute detector calibration,
and the unknown instrumental band shape.

`WSFRRT1` is the default rate column; select `--frame-rate-column WSFRRT2` to
render the alternate rates. Magnitude bins are lower-inclusive and
upper-exclusive, so magnitude 10.0 selects WATAO row 10 and magnitude 15.0 row
1. The animation playback rate is only for viewing; each simulated exposure
uses the physical inverse lookup-table frame rate.

## CPU versus GPU open-loop throughput

Run the warm end-to-end HAKA benchmark with:

```bash
python examples/keck_haka/benchmark.py
```

This constructs one persistent sensor and non-periodic `pyturb` atmosphere per
device, performs untimed warm-up frames, and then measures five wall-clock
seconds per device. Each timed frame advances Mauna Kea turbulence by the
physical 1/750 s exposure, propagates all eight wavelengths through the 57x57
Shack--Hartmann optics, and exposes the noisy 228x228 OCAM2K detector. Source SED
quadrature, pupil/sensor/atmosphere construction, warm-up, host copies,
calibration, plotting, and file output are excluded. GPU work is synchronized
after every five-frame timing batch.

The July 2026 run on an AMD Ryzen 9 9950X3D and NVIDIA RTX 5090 measures 28.3
CPU frames/s and 401.1 GPU frames/s, a 14.2x speedup. Relative to the 750 Hz RTC
cadence these are 0.038x and 0.535x real time. The generated
`haka_cpu_gpu_benchmark.gif` shows the two OCAM2K streams over equal wall-clock
playback: atmosphere clocks and displayed frame counters advance at the measured
device rates, so the GPU stream visibly evolves faster. Intermediate detector
frames are skipped only in the visualization, after timing is complete. GIF and
PNG outputs are ignored by Git; the exact samples, invocation, machine,
dependencies, and methodology are in
[`haka_cpu_gpu_benchmark.json`](haka_cpu_gpu_benchmark.json). These are local
measurements rather than portable performance guarantees. Change the timed
interval or device selection with `--seconds` and `--device cpu|gpu|both`.

The older 945.2 GPU frames/s result for
`shack_hartmann_60x60_float64.toml` is not the same workload. It uses fixed zero
OPD, one wavelength, and a generic CMOS detector. A controlled 200-frame GPU
ablation on this machine gives:

| Warm GPU workload | Frames/s | Time/frame |
| --- | ---: | ---: |
| Existing 60x60, monochromatic zero OPD + generic CMOS | 945.2 | 1.058 ms |
| HAKA geometry, monochromatic zero OPD + OCAM2K | 1,241.9 | 0.805 ms |
| HAKA geometry, eight wavelengths, optics only | 600.3 | 1.666 ms |
| HAKA geometry, eight wavelengths, fixed OPD + OCAM2K | 433.9 | 2.304 ms |
| HAKA geometry, eight wavelengths, evolving `pyturb` + OCAM2K | 405.7 | 2.465 ms |

The HAKA geometry itself is not slower: its monochromatic case exceeds the old
reference because it uses a 228x228 float32 output instead of 360x360 float64.
The eight wavelength-resolved optical states are the dominant added cost. The
OCAM2K spectral detector and per-frame metadata add the next-largest share, and
the non-periodic atmosphere adds about 0.11 ms/frame. Increasing the CUDA timing
batch from five to 100 frames measured 398.2 frames/s, so synchronization cadence
does not explain the gap.

## Explicit assumptions

- Magnitudes are interpreted as catalog Vega V. Every showcase star uses a
  6600 K F6 V spectrum over a top-hat 400--950 nm band. The eng519 exposure uses
  its header airmass of 1.01 and the mean measured Mauna Kea extinction curve.
  Primary, secondary, and tertiary reflections transmit 0.88 each; downstream
  HAKA throughput is exactly 1.0, so the predicted flux should still exceed a
  real OCAM2K frame.
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
  total flux. A matched dark cube and uniform flat would improve the absolute
  detector calibration.
- The OCAM2K preset is calibrated around its 2000 fps, gain-near-600 operating
  point. Its QE, output read noise, CIC, bias, ADC, image-area well, and output
  register limit are held fixed while the supplied EM gain changes; this is a
  mode-comparison simulation, not a per-mode detector calibration.
- `OBWNNAME` and `bkgnd` are retained in the table and manifest. `bkgnd=1` is not
  converted into a photon background because its physical units/meaning were not
  supplied. Magnitudes 5--15 all use the `open` filter rows.
