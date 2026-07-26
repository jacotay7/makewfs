# Changelog

All notable changes to `makewfs` are documented here.

## [Unreleased]

## [1.0.0] - 2026-07-26

- Prepared the first stable public release with versioned package metadata,
  PyPI/CI badges, citation and release documentation, and a trusted-publishing
  GitHub Actions workflow using the `pypi` environment.
- Fixed the benchmark runner on Python 3.10 by using the portable
  `datetime.timezone.utc` API.
- Raised the detector dependency to released `getframes>=2.1.1`, made
  wavelength-resolved detector QE and full spectral truth part of the supported
  contract, and removed the pre-release integrated-signal compatibility path.
- Added an R-band HAKA camera-LUT analysis using representative A0 V through M3
  V continua and generated open-loop Maunakea states. It reports mean active
  4x4-lenslet intensity SNR with OCAM2K photon, EM-excess, dark, CIC, read, and
  quantization noise, fits a smooth ceiling-aware broken-power-law cadence floor
  only to the R>=10 fine-adjustment tail, asymptotes to the true 2067 Hz OCAM2K
  limit, and emits a smooth saturation-constrained policy that never slows below
  that empirical model merely to recover per-frame SNR.
- Generalized the HAKA broadband photon-budget helper from fixed Johnson V
  normalization to an explicit Johnson normalization band, retaining V as the
  showcase and eng519 default.
- Fixed temporal integration to average and forward wavelength-resolved photon
  cubes to the detector, preserving configured spectral QE instead of silently
  falling back to scalar QE.
- Fixed even-sized Shack-Hartmann focal-plane registration: zero slope now lies
  at the intersection of the central four detector pixels, using half-integer
  Fourier samples rather than an asymmetric integer-grid crop.
- Added a Keck II HAKA open-loop worked example with a generated 36-segment
  Keck pupil including a live-data-fitted circle-plus-hexagon secondary shadow
  and six 26 mm support arms, exact
  57x57-by-4x4 (228x228) Shack-Hartmann/OCAM2K geometry,
  magnitude-dependent EM gain and frame rate, temporally integrated `pyturb`
  Maunakea OPD, exposure-matched master-dark subtraction, GIF/MP4 output, and a
  reproducibility manifest with per-frame photon/electron/count flux auditing.
  The supplied real eng519 V=10.16 RTC cube constrains the roughly 54-lenslet pupil
  diameter, compact quadcell sampling, and the eight-output 4x2 OCAM geometry,
  outside-pupil dark/bias levels, and relative conversion gains. With no matched
  dark cube, the RTC comparison subtracts a per-output/repeated-4x4 template and
  per-frame output drift inferred outside the pupil, then reports real and
  simulated lenslet signal/morphology without global rescaling. The
  eng519 comparison now simulates V=10.16 at 750 fps, retains every tenth
  generated phase-screen exposure like the telemetry, and writes a side-by-side
  GIF. The magnitude showcase advances frozen flow by a visible minimum cadence
  at every magnitude without changing the physical detector exposure. HAKA NGS
  photon formation now integrates a V-normalized 6600 K spectrum over the full
  400--950 nm band, applies measured Mauna Kea extinction at the observed
  airmass, applies 0.88 reflectivity to the aluminum primary, secondary, and
  tertiary, uses the sampled clear-pupil collecting area, and passes the
  resolved spectral cube through OCAM2K's wavelength-dependent QE. The reference
  renders now use the Keck-characterized approximately 28 output e-/ADU OCAM2K
  conversion. Team-confirmed independent bench measurements now establish the
  downstream HAKA throughput as 28.7%; it is applied as physical radiometry after
  the telescope mirrors rather than inferred or fitted by the RTC comparison.
  The regenerated eng519 comparison has a real/simulation signal ratio of
  1.00029.
- Added a reproducible warm HAKA CPU/GPU benchmark. It times non-periodic Mauna
  Kea atmosphere evolution, the full eight-wavelength 57x57 Shack--Hartmann
  propagation, and noisy OCAM2K exposure while excluding static setup and
  synchronizing CUDA batches. Its local GIF shows CPU/GPU detector streams over
  equal wall-clock playback with measured FPS, real-time factor, frame counter,
  and atmosphere time overlays.
- Removed generated PNG/GIF artifacts from version control and ignore them
  globally; example scripts continue to create them locally on demand.
- Added public end-to-end GPU execution through `numerics.device = "gpu"`.
  CuPy OPD, SH/PWFS optics, wavelength-resolved photon maps, the `getframes`
  detector chain, truth, and ADU remain device-resident. The runtime reports an
  actionable error when the installed `getframes` lacks its GPU camera contract.
- Added direct `pyturb` GPU OPD → Shack–Hartmann → GPU ADU integration coverage,
  updated SH/PWFS CUDA parity tests, synchronized GPU benchmark mode, and measured
  detector-only timing.
- Added a paired CPU/GPU bulk-throughput artifact and rendered comparison for
  representative SH, broadband LGS, and modulated pyramid workflows, with
  README and performance-guide results plus exact reproduction commands.
- Optimized persistent SH/PWFS execution by caching source/range geometry,
  modulation phasors, resampling grids, flux normalization, and monochromatic
  spectral views; using native orthonormal FFT scaling and an intensity-only SH
  transform; removing redundant validations/resampling; and batching GPU
  metadata scalar transfers. On the RTX 5090 reference matrix this improves CPU
  throughput by 1.19x–1.73x and GPU throughput by 1.42x–2.58x over the initial
  end-to-end implementation while retaining the physics/parity gates.

- Expanded optical verification with a direct-DFT pyramid reference,
  multi-amplitude HCIPy SH response curves, HCIPy low-order pyramid response
  maps, supplementary local OOPAO comparisons, and quantitative SH/pyramid
  metrics in the deterministic validation report.
- Fixed the pyramid propagation grid to honor `numerics.fft_oversampling`, so
  the diffraction halo no longer wraps onto the pupil rims; cropped flux is
  reported as captured rate and independent HCIPy parity improved.
- Reconfigured the shipped example TOMLs to be representative demonstrations:
  pyramid pupils are now separated (`pupil_separation_pixels` larger than
  `pixels_across_pupil`) and source photon rates correspond to a bright guide
  star so detector frames show spots above read noise.
- Added deterministic broadband/finite-source quadrature, measured SED and
  transmission curves, physical SH sampling, field stops, optical blur,
  detector margins, and sodium-range SH elongation examples.
- Added strict configuration-reference documentation for every v1 table and
  key, plus validation and benchmark smoke reports in CI.
- Added headless worked-example CI smoke tests, deterministic plotting backend
  selection, and a 90% enforced branch-coverage gate.
- Added configuration-relative three-column angular source kernels for measured
  or resolved guide-star morphologies, with normalized state provenance.
- Added rotated analytic segment-gap pupils and a cached physical-coordinate
  lenslet-grid rotation/offset path with aligned-grid parity tests.
- Added an optional HCIPy ideal-pyramid cross-check and a dedicated validation
  CI job; HCIPy remains outside runtime dependencies.
- Added configuration-relative measured SH optical blur kernels with unit-sum
  validation, cached convolution, and provenance hashes.
- Added a public API/configuration stability audit and same-run benchmark
  regression envelopes for representative CPU kernels.
- Added versioned benchmark snapshots and isolated non-editable-wheel
  interoperability verification for `pyturb` 1.0 and `getframes`.
- Added a versioned labelled SVG capability gallery with units, color bars,
  seeds, configuration digests, and modeling notes.
- Added wavelength-resolved detector QE through the public `getframes` spectral
  cube contract, with truth preservation and a shipped comparison example.
- Formalized the private optical `ArrayBackend` boundary and added static
  leakage/parity checks so a future device backend does not require sensor
  mathematics to be rewritten.
- Added the original private CUDA 12 CuPy optical path with SH/pyramid parity
  tests; it is retained as a compatibility hook underneath the public
  configuration-driven GPU path.
- Added the monochromatic CPU four-face pyramid engine, modulation support, a
  complete pyramid example configuration, and symmetry/flux/detector tests.
- Added the implementation roadmap and agent guide.
- Added the initial configuration and numerical implementation foundation.
