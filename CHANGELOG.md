# Changelog

All notable changes to `makewfs` are documented here.

## [Unreleased]

- **Compatible GPU Shack--Hartmann optics now use a first-use-JIT compiled
  executor.** CuPy specializes one CUDA kernel for the fixed lenslet, temporal,
  spectral, precision, field-stop, and detector-sampling geometry, then reuses
  its process and disk caches. It fuses sampled-DFT propagation through photon
  mosaics and composes detector-owned focal charge diffusion with native-pixel
  integration once at startup. The array implementation remains the exact
  reference/fallback for CPU, FFT, continuous/native optical blur, and oversized
  CUDA geometries. A cold isolated-cache, alternating 20-frame physical HAKA
  benchmark on a Quadro P620 measured 24.85 ms versus 569.67 ms p50 (22.93x),
  after a 3.176 s first-use compile, with photon/spectral/captured-rate relative
  disagreements below 1e-7.
- `WavefrontSensor.expose()` and `expose_integrated()` now accept an optional
  caller-owned detector `out` array. The detector adapter pairs it with
  `getframes.DetectorWorkspace`, including wavelength-resolved exposures, so a
  high-rate owner can keep one stable contiguous ADU destination. Ordinary calls
  retain independent frame lifetimes; only explicit `out` calls alias the
  caller's storage.
- **Persistent Shack--Hartmann sensors now cache propagation geometry.** Half-sample
  FFT ramps, sampled-DFT kernels, field-stop masks, and backend blur kernels are
  built once per compatible source geometry instead of once per frame. The ordinary
  large FFT path is intentionally unchanged; matched local timings improved the
  geometry-heavy field-stop/DFT paths by roughly 6--20% with optical parity tests.
- **A temporally integrated exposure now renders its samples in one pass.**
  `ShackHartmannEngine.render_integrated` builds the fields for every
  (temporal sample, source state) pair as a single batch, and
  `expose_integrated` uses it when the engine provides it. The motivation is
  not transform size: on a HAKA-scale configuration the transforms are about
  four percent of a render, and the cost is dominated by the fixed dispatch
  overhead of the many small elementwise operations around them, which is paid
  per call however much data the call carries. Presenting the whole exposure at
  once amortises that: the reference HAKA exposure drops from 13.7 ms to
  11.2 ms.

  Averaging the spot intensities before the mosaic is legitimate because
  everything downstream of them -- mosaic assembly, flux scaling, and the
  captured-rate accounting -- is linear in the spots, and there is a test
  asserting the batched and sequential paths agree rather than leaving that as
  an argument. Agreement is to float round-off from the changed summation
  order, about 1e-7 relative in single precision, not bit-for-bit.

- **Detector charge diffusion now reaches the Shack--Hartmann spots.** The
  measured OCAM2K value was previously carried as
  `shack_hartmann.optical_blur_fwhm_pixels` and applied *after* pixel
  integration, where a 0.37-pixel FWHM Gaussian is a numerical no-op, so a
  measured detector property changed nothing. Charge diffusion is detector
  physics, so `getframes` now owns both the value
  (`CameraConfig.charge_diffusion_fwhm_px`, declared by the `andor_ocam2k`
  preset) and the kernel model; the Shack--Hartmann engine asks `getframes` for
  the operator at its own focal-plane oversampling and applies it to the
  oversampled irradiance ahead of the pixel-area integration that collects the
  diffused charge. Configuration that cannot represent the width now fails with
  the required `fft_oversampling` instead of applying nothing.
  **This changes delivered spot profiles and slope gains** for any detector
  declaring a nonzero width, so recorded HAKA evidence must be regenerated.
  `makewfs.charge_diffusion_fwhm_px` and `makewfs.resolve_camera_config` are new
  public helpers for consumers needing a sensor property before a frame exists.
- Added `WavefrontSensor.subaperture_plate_scale_arcsec()` and
  `subaperture_field_of_view_arcsec()`, which report what one detector pixel and
  one subaperture window subtend on sky. A Shack--Hartmann's pixel block is
  already a hard square field stop: each spot is formed and integrated only over
  its own block and the blocks tile without overlap, so light beyond the pixel
  field neither reaches the detector nor contaminates a neighbour. That was
  implicit in the pixel count, where a change to `pixels_per_subaperture`, the
  relay magnification, or the detector margin would move the implied stop
  silently. Both are derived from the same spot-sampling geometry that forms the
  spots rather than restating it. Light spilling between subapertures from a
  physical stop larger than the pixel field remains unmodelled.
- Added `WavefrontSensor.pupil_illumination(shape=None)`, which evaluates the
  configured telescope pupil on a requested grid (default the OPD input grid).
  Consumers that own actuator or wavefront models need the illumination on their
  own grid, and pupil formation belongs here. A configured `custom_mask_path` is
  never resampled: it must already match the requested shape.
- `shack_hartmann.optical_blur_fwhm_pixels` now means genuine focal-plane optical
  blur only, and is applied on the oversampled grid so sub-pixel widths stay
  physical. A measured `optical_blur_kernel_path` is supplied on the native pixel
  pitch and still applies after pixel integration.
- The HAKA example raises `fft_oversampling` from 2 to 4, the minimum that
  represents the OCAM2K's measured 0.37-pixel charge diffusion.
- Fixed arbitrary Shack--Hartmann spot sampling so normalized plate scales no
  longer snap to a nearby integer FFT grid. Integer-compatible geometries retain
  the FFT path; arbitrary and undersampled quadcell modes use a sampled DFT at
  detector-cell quadrature points, with CPU/GPU-compatible batching. Physical
  lenslet models can now provide an explicit `lenslet_pitch_m`, separating
  hardware focal-plane sampling from the telescope-pupil coordinate scale.
- Added detector-owned full-sensor ROI configuration through
  `[detector.roi]`. makewfs now passes ROI origin and shape to getframes instead
  of replacing a preset's native detector resolution. The HAKA OCAM2K simulation
  uses its measured `left_px=4`, `top_px=4`, 228x228 ROI, placing amplifier
  boundaries at y=(56, 116, 176) and x=(116) in RTC image coordinates.
  Stability note: `detector` now serializes a `roi` key, so every configuration
  digest changes even when no ROI is configured. `schema_version` stays 1 and
  existing TOML files load unchanged; only recorded provenance digests must be
  regenerated.
- Accelerated the common two-times Shack--Hartmann detector integration with
  direct flux-preserving strided sums, and made temporal exposure integration
  accumulate rates and OPD incrementally instead of stacking full frame cubes.
- Batch GPU Shack--Hartmann source states only when they share the same FFT
  geometry. CPU and wavelength-dependent field-stop execution remain on the
  sequential reference path. A matched 64-frame Quadro P620 HAKA-class
  benchmark reduces median optics time by 2.16% with a maximum relative
  photon-rate difference of 5.1e-8.

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
  1.00489 after correcting the RTC ROI origin and refitting the amplifier
  responses and secondary shadow.
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
