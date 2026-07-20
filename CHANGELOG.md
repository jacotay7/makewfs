# Changelog

All notable changes to `makewfs` are documented here.

## [Unreleased]

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
  interoperability verification for `pyturb` 1.0 and `getframes` 2.0.
- Added a versioned labelled SVG capability gallery with units, color bars,
  seeds, configuration digests, and modeling notes.
- Added wavelength-resolved detector QE through the public `getframes` spectral
  cube contract, with truth preservation and a shipped comparison example.
- The spectral-QE path uses the upcoming `getframes>=2.1.0` capability when
  available and retains a documented integrated-signal fallback for 2.0.
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
