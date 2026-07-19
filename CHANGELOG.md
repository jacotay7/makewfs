# Changelog

All notable changes to `makewfs` are documented here.

## [Unreleased]

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
- Added the monochromatic CPU four-face pyramid engine, modulation support, a
  complete pyramid example configuration, and symmetry/flux/detector tests.
- Added the implementation roadmap and agent guide.
- Added the initial configuration and numerical implementation foundation.
