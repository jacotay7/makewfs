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
- Added the monochromatic CPU four-face pyramid engine, modulation support, a
  complete pyramid example configuration, and symmetry/flux/detector tests.
- Added the implementation roadmap and agent guide.
- Added the initial configuration and numerical implementation foundation.
