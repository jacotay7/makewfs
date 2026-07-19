# ADR 0003: minimal public API

Status: accepted

The stable entry points are `load_config`, `WavefrontSensor`, and `simulate`.
Users construct one configured sensor and call `photon_rate`, `expose`,
`expose_many`, or `expose_integrated` with phase/OPD arrays. Configuration and
wavefront are the only physical inputs to an exposure; a seed is an optional
detector reproducibility control. Optical engines and detector adapters remain
internal so new sensor types do not expand the top-level API.
