# ADR 0003: minimal public API

Status: accepted

The stable entry points are `load_config`, `WavefrontSensor`, and `simulate`.
Users construct one configured sensor and call `photon_rate`, `expose`,
`expose_many`, or `expose_integrated` with phase/OPD arrays. Configuration and
wavefront are the only physical inputs to an exposure; a seed is an optional
detector reproducibility control. Optical engines and detector adapters remain
internal so new sensor types do not expand the top-level API.

[ADR 0005](0005-lenslet-offset-states.md) adds an optional `offset_state` index
that selects one of the Shack-Hartmann lenslet-grid positions declared in the
configuration; it adds no instrument parameter outside the configuration.
