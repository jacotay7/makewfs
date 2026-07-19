# API and configuration stability

The stable top-level names are `load_config`, `WavefrontSensor`, and
`simulate`. Configuration schema version 1 is strict: unknown keys and invalid
cross-field combinations fail rather than being guessed. New physical fields
must be documented and validated before they are considered stable.

The per-frame contract is a wavefront array plus configuration (and an optional
detector seed). Atmosphere, reconstruction, controllers, and detector physics
remain outside the package boundary.
