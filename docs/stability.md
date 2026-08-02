# API and configuration stability

The stable top-level names are:

| Name | Contract |
| --- | --- |
| `load_config(path)` | Load and validate schema-version-1 TOML, resolving referenced files relative to the TOML. |
| `WavefrontSensor(config)` | Construct one cached optical/detector pipeline; call `photon_rate`, `expose`, `expose_many`, or `expose_integrated` with phase/OPD arrays. `expose` and `expose_integrated` optionally accept caller-owned `out` storage with an explicit alias lifetime. |
| `simulate(config, wavefront, ...)` | One-shot convenience wrapper around the same pipeline. |
| `WFSConfig` / `Config` | Frozen normalized configuration value; `Config` is a compatibility alias of `WFSConfig`. |
| `ConfigError` | Validation exception for actionable configuration failures. |
| `__version__` | Package version metadata. |

The three workflow names are the recommended minimal API. Configuration types
and `ConfigError` are public to support typed callers and validation UIs, but
runtime engines, detector adapters, source-state classes, and numerical helper
functions are intentionally not re-exported from `makewfs`.

The complete schema key inventory is maintained in
[`configuration.md`](configuration.md); the strict loader rejects unknown keys.
Every new physical field must be documented there, validated, covered by a
regression test, and recorded in the changelog before it is considered stable.

The physical per-frame contract is a wavefront array plus configuration (and an
optional detector seed). `out` controls storage lifetime only and cannot change
the simulated result. Atmosphere, reconstruction, controllers, and detector
physics remain outside the package boundary.
