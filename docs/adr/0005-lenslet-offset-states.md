# ADR 0005: pre-declared Shack-Hartmann lenslet-grid offset states

Status: accepted (2.2.0)

## Context

Dithered, modulated, and time-varying Shack-Hartmann schemes move the lenslet
grid relative to the pupil from frame to frame
([#19](https://github.com/jacotay7/makewfs/issues/19)). Before 2.2 the only
control was the static `shack_hartmann.lenslet_grid_offset_fraction`, so each
position needed its own `WavefrontSensor`, with duplicated pupil, propagation,
and detector state.

ADR 0003 makes the wavefront the only physical per-frame input, with an optional
detector seed; every instrument choice lives in versioned configuration. An
arbitrary per-call offset, such as `expose(opd, lenslet_offset=(fx, fy))`,
would move instrument geometry out of the configuration: frames could no longer
be reproduced from the configuration and its digest, the valid-subaperture map
would no longer be a static calibration artifact, and each new offset would
rebuild geometry inside the frame loop.

## Decision

- `shack_hartmann.lenslet_grid_offset_states_fraction` declares the offset
  states as a list of `[x, y]` offsets in lenslet pitches. Each is added to the
  static `lenslet_grid_offset_fraction`. Without the key there is one state,
  the static offset.
- The sensor builds and caches every state's geometry at construction: the
  translated pupil illumination, lenslet mask, valid-subaperture map, source and
  sodium-range tilts, flux normalization, and (for `lgs_model = "kernel"`) the
  LGS kernels. The compiled CUDA executor is cached per state.
- `photon_rate`, `reference`, `expose`, `expose_integrated`, and
  `valid_subapertures` accept a keyword `offset_state: int = 0` that selects a
  declared state; `expose_many` accepts an iterable `offset_states`.
  `lenslet_grid_offsets_fraction()` lists the effective offsets in index order.
  An index outside the declared states is an error; a pyramid accepts only `0`.
- Shack-Hartmann frames record `wfs_lenslet_offset_state`,
  `wfs_lenslet_grid_offset_fraction`, and `wfs_lenslet_offset_state_count`.

The per-frame inputs are therefore the wavefront, the optional seed, and an
index into configured geometry. The index selects intent that is already in the
configuration; it does not introduce any instrument parameter that the
configuration and digest do not record.

## Consequences

- A dithered loop is one sensor and one detector; switching state is a cached
  lookup.
- Memory grows with the number of states times the number of propagated source
  states, because each state keeps its own source-tilt arrays.
- Existing configurations are unchanged: the new key is omitted from the
  normalized configuration at its default, so their digests do not move, and
  `offset_state` defaults to `0`.
- A continuously varying offset is not supported; declare the positions a
  scheme visits. A future ADR may revisit this if a use case needs it.
