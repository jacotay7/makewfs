# ADR 0004: backend boundary

Status: accepted

CPU NumPy/SciPy is the reference implementation. `ArrayBackend` owns array
creation, elementwise operations, reductions, centered FFTs, interpolation,
and optical blur hooks. Sensor engines receive a backend instance and do not
call NumPy allocation/FFT/reduction functions directly; an AST guard and CPU
injection-parity tests protect that rule.

File readers, configuration parsing, source quadrature, and metadata are explicit
host-side boundaries. `ArrayBackend.scalar` is used only where a scalar must
cross into geometry/metadata, and `ArrayBackend.to_host` remains the named escape
hatch for file-facing workflows.

Amendment (post device-resident `getframes` gate): schema v1 now accepts
`numerics.device = "gpu"`. The detector adapter passes CuPy photon-rate arrays
directly to `getframes.Camera(..., device="gpu")`; `Frame.data` and truth remain
on device. The old private backend-injection hook remains for compatibility, but
the configuration field is the supported user contract. CPU NumPy/SciPy remains
the numerical reference.
