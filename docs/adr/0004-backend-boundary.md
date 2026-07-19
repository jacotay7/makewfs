# ADR 0004: backend boundary

Status: accepted

CPU NumPy/SciPy is the reference implementation. `ArrayBackend` owns array
creation, elementwise operations, reductions, centered FFTs, interpolation,
and optical blur hooks. Sensor engines receive a backend instance and do not
call NumPy allocation/FFT/reduction functions directly; an AST guard and CPU
injection-parity tests protect that rule.

File readers, configuration parsing, source quadrature, metadata, and the
`getframes` adapter are explicit host-side boundaries. `ArrayBackend.scalar`
is used only where a scalar must cross into geometry/metadata, and
`ArrayBackend.to_host` is the named device-to-host escape hatch. The CPU
backend remains the only supported runtime backend. A future CuPy backend must
implement the same contract and reproduce the CPU validation suite before it
is exposed.
