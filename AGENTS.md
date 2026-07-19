# AGENTS.md

This file is the operating guide for AI agents working in `makewfs`. It applies
to the entire repository.

## Start here

Before changing anything:

1. Read `README.md` and the complete `ROADMAP.md`.
2. Read `pyproject.toml` when present, the relevant source/tests/docs, and any
   linked ADR once those files exist.
3. Run `git status --short --branch`. Preserve all user changes and unrelated
   work; never reset or overwrite them.
4. Identify the smallest unchecked roadmap item that contains the requested work
   and state its acceptance criteria.
5. Inspect the public sibling API before proposing cross-repository work:
   `/home/donkeykong/pyturb` for atmosphere and
   `/home/donkeykong/getframes` for detector/radiometry.

The repository is currently in the CPU Shack-Hartmann and fixed-mask four-face
pyramid stage, with deterministic source spectral/angular quadrature, measured
source curves, physical SH sampling controls, and a documented SH sodium-range
geometry model. Do not present range-resolved turbulent LGS OPD,
wavelength-resolved detector QE, independent HCIPy parity, or GPU behavior as
implemented until their gates pass.

## Product boundary

`makewfs` owns wavefront-sensor image formation:

```text
pupil OPD/phase + static config
              |
              v
       makewfs optics
              |
              v
 photon rate [photons/s/native detector pixel]
              |
              v
     getframes.Camera.expose
              |
              v
        Frame data [ADU]
```

- Atmosphere, frozen flow, Cn2 profiles, off-axis footprints, and LGS cone-effect
  phase belong to `pyturb`.
- QE, shot/read/dark noise, gain, detector artifacts, digitization, calibration,
  and detector presets belong to `getframes`.
- Reconstruction, centroid/slopes, DMs, and controllers belong to downstream AO
  software.
- `makewfs` may model finite guide-source and sodium-layer image morphology
  because it is part of WFS image formation, but it never predicts LGS return
  flux or evolves the sodium layer.

Do not copy sibling physics for convenience. If their public API is insufficient,
write a failing integration test/design note, use the conditional gates in
`ROADMAP.md`, and make the smallest change in the owning repository.

## Stable contracts to protect

- The per-frame runtime inputs are the wavefront and an optional noise seed. All
  instrument/source/detector choices live in versioned config.
- OPD metres are the canonical internal wavefront quantity. Phase-radian input
  must declare its reference wavelength; units are never inferred.
- Arrays use `(y, x)` order and documented centered pixel coordinates. Never fix
  a sign or transpose mismatch by visual trial and error—add an analytic ramp test.
- Ideal output is a non-negative photon-rate map in photons/s/native detector
  pixel. Only `getframes` turns it into electrons or ADU.
- Intensities, not fields, are summed over incoherent wavelengths, modulation
  points, finite-source samples, and sodium slices.
- Cropping reports lost flux; it does not renormalize it away.
- The intended top-level API is `load_config`, `WavefrontSensor`, and `simulate`.
  Keep other implementation objects out of `makewfs.__init__` unless an API review
  explicitly accepts them.
- The optical core must not import `pyturb`. Only the detector adapter imports
  `getframes.Camera`; radiometry may import documented `getframes` radiometry APIs.

## Numerical and physics standards

- Start from a derivation, primary paper, or maintained independent reference.
  Cite it in the module and user guide.
- Every physics feature needs a quantitative assertion against theory or an
  independent calculation. “The plot looks right” is not validation.
- Required invariants include piston invariance, non-negative intensity, explicit
  flux accounting, photon-rate linearity, stable axis/sign conventions, and
  convergence with numerical sampling.
- Use explicit centered FFT helpers and normalization. Do not scatter `fftshift`
  conventions through sensor implementations.
- Use flux-conserving pixel-area integration/rebinning; interpolation is not a
  substitute for integrating detector pixels.
- Preserve `float32/complex64` and `float64/complex128` pairs. Test both; do not
  allow silent promotion in a hot path.
- Randomness uses passed `numpy.random.Generator` instances or reproducibly
  derived named seeds. Never use global `np.random` state.
- Static grids, masks, ramps, normalization constants, and detector construction
  are cached on the persistent sensor. Benchmark construction separately from
  warm per-frame operation.
- Write array operations behind the small backend boundary. Avoid unconditional
  NumPy conversion, host scalar reads, and NumPy-only allocations in optical
  kernels so a later CuPy backend remains possible.
- CPU correctness comes first. Never expose `device="gpu"` until parity tests and
  the documented detector-boundary behavior exist.

## Configuration rules

- TOML is canonical. All physical keys include units in their names.
- Config models are immutable and contain only serializable intent, never runtime
  arrays, FFT plans, RNG state, or camera state.
- Reject unknown keys, incompatible alternatives, non-finite values, invalid
  ranges, mismatched detector geometry, and unsupported schema versions with
  path-specific actionable messages.
- Resolve file references relative to the config file. Hash referenced masks,
  curves, and static OPD for provenance.
- A new user-facing field requires validation tests, serialization/digest tests,
  config-reference documentation, and at least one example where appropriate.
- Backward-incompatible schema changes require a schema-version change and a
  migration/stability note.

## Code organization

Follow the target layout in `ROADMAP.md`:

- `config.py` parses and validates; it does not propagate optics.
- `wavefront.py`, `pupil.py`, and `sampling.py` hold shared numerical rules.
- `sensors/` contains deterministic ideal optical engines and no camera noise.
- `radiometry.py` produces source photon budgets using public `getframes` tools.
- `detector.py` is a narrow adapter to `getframes.Camera.expose`.
- `api.py` owns the user facade and caching lifecycle.
- `validation/` produces theory/reference evidence; `benchmarks/` measures speed;
  neither is imported by the runtime package.

Keep functions small enough that their units and normalization can be tested in
isolation. Prefer immutable dataclasses and pure numerical kernels. Public APIs
have type hints and NumPy-style docstrings. Internal names start with `_` unless
another module has a deliberate need for them.

## Tests and quality gate

Once Phase 0 creates the tooling, the normal pre-handoff gate is:

```bash
ruff check .
ruff format --check .
mypy
pytest --cov=makewfs --cov-branch --cov-report=term-missing
mkdocs build --strict
python -m build
```

Also run the narrowest relevant tests while iterating. Mark slow statistical,
validation, GPU, and example tests explicitly; ordinary tests must stay quick.
When sibling packages are installed, run `python -m pytest -m interop` as a
separate compatibility check.

Test public behavior, units, signs, shapes, failure modes, precision, and seeded
reproducibility. Statistical tests use ensemble uncertainty and non-flaky
tolerances. Independent reference packages such as HCIPy are optional validation
dependencies, never core dependencies.

For performance changes, run the affected warm and cold benchmarks and report the
hardware/dependency context. Do not claim a speedup from one timing sample or
weaken physics accuracy to win a benchmark without an explicit documented mode.

## Documentation and examples

- Every public feature lands with its API docstring and the relevant user guide.
- Every configuration field appears in the configuration reference with units,
  default, allowed range, interactions, and an example.
- Examples are scripts plus TOML, deterministic by default, headless, and able to
  save plots. Give them a reduced CI mode; never rely on an interactive notebook
  as the only executable form.
- Plot labels include units and state whether an image is ideal photon rate,
  expected electrons, or noisy ADU.
- Be explicit about approximations, especially partial sampling, chromatic
  pyramid behavior, LGS mean-altitude OPD, and detector cropping.
- Update `CHANGELOG.md` under Unreleased for user-visible changes.

## Roadmap and handoff discipline

- Work in dependency order. Shared contracts/ADRs precede parallel sensor work.
- Take bounded vertical slices; avoid sweeping “implement a whole phase” changes.
- Check a roadmap box only when implementation, tests, docs, and required
  validation/benchmark are all complete and passing.
- Do not mark conditional sibling work as required until its gate is demonstrated.
- At handoff, summarize files changed, assumptions, physics/reference basis,
  commands run and results, benchmark impact, and remaining roadmap items.
- Maintain this file: update it when the repository layout, quality-gate commands,
  stable contracts, or current implementation stage changes.
- If blocked, document the exact failing contract and evidence. Do not work around
  it by absorbing atmosphere or detector behavior into this repository.
