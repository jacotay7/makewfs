# makewfs roadmap

This is the implementation plan and checklist for `makewfs`. It is written so
that an AI agent can take one bounded checklist item, implement it, validate it,
document it, and leave the repository in a healthy state.

The repository now contains the initial CPU Shack–Hartmann implementation as
well as this plan. Unchecked boxes are remaining work. A checked box must mean
that the code, tests, documentation, and relevant validation or benchmark all
exist and pass. Do not check work off for a stub or an unverified visual.

## 1. Mission and definition of success

`makewfs` converts one pupil-plane wavefront plus an adaptive-optics designer's
configuration into a realistic wavefront-sensor detector image.

The 1.0 release is complete when all of the following are true:

- A user can configure and run a Shack–Hartmann or four-face pyramid WFS without
  writing optical propagation or detector code.
- The stable per-frame contract is only `wavefront + config`; it works equally
  for an open-loop phase screen or a closed-loop residual phase.
- The ideal WFS optics produce a photon-rate map in photons/s/native detector
  pixel, and `getframes` converts that map into the detector's ADU frame.
- Natural guide stars, finite source size, guide-star magnitude/flux, broadband
  sensing, realistic telescope pupils, and useful sodium-LGS morphology are
  represented with documented assumptions.
- The moving-atmosphere examples use `pyturb`, and all camera noise and detector
  effects use `getframes`.
- Numerical conventions, flux conservation, sensor response, and stochastic
  behavior have quantitative tests against analytic results or an independent
  reference.
- The common CPU path is vectorized, benchmarked, and free of repeated setup in
  the per-frame loop. Its `ArrayBackend` boundary also supports the private
  parity-tested CuPy optical path; detector execution remains host-side.
- Every public object and every configuration field is documented. The
  quickstart, configuration reference, physics guides, examples, validation
  gallery, performance notes, and API reference build in strict mode.
- Ruff, formatting, strict typing, tests, coverage, docs, package builds, and
  supported-Python smoke tests run in CI.

## 2. Firm scope boundaries

### makewfs owns

- Strict, versioned configuration in vocabulary useful to AO instrument
  designers.
- Input wavefront validation, units, coordinates, pupil definition, static WFS
  path aberrations, and wavelength conversion.
- Diffraction propagation through Shack–Hartmann and pyramid WFS optics.
- Guide-source image morphology needed by the WFS: angular extent, spectral
  weighting, and geometric sodium-layer spot elongation.
- Flux-conserving sampling onto native detector pixels.
- Source normalization from direct photon rate or magnitude by reusing
  `getframes` radiometry.
- Handoff of a photon-rate map to a configured `getframes.Camera`.
- Reference images, truth/provenance metadata, configuration CLI, examples,
  validation, and performance benchmarks.
- A small sensor protocol that allows more WFS types to be added later.

### makewfs does not own

- Atmospheric turbulence, frozen flow, Cn2 profiles, cone-effect phase, or
  scintillation. Use `pyturb`; amplitude fluctuations remain out of scope while
  the input contract is phase/OPD only.
- Shot noise, QE, PRNU, read noise, gain, dark current, detector artifacts,
  digitization, persistence, or calibration. Use `getframes`.
- Wavefront reconstruction, centroiding algorithms, slope extraction,
  interaction-matrix inversion, deformable mirrors, controllers, or complete AO
  loop orchestration. Downstream AO code consumes the images.
- Laser launch propagation, sodium atomic physics, sodium-profile evolution, or
  LGS return-flux prediction. A user supplies the LGS photon rate and a simple
  sodium density profile; `makewfs` only forms the WFS image implied by them.
- General ray tracing, Fresnel propagation through an arbitrary optical train,
  polarization, or opto-mechanical tolerancing.

These boundaries are architectural rules. If work becomes difficult at an
external boundary, first write an integration test that demonstrates the missing
capability, then add it to the repository that owns that physics.

## 3. What is already available locally

The roadmap is based on an audit of the sibling repositories on 2026-07-19.

### pyturb 1.0.0 (`/home/donkeykong/pyturb`)

- `Atmosphere.frames(dt=..., steps=...)` yields time and `(n, n)` OPD arrays in
  metres. `Atmosphere.opd(wavelength=...)` can instead return phase in radians.
- Layered frozen flow, off-axis directions, boiling, named site profiles, and
  the LGS cone effect already exist on CPU and GPU.
- `pyturb.to_numpy` provides an explicit device-to-host boundary.
- Its explicit non-goals include WFS, detector, DM, and controller simulation.

The primary v1 integration is therefore direct and needs no wrapper in the core
package:

```python
for time_s, opd_m in atmosphere.frames(dt=config.exposure_s, steps=1000):
    frame = wfs.expose(opd_m)
```

### getframes 2.1.0 (`/home/donkeykong/getframes`)

The installed/released sibling is now `getframes 2.1.0` (the audit baseline was
2.0.0). The scalar dependency constraint remains `getframes>=2.0`.

- `Camera.expose(photon_rate, exposure, ...)` accepts a scalar or native-pixel
  photon-rate map and returns an array-like `Frame` in ADU with noise-free truth.
  Its 2.1.0 signature adds `binning`/`binning_mode`, which makewfs uses.
- CCD, CMOS, sCMOS, EMCCD, and eAPD noise paths, fixed detector structure,
  binning, time-series state, calibration, FITS output, and seeded
  reproducibility already exist.
- AO-relevant presets already include OCAM2K/CCD220, SAPHIRA/eAPD, NUVU cameras,
  and the Keck Little Joe CCD39. Note that the SAPHIRA/eAPD presets are near-IR
  (their QE curve is zero at visible sensing wavelengths), so visible-band WFS
  examples use CCD/EMCCD/sCMOS/CMOS presets instead.
- Broadband photometry, QE curves, SEDs, transmission products, sky, and thermal
  radiometry already exist (`getframes.QE`, `SED`, `Bandpass`, `Spectrum`).
- **Wavelength-resolved cube exposure** (`Camera.expose_spectral` + spectral
  `FrameTruth`) is implemented on getframes `main` and slated for the **2.1.1**
  release. It is present in this workspace's editable getframes checkout (so the
  makewfs spectral test passes locally), but released 2.1.0 does not include it.
  The section 13 gate stays open until 2.1.1 is released and pinned; against a
  getframes without it, makewfs uses the documented QE-weighted integrated
  fallback.

The detector handoff must call this public API. No detector-noise functions are
to be copied into `makewfs`.

### Upstream status

No sibling change is required for the first monochromatic Shack–Hartmann or
pyramid vertical slice. Conditional upstream work is listed in section 13; it
must not be implemented until its acceptance test demonstrates the need.

### Current implementation checkpoint (2026-07-19)

- [x] Repository foundation, strict TOML configuration, CPU backend boundary,
  ADRs, CI/docs scaffolding, and a validated minimal API.
- [x] Batched CPU Shack–Hartmann propagation with OPD/phase input conversion,
  analytic pupils, direct/magnitude source normalization, and getframes output.
- [x] Monochromatic four-face pyramid propagation with deterministic circular
  modulation, fixed face-order provenance, and a complete detector example.
- [x] CLI, closed-loop sequence/integration helpers, worked example scripts,
  and regression tests for seeded detector behavior.
- [x] Deterministic wavelength/SED/transmission quadrature, finite NGS angular
  extent, and a documented Shack–Hartmann sodium-range elongation model.
- [x] Executable validation metrics, benchmark runners, LGS elongation and
  detector-choice examples, and closed-loop injection example.
- [x] Pyramid propagation now honors `numerics.fft_oversampling` for its FFT
  grid, so the diffraction halo no longer wraps onto the pupil rims; cropped
  flux is reported as captured rate and HCIPy parity improved (~0.55 -> >0.90).
- [x] The shipped example TOMLs are representative (separated pyramid pupils,
  bright-guide-star photon rates), and all worked examples were reworked for
  legibility: colorbars with units, real named getframes presets, a GIF for the
  moving-atmosphere example, and a verified LGS-elongation figure.
- [ ] Broader independent validation, public GPU support, the wavelength-resolved
  detector-QE gate (implemented on getframes `main`, awaiting the getframes 2.1.1
  release before it can be pinned), and release completion remain staged. The
  private CuPy optical path now has CPU parity evidence, and the versioned
  documentation gallery and relative benchmark regression envelopes are in place.

## 4. End-state user experience

### Configuration first

TOML is the canonical human-readable format. It matches `getframes` presets,
ships in Python 3.11's standard library, supports comments, and has unambiguous
types. Python 3.10 uses the `tomli` compatibility dependency.

Every physical field carries a unit suffix. The loader rejects unknown keys,
conflicting ways to specify the same quantity, invalid cross-table combinations,
and unsupported schema versions. It never guesses units.

A minimal Shack–Hartmann configuration should look like this:

```toml
schema_version = 1

[input]
quantity = "opd"
unit = "m"
shape = [512, 512]

[telescope]
pupil_diameter_m = 8.0
central_obscuration_ratio = 0.14

[source]
kind = "ngs"
normalization = "magnitude"
magnitude = 12.0
magnitude_system = "vega"
band = "R"
throughput = 0.35

[sensor]
kind = "shack_hartmann"
wavelength_m = 7.0e-7

[shack_hartmann]
lenslets_across_pupil = 20
pixels_per_subaperture = 12
spot_sampling_pixels_per_lambda_over_d = 2.0
minimum_illuminated_fraction = 0.25

[detector]
preset = "andor_ocam2k"
exposure_s = 0.001
temperature_c = -45.0
binning = 1
binning_mode = "digital"

[numerics]
dtype = "float32"
fft_oversampling = 2
fft_workers = 1
```

For a laboratory source or when the photon budget is already known, the
`[source]` table instead uses:

```toml
kind = "ngs"
normalization = "detector_photon_rate"
detector_photon_rate_per_s = 2.0e6
```

That rate is defined at the detector surface before detector QE. Exactly one
normalization path is permitted.

### Minimal Python API

The stable public surface should stay small:

```python
import makewfs

config = makewfs.load_config("wfs.toml")
wfs = makewfs.WavefrontSensor(config)       # build and cache static optics once
# Equivalently: makewfs.WavefrontSensor.from_toml("wfs.toml")

ideal_rate = wfs.photon_rate(opd_m)         # ndarray, photons/s/native pixel
reference = wfs.reference()                 # ideal zero-OPD photon-rate image
frame = wfs.expose(opd_m, seed=0)           # getframes.Frame, ADU

# One-off convenience; not recommended inside a high-rate loop.
frame = makewfs.simulate(opd_m, "wfs.toml", seed=0)
```

`WavefrontSensor` is a facade that dispatches to the configured sensor engine;
users do not need to construct a concrete optics class. Concrete sensor engines
remain importable from `makewfs.sensors` for expert work but are not top-level
API.

For a sequence, `wfs.expose_many(phases, seeds=None)` returns an iterator and
reuses the same cached optics. It must not eagerly stack an unbounded closed-loop
stream.

`wfs.expose_integrated(phase_samples, seed=...)` represents one detector exposure
with uniformly weighted temporal OPD samples. It sums ideal intensities before
calling the detector once. This is deliberately distinct from `expose_many`,
which performs one detector read per phase. A single 2-D snapshot remains the
normal closed-loop path.

### CLI

The CLI is the file-based equivalent of the API:

```bash
makewfs validate-config wfs.toml
makewfs render wfs.toml phase.npy --output frame.fits --seed 0
makewfs ideal wfs.toml phase.fits --output photon-rate.npy
```

Supported phase input is `.npy`, `.npz`, or FITS. Output is FITS, `.npy`, or
`.npz`; formats which can carry it include the full configuration and provenance.
The CLI does not generate an atmosphere.

## 5. Data and physics contracts

### 5.1 Input wavefront

- The basic input is a real 2-D array with axes `(y, x)`, origin at the array
  centre, and `+x` right / `+y` up in the pupil plane.
- `[input].shape = [height, width]` fixes the closed-loop array contract so grids
  and masks can be built once. Every per-frame array must match it. A temporal
  integration is an iterable or `(n_time, height, width)` stack of arrays with
  that same trailing shape.
- `[input].quantity` is `"opd"` or `"phase"`. OPD in metres is canonical because
  it is achromatic and is exactly what `pyturb` emits. Phase requires
  `unit = "rad"` and `reference_wavelength_m`; internally it is converted to OPD
  before polychromatic propagation.
- The array spans `[input].grid_extent_m`, defaulting to the configured telescope
  pupil diameter. Pixel centres, FFT centring, even/odd sizes, and axis signs are
  documented and tested.
- When a sensor needs a different internal pupil sampling (for example a number
  divisible by the SH lenslet count), resample canonical OPD—not wrapped phase—on
  the documented physical coordinate grid. Plane and band-limited-mode tests
  guard interpolation accuracy. The original dynamic input is never silently
  cropped to make a reshape convenient.
- The configured pupil supplies amplitude; no amplitude array is a per-frame
  input. Non-finite values inside illuminated pupil pixels are errors. Values
  outside the pupil are ignored and zeroed.
- Constant piston must leave every ideal intensity image invariant to numerical
  precision. Piston is not silently removed except as an explicitly documented
  numerical stabilization before evaluating `exp(i phase)`.
- Static WFS-path OPD may be a config-referenced `.npy` or FITS map. It is added
  to the dynamic input and recorded in provenance.

### 5.2 Pupil

- Built-ins: circular, annular, central obscuration, configurable spiders, pupil
  rotation, and segment gaps sufficient for representative telescopes.
- A custom amplitude mask may be supplied by path in the config. This remains
  `phase + config` because the static aperture is part of the instrument config.
- Masks are sampled by pixel-area integration or controlled supersampling, not a
  binary pixel-centre shortcut at coarse resolution.
- Flux normalization uses the sampled illuminated area and reports the launched,
  optically transmitted, and detector-captured rates separately.

### 5.3 Wavelength and incoherent sums

- The complex pupil field at wavelength `lambda` is
  `pupil * exp(2j*pi*opd/lambda)`.
- Monochromatic mode uses `[sensor].wavelength_m`.
- Broadband mode uses wavelength nodes and normalized photon weights derived
  from a configured band/SED/transmission product. Intensities, never complex
  fields, are summed across mutually incoherent wavelength, modulation, source,
  and sodium-range samples.
- Quadrature convergence is documented and testable. A broadband result records
  the wavelength nodes, weights, and captured flux.

### 5.4 Flux and detector handoff

- Sensor engines first return a dimensionless, flux-conserving native-pixel
  intensity distribution plus captured-flux diagnostics.
- Radiometry scales it to photons/s/native detector pixel. Cropping or field stops
  are allowed to lose flux, but loss is explicit in metadata and never hidden by
  renormalizing the cropped image.
- `getframes.Camera.expose` is the scalar conversion from photon rate to noisy
  ADU. The optional `Camera.expose_spectral` cube boundary is equivalent for
  wavelength-resolved QE and preserves incident spectral truth. Exposure,
  temperature, binning, precision, preset/overrides, and detector seed come from
  config or an explicit per-call seed.
- Optics are deterministic. All random detector effects remain in `getframes`.
  All optional Monte Carlo source sampling uses an independent, named RNG stream
  and has a deterministic quadrature default.
- The returned `getframes.Frame` is tagged with sensor kind, config digest,
  wavelength summary, launched/captured photon rates, input OPD RMS, dependency
  versions, precision, and seed. The ideal map is already available as
  `frame.truth.photon_rate` when truth is enabled.

### 5.5 Sign and layout conventions

- A positive x phase ramp and its predicted detector motion are fixed by an
  analytic tilt test before either sensor implementation is accepted.
- A Shack–Hartmann mosaic is indexed in the same `(lenslet_y, lenslet_x)` order as
  the pupil; detector subimages are laid out without implicit transposes.
- The four pyramid pupils have a named, documented order tied to mask-face signs.
  The order is present in metadata; code must not depend on an unexplained visual
  quadrant convention.

## 6. Target architecture

```text
src/makewfs/
  __init__.py          # deliberately small stable exports
  __about__.py         # version
  api.py               # WavefrontSensor facade and simulate()
  config.py            # immutable validated config + TOML loading
  backend.py           # CPU boundary plus private experimental CuPy backend
  wavefront.py         # units, coordinates, OPD conversion, validation
  pupil.py             # analytic/custom pupil sampling
  sampling.py          # centred FFTs, flux-conserving integration/rebinning
  radiometry.py        # source normalization using getframes radiometry
  detector.py          # narrow getframes adapter; no detector physics
  provenance.py        # config digest and metadata
  sensors/
    __init__.py
    base.py             # internal sensor-engine protocol
    shack_hartmann.py   # batched subaperture propagation
    pyramid.py          # focal mask and re-imaged pupil propagation
    lgs.py              # sodium-range/source morphology used by WFS optics
  cli.py
  py.typed
tests/
  unit/
  integration/
  validation/
validation/
  run.py               # reproducible gallery + JSON metrics
benchmarks/
  run.py
  artifacts/           # versioned reference results and environment metadata
docs/
examples/
```

The sensor engine protocol accepts validated OPD and static config, returns an
ideal intensity plus diagnostics, and contains no camera calls. The detector
adapter is the only module that imports `getframes.Camera`. `pyturb` is an
examples/interop optional dependency and is never imported by the optical core.

## 7. Configuration model

The schema is composed of frozen dataclasses (or equivalently strict immutable
models) with explicit `from_dict` validation. Runtime FFT plans, cached masks,
camera state, and RNGs do not live in configuration objects.

### Common tables

- `schema_version`: required integer; v1 loader only accepts `1` until a migration
  mechanism exists.
- `[input]`: quantity, unit, shape, optional reference wavelength, grid extent,
  static OPD path.
- `[telescope]`: pupil diameter, aperture kind, obscuration, spiders, rotation,
  segment gaps, or custom mask.
- `[source]`: NGS/LGS, direct detector-surface photon rate or magnitude path,
  magnitude system/band, field angle, angular extent, SED/transmission paths,
  throughput, background, and wavelength quadrature.
- `[sensor]`: kind and reference wavelength.
- `[detector]`: `getframes` preset or complete inline `CameraConfig`, allowed
  overrides, exposure, temperature, binning, binning mode, precision, truth.
- `[numerics]`: float32/float64, optional internal pupil sampling, FFT
  oversampling, FFT workers, quadrature and modulation chunk sizes, pupil
  supersampling, and diagnostic tolerances.
- `[random]`: optional base seed and independently derived stream names. A
  per-call seed overrides only detector/source sampling for that exposure.

Unknown keys are errors in the core schema. Vendor-specific notes belong under an
explicit `[metadata]` table and are preserved without affecting simulation.
NGSs may use direct-rate or magnitude normalization. The v1 sodium-LGS path
requires a configured return photon rate because predicting laser return is an
explicit non-goal.

### Shack–Hartmann table

The common normalized mode requires:

- `lenslets_across_pupil`
- `pixels_per_subaperture`
- `spot_sampling_pixels_per_lambda_over_d`
- `minimum_illuminated_fraction`

It may configure square-lenslet fill factor, lenslet-grid rotation/offset,
detector margins/gaps, field stop, spot blur/source extent, and partially
illuminated-subaperture policy.

An optional physical-instrument mode accepts lenslet pitch, lenslet focal length,
relay pupil diameter, detector pixel pitch (normally from the camera preset), and
relay magnification. The loader derives the normalized sampling and rejects
inconsistent redundant values. Normalized mode is the quickstart because it
doesn't force users to model an irrelevant relay merely to explore WFS behavior.

### Pyramid table

The initial four-face ideal-pyramid mode requires:

- `pixels_across_pupil`
- `pupil_separation_pixels`
- `modulation_radius_lambda_over_d` (`0` is unmodulated)
- `modulation_samples` (`1` is valid only at zero radius)

Optional fields control detector margins, focal-plane oversampling/extent,
modulation path/weights, pupil cropping, physical apex angle, refractive-index
model, and manufacturing defects. The ideal separation parameter is canonical
for early releases. A physical mask model derives its wavelength-dependent pupil
separation from apex angle and refractive index and is added only with broadband
validation.

### LGS source table

LGS morphology requires mean range, sodium range samples/density weights, launch
telescope position in the entrance pupil, and optional intrinsic angular FWHM.
Built-in Gaussian and top-hat sodium profiles are conveniences; an AO designer
may supply a normalized range/density table. Total return photon rate is always
configured, never predicted.

Version 1 uses the provided input OPD as the phase at the configured mean beacon
range while summing geometric spot offsets over the sodium profile. This captures
the appearance and centroid consequences of perspective elongation without
pretending that one phase map contains a range-resolved turbulent volume. The
approximation must be visible in the LGS guide and metadata.

The initial sodium perspective/elongation model is specific to Shack–Hartmann
subapertures. The v1 pyramid sensor may use a configured finite angular source,
but it must not claim a range-resolved sodium-LGS model without a separate design
and validation of the near-field geometry.

## 8. Phased implementation checklist

### Phase 0 — repository foundation and locked contracts (0.0.x)

- [x] Add `pyproject.toml` using Hatchling, a `src/` layout, Python >=3.10,
  typed-package marker, dynamic version, and MIT license matching the sibling
  projects. Test Python 3.10 through the current stable version supported by the
  runtime dependencies rather than baking in an unnecessary upper bound.
- [x] Add runtime dependencies (`numpy`, `scipy`, `getframes>=2.0`) and separated
  `dev`, `docs`, `examples`, `interop`, and optional CUDA 12 `gpu` extras.
  `pyturb` belongs to `interop/examples`, not core.
- [x] Configure Ruff lint and format, strict mypy, pytest strict markers, branch
  coverage, and pre-commit. Use 100-character lines and NumPy-style docstrings to
  match `getframes` unless an ADR records a change.
- [x] Add `LICENSE`, `CHANGELOG.md`, `CONTRIBUTING.md`, `.gitignore`,
  `.pre-commit-config.yaml`, and package/version smoke tests.
- [x] Add CI for lint/format/type-check, strict docs build, Linux/macOS/Windows
  tests on supported Python versions, wheel/sdist build and install, and coverage
  upload. Establish an initial 90% branch-coverage target for `src/makewfs`.
- [x] Implement immutable config models, strict TOML loading, useful path-aware
  validation errors, schema versioning, config round-trip/digest, and a
  `validate-config` CLI.
- [x] Commit example minimal SH and pyramid TOML files and test that every shipped
  config loads.
- [x] Write and accept short ADRs for units/coordinates, flux normalization, FFT
  normalization, public API, and backend boundary. Link each ADR from the docs.
- [x] Implement the CPU backend boundary with centered FFT helpers and explicit
  `float32/complex64` and `float64/complex128` pairs. Do not expose a fake GPU
  option; unsupported devices fail clearly.
- [x] Create the MkDocs structure and API reference generation, with strict
  broken-link and warning handling.

**Exit:** a wheel installs, config validation works, all repository gates run in
CI, and the numerical/API contracts needed by both sensors are frozen.

### Phase 1 — Shack–Hartmann end-to-end vertical slice (0.1)

- [x] Implement wavefront validation and OPD/phase conversion with all unit,
  shape, non-finite, piston-invariance, and coordinate-sign tests.
- [x] Implement physical-coordinate OPD resampling onto the internal computation
  grid when input dimensions do not match the lenslet partition. Validate planes,
  low-order Zernikes, boundary behavior, precision, and absence of wrapped-phase
  interpolation artifacts.
- [x] Implement sampled circular/annular pupils with obscuration and spiders;
  verify sampled area convergence as resolution/supersampling increases.
- [x] Partition the pupil into a square lenslet grid, preserve partial lenslet
  illumination, and produce explicit lenslet-validity/illumination maps.
- [x] Implement flux-normalized Fraunhofer propagation of all subapertures using
  batched FFT axes—no Python loop over lenslets on the steady-state hot path.
- [x] Integrate oversampled spot intensity into native detector pixels with a
  charge/flux-conserving operation rather than image interpolation.
- [x] Assemble the spot mosaic with fixed axis/sign/layout conventions and report
  per-subaperture captured flux.
- [x] Implement direct detector-surface photon-rate normalization and confirm the
  ideal image sum matches the requested rate when uncropped.
- [x] Implement the narrow `getframes` adapter: preset/inline camera, ROI
  resolution validation, exposure, temperature, binning, precision, truth,
  provenance, and seeded frame generation.
- [x] Implement `WavefrontSensor`, `WavefrontSensor.from_toml`, `photon_rate`,
  `reference`, `expose`, `expose_many`, and one-shot `simulate`; freeze only these
  after user testing.
- [x] Add analytic validation: zero-OPD symmetry, piston invariance, total and
  per-lenslet flux, square-aperture diffraction profile, and spot displacement
  under a known phase ramp.
- [x] Add independent brute-force small-grid comparisons for random phase maps so
  vectorization cannot conceal an indexing/transposition error.
- [x] Add seeded end-to-end tests showing expected `getframes` QE, Poisson/read
  statistics, binning shape, saturation, and reproducibility without duplicating
  getframes' own unit suite.
- [x] Write the quickstart, configuration reference, units/conventions guide,
  Shack–Hartmann physics guide, detector handoff guide, and API docs.
- [x] Add a deterministic quickstart example that saves phase, ideal rate, noisy
  ADU, and a spot-row profile as plots; smoke-run it in CI at reduced resolution.
- [x] Establish SH benchmarks for at least 20x20 and 60x60 lenslets, float32 and
  float64, cold construction and warm per-frame paths, with environment metadata.

**Exit:** a TOML file plus OPD array produces a quantitatively validated noisy
Shack–Hartmann frame through `getframes`.

### Phase 2 — Shack–Hartmann fidelity, radiometry, and LGS (0.2)

- [x] Add magnitude normalization by calling `getframes` bandpass/telescope
  radiometry; test Pogson scaling, collecting area, obscuration, throughput, and
  direct-rate equivalence.
- [x] Add custom transmission/SED inputs and deterministic wavelength quadrature.
  Test monochromatic convergence, broadband flux, and diffraction scaling.
- [x] Complete pupil support: custom masks, pupil rotation, segment gaps, static
  WFS-path OPD, lenslet-grid rotation/offset, and sampled partial subapertures.
- [x] Add finite NGS angular size and user-supplied source kernels as incoherent
  sums/convolutions with flux conservation and centroid tests.
- [x] Add physical lenslet/relay configuration and cross-check the derived spot
  sampling and tilt displacement against the normalized configuration.
- [x] Add field stops, detector margins, optical blur, and explicit launched vs
  captured flux accounting. Cropping must never silently renormalize.
- [x] Add LGS sodium range/density profiles, launch position, and geometric
  subaperture-dependent elongation. Validate direction, length, centroid, flux,
  central-launch symmetry, and the thin-layer limit analytically.
- [x] Document the mean-altitude OPD approximation and show the division of labor:
  `pyturb.Atmosphere(..., lgs_altitude=mean_range)` supplies cone-effect OPD;
  `makewfs` supplies detector-plane sodium elongation.
- [ ] Resolve the wavelength-resolved detector-QE gate in section 13: the
  broadband SH spectrum varies materially across pixels, and the
  `Camera.expose_spectral` cube API is implemented on getframes `main` (slated
  for the 2.1.1 release) and already exercised by the makewfs spectral test
  against the local editable checkout. Released `getframes 2.1.0` does not carry
  it, so pin `getframes>=2.1.1` once it releases before claiming the full gate;
  until then makewfs uses the QE-weighted integrated fallback.
- [x] Cross-validate selected monochromatic SH cases against an independent
  Fourier-optics calculation; the small-grid direct DFT reference is frozen in
  the core test suite, while optional package references remain validation-only.
- [x] Add the NGS magnitude-series, realistic pupil, polychromatic, and LGS worked
  examples listed in section 10.
- [x] Extend benchmarks across wavelength and sodium-range sample counts; document
  accuracy/performance choices and stable convergence defaults.

**Exit:** the SH path covers the realistic source, pupil, radiometry, and sampling
choices needed for AO design studies, with LGS limitations stated honestly.

### Phase 3 — pyramid WFS end-to-end (0.3)

- [x] Implement the ideal four-face focal-plane phase mask using a centred,
  unit-tested coordinate system and explicit quadrant/face naming.
- [x] Propagate pupil field → focal plane → pyramid mask → re-imaged pupil plane
  with correct FFT normalization, padding, focal extent, and output sampling.
- [x] Support separated and intentionally overlapping pupil images without
  assuming each detector pixel belongs to only one geometric pupil.
- [x] Implement unmodulated mode and deterministic circular modulation as an
  incoherent weighted sum of tip/tilt samples. Batch or memory-bound chunks run
  through the FFT; there is no Python loop over output pixels.
- [x] Implement detector mosaic/cropping and source normalization through the same
  common radiometry and `getframes` adapter as SH.
- [x] Validate zero-phase four-pupil symmetry, piston invariance, flux
  conservation, phase-ramp sign, pupil ordering, and convergence with focal-grid
  and modulation sampling.
- [x] Validate small-signal push/pull responses for tip, tilt, focus, and selected
  Zernikes: response is antisymmetric near zero; modulation increases linear
  range while reducing low-order sensitivity.
- [x] Cross-validate fixed ideal-pyramid cases and response maps against HCIPy
  within documented discretization tolerances. HCIPy remains validation-only.
- [x] Add broadband propagation. If a physical pyramid mask is enabled, implement
  refractive-index/apex-angle chromatic separation and validate it separately
  from ideal fixed-separation mode.
- [x] Add optional finite source extent and static path OPD through the common
  incoherent-source machinery.
- [x] Write the pyramid physics/sampling/modulation guide, configuration reference,
  validation discussion, performance guide, and API docs.
- [x] Add unmodulated/modulated PWFS and fair SH-versus-PWFS worked examples.
- [x] Establish pyramid benchmarks at representative 40-, 60-, and 80-pixel pupil
  samplings with 1, 8, and 32 modulation samples, warm/cold and both precisions.

**Exit:** the same public API and detector pipeline produce validated unmodulated
or modulated pyramid frames.

### Phase 4 — workflows, examples, and closed-loop ergonomics (0.4)

- [x] Implement `.npy`/`.npz`/FITS phase readers and ideal/frame writers in the
  CLI without making file I/O part of the hot `expose` path.
- [x] Make configuration-relative paths resolve relative to the TOML file, not the
  process working directory. Record file hashes for masks/curves/static OPD.
- [x] Add iterator-based phase sequences and deterministic seed derivation; verify
  one-frame-at-a-time operation has bounded memory and cached optics.
- [x] Add `expose_integrated` for one detector exposure containing uniformly
  weighted temporal phase samples. Verify it averages ideal intensities (never
  complex fields or phase), invokes `getframes` once, conserves mean photon rate,
  and remains distinct from the multiple-read `expose_many` path.
- [x] Add lightweight timing/provenance hooks useful to a closed-loop driver,
  without adding controller/DM/reconstructor abstractions.
- [x] Add `pyturb` and `getframes` compatibility tests pinned to their supported
  public contracts. Test minimum supported and current local versions in CI where
  practical.
- [x] Create every worked example in section 10 as a script plus TOML config. Each
  saves plots non-interactively, has a fast CI smoke mode, and records its seed.
- [x] Add an example showing the closed-loop injection point with user-supplied
  `residual_opd -> wfs.expose(residual_opd)` callbacks. Any toy reconstructor or
  controller stays inside the example and is labeled non-production.
- [x] Build a documentation gallery from versioned example outputs. Check that
  plots have units, color bars, parameter summaries, accessible labels, and no
  scientifically ambiguous normalization. The versioned SVG and manifest are
  under `docs/gallery/`.
- [x] Add troubleshooting guides for sampling/aliasing, phase units/signs,
  undersized detectors/cropping, flux/QE, LGS approximation, performance, and
  reproducibility.

**Exit:** configuration-driven examples demonstrate the full intended workflow,
including moving turbulence and realistic detectors, while core `makewfs` still
accepts only wavefront plus config.

### Phase 5 — CPU optimization and GPU-ready boundaries (0.5)

- [x] Profile warm SH and PWFS paths before optimizing; store profiles/benchmark
  metadata and identify FFT, assembly, quadrature, and detector costs separately.
- [x] Ensure static pupil/lenslet masks, phase ramps, pyramid masks, quadrature,
  index maps, and detector objects are built once and cached immutably.
- [x] Remove avoidable Python loops and temporaries from hot paths; use batched
  FFTs, in-place-safe operations, and configurable chunks where the full batch
  would exceed a memory budget. Source-state iteration remains intentionally
  bounded to avoid materializing a potentially large wavelength/source cube.
- [x] Verify float32/complex64 does not accidentally promote in hot operations;
  quantify its image/response error against float64 and document when to select
  each precision.
- [x] Add `scipy.fft` worker control and confirm determinism/tolerances across
  worker counts and supported platforms.
- [x] Establish performance-regression checks from stable benchmark kernels. CI
  should compare relative kernels or generous envelopes, not fragile wall-clock
  promises from shared runners.
- [x] Audit backend leakage: `ArrayBackend` now owns runtime array allocation,
  reductions, FFTs, interpolation, and optical blur hooks; sensor modules have
  no direct NumPy allocation/FFT/reduction calls. File I/O, config, metadata,
  source quadrature, and the CPU `getframes` boundary remain explicit host
  operations. An AST guard and injected-CPU parity tests enforce the contract.
- [x] Implement a private experimental CuPy optical backend after the audit,
  with CPU/GPU image and response parity tests. The optional `gpu` extra uses
  CUDA 12 CuPy; `WavefrontSensor(..., _backend=cupy_backend())` is intentionally
  private, and the detector boundary performs one explicit device-to-host copy.
  Public GPU support remains gated on an end-to-end detector design.
- [x] Publish benchmark tables with hardware, dependency versions, precision,
  input size, modulation/wavelength samples, construction time, warm latency,
  throughput, and Python-level peak memory. The snapshot documents that C-level
  allocator accounting remains a future metric.

**Exit:** CPU behavior is measured and optimized, and the optical kernel can gain
a real GPU backend without an architectural rewrite.

### Phase 6 — 1.0 validation and release

- [x] Create `validation/run.py` to produce a deterministic visual gallery and
  machine-readable metrics for analytic sensor response, flux, convergence,
  broadband behavior, LGS geometry, and reference cross-checks.
- [x] Run validation in CI and upload the gallery/JSON as artifacts. Physics
  tolerances cite a derivation, paper, or independent implementation and explain
  sampling limitations.
- [x] Achieve the agreed branch-coverage threshold with meaningful assertions;
  do not cover numerical code with smoke-only tests.
- [x] Run the complete quality gate on every supported platform and Python,
  including wheel installation and all shipped configuration/example smoke tests.
- [x] Complete README, quickstart, concepts, configuration reference, both sensor
  guides, radiometry/detector/interop guides, examples, validation, performance,
  API, contributing, changelog, citation, and release documentation.
- [x] Audit every public name and config key. Remove accidental public surfaces,
  document stability, and record intentional future extension points.
- [x] Verify clean-room installation using non-editable versioned `pyturb` and
  `getframes` wheels, not editable sibling checkouts; record minimum compatible
  scalar versions (`pyturb>=1.0`, `getframes>=2.0`). Released `getframes 2.1.0`
  provides binning but not the spectral cube; that cube lands in getframes 2.1.1
  (already on `main`), so the optional spectral-QE full-truth path passes against
  the local editable checkout but stays behind a version guard until 2.1.1
  releases and can be pinned.
- [x] Build and inspect sdist/wheel contents and run package metadata checks.
- [ ] Tag `1.0.0`, publish docs, and cut a reproducible release.

**Exit:** both sensors and all promised examples satisfy the definition of success
in section 1, with a stable minimal API and no copied atmosphere/detector physics.

### Phase 7 — post-1.0 sensor expansion

Only begin a new sensor after a short design note identifies its image-formation
operator, independent validation source, configuration vocabulary, and why it
cannot be represented as an existing engine variant.

- [ ] Generalize the pyramid engine to three-face, roof, flattened, and other
  Fourier-mask variants while preserving the 1.0 four-face config.
- [ ] Add correlation Shack–Hartmann workflows for strongly elongated LGS spots
  only if image generation (not centroiding/reconstruction) needs new core output.
- [ ] Evaluate Zernike phase-contrast and curvature WFS engines using the common
  propagation, radiometry, detector, and validation contracts.
- [ ] Add sensor registration/manufacturing defect models only from measured or
  well-sourced parameterizations, all disabled by default.
- [ ] Consider amplitude input/scintillation only as a versioned extension to the
  input contract and only in coordination with the atmosphere/propagation owner.

## 9. Verification strategy

### Test layers

1. **Unit tests** cover config validation, units, pupil sampling, coordinate
   transforms, FFT normalization, rebinning, layout, metadata, and error paths.
2. **Invariant/property tests** cover piston invariance, non-negative intensity,
   flux conservation, scale linearity with photon rate, and deterministic seeds.
3. **Analytic physics tests** cover SH tilt displacement and diffraction, LGS
   elongation geometry, pyramid symmetry and small-signal behavior, and wavelength
   scaling.
4. **Independent-reference tests** compare small fixed cases to a direct
   implementation and selected cases to HCIPy. Reference packages are never
   runtime dependencies.
5. **Statistical integration tests** sample enough `getframes` frames to verify
   mean/variance at the boundary with uncertainty-aware tolerances.
6. **Example smoke tests** run every public workflow cheaply and prove configs,
   imports, output paths, and plotting stay valid.
7. **Benchmarks** measure cold setup and warm frame time separately and guard
   against major regression.

### Mandatory physical assertions

- Zero phase yields the documented reference image.
- Adding integer or non-integer constant piston leaves intensity unchanged.
- An uncropped optical system conserves photons within a stated tolerance.
- Multiplying source photon rate multiplies ideal intensity exactly.
- A known x/y OPD ramp produces the analytic SH displacement with the documented
  sign and does not transpose axes.
- SH lenslets do not exchange flux unless the configured optical model says they
  do; partial pupil illumination is represented rather than silently discarded.
- Pyramid pupil ordering and differential response signs are stable and named.
- Increasing pyramid modulation sampling converges; increasing modulation radius
  demonstrates the expected sensitivity/linearity trade rather than merely a
  different-looking image.
- LGS elongation is zero for a zero-thickness sodium layer, is radial relative to
  launch geometry, and increases with subaperture-launch baseline and layer
  thickness according to the geometric small-angle prediction.
- Float32 agrees with float64 within documented science tolerances.
- Seeded detector frames repeat exactly on a fixed backend; unseeded repeated
  calls advance state.

Numerical tolerances must be justified by grid sampling or Monte Carlo
uncertainty. Never loosen a tolerance only to make CI green.

## 10. Worked example matrix

Every example has a TOML file, a runnable script, explanatory narrative, and
saved plots. It uses `matplotlib` only through the examples extra.

1. **Five-minute quickstart** — a configured low-order phase map through a 10x10
   SH sensor: input OPD, ideal spots, noisy detector frame, line profile.
2. **Moving atmosphere through a real detector** — `pyturb` frozen-flow OPD →
   20x20 SH → OCAM2K; show phase and ADU frames over time plus flux/latency traces,
   and demonstrate optional intra-exposure OPD integration.
3. **One camera, several guide-star magnitudes** — same optics, exposure, camera,
   and seeds across a magnitude grid; show spot mosaics, photon budgets,
   saturation/faint limits, and measured image statistics without implementing a
   production centroid estimator.
4. **Shack–Hartmann design trade** — lenslet count, pixels per subaperture,
   wavelength, field stop, and detector binning; show sampling and captured flux.
5. **Realistic pupils and broadband sensing** — obstruction/spiders/segments,
   wavelength-dependent spots, custom throughput/QE, and static WFS-path OPD.
6. **What an LGS looks like** — NGS versus centre-launched and side-launched
   sodium LGS; show subaperture-dependent elongation and use `pyturb` mean-altitude
   cone-effect OPD with the approximation labeled.
7. **Pyramid modulation** — the focal mask and four pupil images at zero, small,
   and large modulation radius; plot push/pull Zernike response and linear range.
8. **Pyramid versus Shack–Hartmann** — same pupil OPD sequence, source photon rate,
   exposure, passband, detector noise assumptions, and comparable spatial
   sampling; compare raw images, differential responses, linearity, and compute
   cost without claiming a universal winner.
9. **Closed-loop injection** — external residual OPD enters a persistent sensor
   object each iteration. Show how an external reconstructor/controller consumes
   the returned frame, while keeping those algorithms outside the package.
10. **Detector choice** — reuse identical ideal WFS photon-rate maps across real
    `getframes` presets to isolate detector consequences at a faint magnitude
    where noise character differs. The shipped example uses visible-band CCD,
    EMCCD, sCMOS, and CMOS presets; eAPD/SAPHIRA presets are near-IR (zero QE at
    the visible sensing wavelength) and so are not used in this visible example.
11. **CPU precision and throughput** — float32/float64 accuracy, warm latency,
    batch/chunk choices, and a clear future-GPU boundary.

The full-resolution example gallery may be scheduled rather than run on every PR;
each example also has a small deterministic CI mode.

## 11. Documentation inventory

- `docs/index.md`: purpose, status, boundaries, and navigation.
- `docs/quickstart.md`: install, TOML, first phase-to-frame simulation.
- `docs/concepts.md`: phase/OPD, pupil field, photons, detector samples, SH and
  pyramid principles.
- `docs/configuration.md`: every table/key, units, defaults, constraints, examples,
  schema compatibility, and complete configs.
- `docs/units-and-coordinates.md`: array axes, origin, sign, FFT, phase convention,
  wavelength, pupil/mosaic/pyramid layout.
- `docs/shack-hartmann.md`: physical and normalized models, sampling, partial
  lenslets, field stops, source size, and limitations.
- `docs/pyramid.md`: propagation, separation, overlap, modulation quadrature,
  chromaticity, sampling, and limitations.
- `docs/guide-stars.md`: photon-rate/magnitude paths, spectral weighting, finite
  NGS, LGS geometry, sodium input, and non-goals.
- `docs/detectors.md`: exact `getframes` boundary, presets, QE, binning,
  backgrounds, truth, calibration, and saturation.
- `docs/interop.md`: direct `pyturb` frames, existing arrays, closed-loop residuals,
  FITS/NumPy, and device-copy guidance.
- `docs/examples.md`: gallery with configs and reproducibility instructions.
- `docs/validation.md`: theory/reference comparisons, tolerances, known numerical
  limitations, and how to reproduce artifacts.
- `docs/performance.md`: benchmarks, precision, caching, memory, FFT workers, and
  GPU roadmap.
- `docs/api.md`, `docs/contributing.md`, `docs/stability.md`, and ADRs.

Documentation is part of the feature. A user-facing config field is incomplete if
it appears only in a dataclass docstring.

## 12. Performance and future GPU design

### CPU rules from the first implementation

- Construct the sensor once. Static grids, masks, phase ramps, normalization,
  index maps, and camera config are immutable and cached.
- Express SH as a batch of subaperture FFTs and PWFS modulation/wavelength/source
  samples as a batch or bounded chunks. Avoid per-lenslet/output-pixel loops.
- Separate construction benchmarks from warm per-frame latency.
- Preserve float32/complex64 on the fast path and retain float64/complex128 for
  validation and accuracy-sensitive work.
- Track peak memory as well as throughput. A faster method that cannot fit a
  representative 60x60 SH or modulated PWFS case is not a general win.
- Normalize FFTs and photon flux explicitly; backend defaults must not define the
  physics.

### GPU path

`pyturb` already emits CuPy arrays. The optical core now keeps array creation,
FFTs, reductions, indexing, and scalar access behind `ArrayBackend`; the private
CuPy path is parity-tested but not part of the public configuration/API.

Full device-resident phase → ADU will additionally require a GPU-capable
`getframes` signal chain. Until that exists, an experimental GPU optical path
ends in one explicit device-to-host copy at the detector adapter. This is
acceptable for exploration but is not advertised as end-to-end GPU support.

## 13. Conditional sibling-repository work

These are not automatic `makewfs` tasks. Open a focused issue/PR in the owning
repository only if the preceding phase proves the need.

### getframes: wavelength-resolved rate-map exposure

- [x] **Gate:** demonstrate a broadband WFS whose spatial spectrum changes across
  detector pixels, so a single scalar effective QE gives a materially wrong
  result.
- [x] If gated in, add a public `getframes` API accepting either a wavelength +
  photon-rate cube or a pre-QE electron-rate map with correct `FrameTruth`
  semantics. Apply QE/PRNU/noise exactly once and test equivalence to scalar
  monochromatic exposure.
- [ ] Release and pin the first `getframes` version with that contract before
  using it from `makewfs`. The contract (`Camera.expose_spectral` + spectral
  `FrameTruth`) is implemented on getframes `main` and slated for **2.1.1**;
  released 2.1.0 did not include it. This stays open until 2.1.1 is released and
  makewfs pins `getframes>=2.1.1` for the spectral path.

The current compatibility path precomputes a QE-weighted electron rate and calls
`Camera.expose(..., quantum_efficiency=1.0)` only when the released camera lacks
the public cube API. The preferred path calls `Camera.expose_spectral` and keeps
the incident photon cube in `FrameTruth`; `makewfs` must not depend on private
helpers or mislabel electron rate as photon rate.

### pyturb: same-realization multi-range LGS OPD

- [ ] **Gate:** require range-resolved turbulent OPD across sodium slices, beyond
  the documented mean-altitude approximation used for v1 images.
- [ ] If gated in, add a public `pyturb` readout that evaluates the same layered
  atmospheric realization at multiple finite beacon ranges in one call. Validate
  the thin-layer/mean-range limits, per-layer cone magnification, time coherence,
  and CPU/GPU behavior.
- [ ] Release and pin the first `pyturb` version with that contract before using it
  in an example or claiming range-resolved LGS turbulence.

### getframes: device-resident detector path

- [ ] **Gate:** a measured end-to-end GPU use case is dominated by the host copy or
  CPU detector chain after the `makewfs` optical kernel is accelerated.
- [ ] If gated in, design GPU support in `getframes` itself, including its RNG,
  stochastic distributions, fixed patterns, truth arrays, and CPU/GPU statistical
  parity. Do not recreate a second detector in `makewfs`.

## 14. Scientific references and independent implementations

Implementation agents must read the relevant primary source before adding a
physics model and cite it in the module/doc guide. Starting points:

- O. Fauvarque et al., *General formalism for Fourier based Wave Front Sensing:
  application to the Pyramid Wave Front Sensors* (2016):
  <https://arxiv.org/abs/1607.03269>.
- E. H. Por et al., *High Contrast Imaging for Python (HCIPy)* (2018), including
  simulated SH and pyramid images: <https://doi.org/10.1117/12.2314407>.
- HCIPy's maintained pyramid tutorial is a useful independent implementation and
  sampling comparison, not a runtime dependency:
  <https://docs.hcipy.org/dev/tutorials/PyramidWFS/PyramidWFS.html>.
- V. Akondi, S. Steven, and A. Dubra, *Centroid error due to non-uniform lenslet
  illumination in the Shack-Hartmann wavefront sensor* (2019), for the relation
  between lenslet illumination, average slope, and diffraction-model centroids:
  <https://doi.org/10.1364/OL.44.004167>.
- L. Schreiber et al., *Laser guide stars for extremely large telescopes:
  efficient Shack–Hartmann wavefront sensor design using the weighted
  centre-of-gravity algorithm* (2009), for sodium profiles, launch geometry,
  small-angle elongation, and subaperture images:
  <https://doi.org/10.1111/j.1365-2966.2009.14797.x>.

Avoid copying equations from secondary summaries when the original paper or
maintained reference implementation is available.

## 15. Agent execution order and definition of done

The critical path is Phase 0 → Phase 1 → Phase 2 and Phase 3 → Phase 4 → Phase 5
→ Phase 6. Within a phase, independent docs, tests, examples, and implementation
tasks may run in parallel only after their shared contract/ADR is accepted.

Good agent-sized work items are narrow vertical changes such as “validated OPD
conversion and tests,” “batched SH FFT against direct reference,” or “getframes
adapter plus integration test.” Avoid a task titled only “implement Phase 2.”

An item is done only when:

1. The implementation respects the ownership boundaries and stable contracts.
2. Unit and integration tests cover success, failure, units, and signs.
3. Physics behavior has an analytic or independent-reference assertion where
   applicable.
4. The public docstring and relevant guide/config reference are updated.
5. A performance-sensitive change includes or updates a benchmark.
6. Ruff, format check, strict mypy, pytest/coverage, docs strict build, and package
   build pass locally.
7. `CHANGELOG.md` is updated and this roadmap checkbox is changed only after all
   preceding conditions pass.
8. The handoff states assumptions, commands run, results, and any remaining
   conditional upstream need.

If a proposed change crosses the atmosphere, detector, reconstruction, or
controller boundary, stop and update the roadmap/design note before writing code.
