# Validation

Every optical feature must have an invariant, analytic, or independent-reference
test. The first Shack-Hartmann slice currently checks:

- OPD/phase conversion and physical-grid resampling;
- circular pupil area and non-negative intensity;
- zero-OPD reference symmetry and constant-piston invariance;
- deterministic seeded detector frames through `getframes`; and
- temporal integration as an average of ideal intensities before one detector
  read.

The pyramid slice additionally checks equal unmodulated pupil quadrants,
modulation flux conservation, push/pull antisymmetry for tilt and focus,
reduced low-order sensitivity under modulation, and explicit face-order
provenance. These are discretized CPU checks, not a claim of independent-
reference parity.

The repository includes an optional HCIPy cross-check for a fixed ideal pyramid
case. It is a validation-only dependency, never a runtime dependency; broader
convergence studies, SH references, and sodium-LGS geometry cross-checks remain
in the roadmap.

Run the optional reference test with:

```bash
python -m pip install -e ".[dev,validation]"
python -m pytest -q -m validation
```

Run the deterministic repository-level report with:

```bash
python validation/run.py --output validation-metrics.json --plot validation.png
```

The JSON records the normalized configuration digest, reference flux, piston
error, warm render time, and pyramid quadrant spread. The independent small-grid
and analytic assertions live in `tests/test_optics_validation.py`; the small
Shack–Hartmann case compares a random phase map against a direct brute-force DFT
through mosaic assembly.
