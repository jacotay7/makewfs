# Validation

Every optical feature must have an invariant, analytic, or independent-reference
test. The verification is deliberately layered so an external package cannot
hide an error shared by similar FFT implementations.

| Layer | Shack-Hartmann evidence | Four-face pyramid evidence |
| --- | --- | --- |
| Invariants | non-negative intensity, piston invariance, flux bounds, axis/layout tests | non-negative intensity, piston invariance, quadrant symmetry, face order, flux bounds |
| Analytic | square-aperture diffraction and absolute centroid displacement for a known OPD slope | small-signal push/pull antisymmetry and the modulation sensitivity/linearity trade |
| Independent arithmetic | random small-grid spots and a full mosaic propagated by a direct DFT | random small-grid pupil → mask → pupil propagation using direct forward and inverse DFTs |
| Independent packages | two-axis, multi-amplitude centroid curves against HCIPy and OOPAO | flat image and tip/tilt/focus response maps against HCIPy and OOPAO |

The direct-DFT tests use explicit transform summations rather than NumPy/SciPy
FFT transforms or the package's centred FFT helper. They are small enough for
the ordinary test suite and protect normalization, shifts, mask signs, mosaic
assembly, and axis ordering even when no optional reference package is installed.

## Independent package comparisons

[HCIPy](https://docs.hcipy.org/) is the maintained CI reference. With HCIPy
0.7.0, the fixed pyramid reference image must correlate above 0.9 and have
normalized RMS error below `1e-4`. Tip, tilt, and focus push/pull maps must
correlate above 0.95 after a fixed 180-degree detector-frame rotation, which
accounts for the packages viewing the pyramid faces from opposite sides. Their
response-norm ratio must remain between 0.8 and 1.25.

The HCIPy Shack-Hartmann comparison samples four positive/negative tilts on both
axes. The centroid curves must correlate above 0.999, cross-axis motion must stay
below 0.01 pixel, and the two axis gains must agree within 1%. The accepted
HCIPy/makewfs gain range is wider (0.55--0.9) because HCIPy performs a Fresnel
propagation on a finer grid that is subsequently pixel-area integrated. The
separate analytic slope test fixes makewfs's absolute pixel scale to 5%, so an
independent-package gain difference cannot redefine the physical sampling.

[OOPAO](https://github.com/cheritier/OOPAO) is supplementary rather than a pinned
CI dependency. Its diffractive SH raw frame stores detector axes transposed
relative to makewfs; after that explicit mapping, both axis curves must correlate
above 0.999 and their gains must agree within 1%. Its unmodulated pyramid
tip/tilt/focus maps must correlate above 0.9 with response norms within 25%.
OOPAO exercises a different end-to-end object model and is valuable corroboration,
but its development checkout/API and optional CuPy behavior are not stable enough
to make it a package dependency.

Run the optional reference test with:

```bash
python -m pip install -e ".[dev,validation]"
python -m pytest -q -m validation
```

To include a local OOPAO checkout, install it or put its repository root on
`PYTHONPATH`; otherwise its two tests skip cleanly:

```bash
PYTHONPATH=/path/to/OOPAO python -m pytest -q tests/test_oopao_validation.py
```

Some OOPAO revisions select CuPy whenever it is importable. The pyramid
comparison skips when that combination has no usable CUDA device rather than
mistaking an external backend initialization failure for a makewfs physics
failure.

Run the deterministic repository-level report with:

```bash
python validation/run.py --output validation-metrics.json --plot validation.png
```

The JSON records the normalized configuration digest, reference flux, piston
error, and warm render time. For Shack-Hartmann it also records predicted and
measured tilt displacement, lenslet-to-lenslet spread, relative scale error, and
cross-axis leakage. For the pyramid it records quadrant spread plus tip, tilt,
and focus normalized response and push/pull antisymmetry. Ordinary tests require
the shipped SH tilt scale error to stay below 5%, its cross-axis leakage below
0.01 pixel, and each pyramid antisymmetry residual below 3%.

The remaining independent-reference gap is broader chromatic, partial-pupil,
finite-source, and sodium-LGS coverage. Those models retain analytic and
convergence tests, but the current external-package comparisons intentionally use
small monochromatic NGS cases whose conventions can be matched unambiguously.
