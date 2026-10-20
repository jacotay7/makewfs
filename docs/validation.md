# Validation

Every optical feature must have an invariant, analytic, or independent-reference
test. The verification is deliberately layered so an external package cannot
hide an error shared by similar FFT implementations.

| Layer | Shack-Hartmann evidence | Four-face pyramid evidence |
| --- | --- | --- |
| Invariants | non-negative intensity, piston invariance, flux bounds, axis/layout tests | non-negative intensity, piston invariance, quadrant symmetry, face order, flux bounds |
| Analytic | square-aperture diffraction, absolute centroid displacement for a known OPD slope, and windowed `sinc^2` flux of fully illuminated lenslets across spot samplings and fields of view (no aliased replicas) | small-signal push/pull antisymmetry and the modulation sensitivity/linearity trade |
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

[OOPAO](https://github.com/cheritier/OOPAO) (1.0 or later, from PyPI) is
supplementary rather than a CI dependency. Its diffractive SH raw frame stores detector axes transposed
relative to makewfs; after that explicit mapping, both axis curves must correlate
above 0.999 and their gains must agree within 1%. Its unmodulated pyramid
tip/tilt/focus maps must correlate above 0.9 with response norms within 25%.
OOPAO exercises a different end-to-end object model and is valuable corroboration,
but its API and optional CuPy behavior are not stable enough to make it a core
dependency (1.0 changed the Shack-Hartmann measurement call, for example).

Run the optional reference test with:

```bash
python -m pip install -e ".[dev,validation]"
python -m pytest -q -m validation
```

On Python 3.11 and later the `validation` extra also installs OOPAO (its pyFFTW
dependency has no Python 3.10 wheels). Without OOPAO 1.0 or later, its two tests
skip cleanly:

```bash
python -m pip install "OOPAO>=1.0"
python -m pytest -q tests/test_oopao_validation.py
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

## Fast LGS kernel mode

`validation/lgs_kernel.py` renders one instrument with
`source.lgs_model = "quadrature"` and `"kernel"` and reports image, centroid
(a validation-only observable), and captured-flux differences plus warm frame
times. The default instrument is
`benchmarks/configs/shack_hartmann_20x20_lgs_quadrature.toml`: a 20 x 20
sensor at 589 nm, 16 pixels of 0.15 arcsec per subaperture, a 1 arcsec FWHM
source, a 10 km FWHM sodium layer sampled at five ranges about 90 km, a launch
4.5 m off axis, and 30 nm RMS of random OPD. Results in float64
(`python validation/lgs_kernel.py`):

| Reference | Propagated states | Image L1 difference | Peak difference | Max / RMS centroid difference | Captured fraction (reference / kernel) |
| --- | --- | --- | --- | --- | --- |
| Dense quadrature (Gaussian source sampled every 0.5 sigma to 4 sigma) | 1445 | 0.14% | 0.34% | 0.0017 / 0.0007 px | 0.8418 / 0.8416 |
| Configured quadrature (3 x 3 Gauss-Hermite source) | 45 | 70% | 75% | 0.21 / 0.08 px | 0.8407 / 0.8416 |

The kernel matches the dense reference, which isolates the kernel
approximation. The large difference from the configured quadrature is that
quadrature's own error: a 1 arcsec source spans about seven pixels, and nine
Gauss-Hermite points render it as a 3 x 3 grid of spots. The kernel is kept
non-negative and conserves weight and first moment by construction.

`tests/test_lgs_kernel.py` asserts, against the quadrature on an 8 x 8 sensor:

- without source extent (both models translate the same spot), for centre,
  side, and opposite-side launches and a 10-degree rotated lenslet grid: image
  L1 difference below 4%, centroids within 0.01 pixel, and per-subaperture flux
  within 0.2% of the brightest subaperture. The remaining difference is the
  linear deposition, and it converges at second order in the focal sampling:
  3.2%, 0.73%, and 0.19% at `fft_oversampling` 2, 4, and 8 for a side launch
  (the test asserts the factor-of-four drop from 2 to 4);
- a thin beacon with a 0.25 arcsec source against a dense Gaussian quadrature:
  L1 difference below `1e-4`;
- a unit-impulse kernel (thin beacon, no extent) reproduces the point-source
  image to `1e-12`;
- truncation: with streaks longer than the window, the captured rate matches
  the quadrature to 0.2%, equals the sum of the photon-rate map, and is below a
  thin beacon's; each kernel holds unit weight before truncation;
- centre-launch mirror and transpose symmetry, radial elongation within 3
  degrees, float32 versus float64, two wavelengths with a field offset, offset
  states, and temporal integration; CPU/GPU parity is a `gpu` test.

The remaining independent-reference gap is broader chromatic, partial-pupil,
finite-source, and sodium-LGS coverage. Those models retain analytic and
convergence tests, but the current external-package comparisons intentionally use
small monochromatic NGS cases whose conventions can be matched unambiguously.
