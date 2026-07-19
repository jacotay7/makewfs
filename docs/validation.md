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
modulation flux conservation, and explicit face-order provenance. These are
discretized CPU checks, not a claim of independent-reference parity.

The roadmap adds analytic tilt displacement, lenslet diffraction, broadband
convergence, sodium-LGS geometry, and pyramid modulation response. HCIPy is a
validation-only dependency, never a runtime dependency.
