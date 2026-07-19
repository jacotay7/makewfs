# Pyramid wavefront sensor

The planned pyramid engine will propagate the pupil field to a focal-plane
four-face phase mask and back to a re-imaged pupil detector plane. It will support
unmodulated operation and deterministic circular modulation, with intensity
accumulated over modulation samples.

The implementation is staged after the validated Shack-Hartmann vertical slice.
Its face ordering, pupil separation, focal sampling, overlap behavior, and
modulation convergence will be fixed by analytic tests and cross-checked against
the independent HCIPy implementation. See the
[roadmap](https://github.com/jacotay7/makewfs/blob/main/ROADMAP.md) for the
acceptance criteria.
