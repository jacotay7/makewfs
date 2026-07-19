# ADR 0002: flux and detector normalization

Status: accepted

Optical engines return a non-negative photon-rate map in photons/s/native
detector pixel. A configured direct rate is interpreted at the detector surface;
magnitude normalization delegates to `getframes` radiometry. Fourier transforms
are unitary, and finite detector cropping reports captured flux instead of
renormalizing it. `getframes.Camera.expose` (or its optional
`expose_spectral` cube variant) is the only boundary that applies QE, shot
noise, read noise, gain, saturation, and digitization.
