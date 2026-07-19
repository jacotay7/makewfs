# ADR 0001: units and coordinates

Status: accepted

Input OPD is metres and input phase is radians with an explicit reference
wavelength. Internally, every sensor converts to OPD metres before applying the
configured sensing wavelength. Arrays use NumPy `(y, x)` order. Coordinates are
centred on the array and increase to the right and downward in image indexing;
physical x/y ramps are tested numerically rather than inferred from a plot.

This keeps closed-loop residual injection independent of the sensing wavelength
and prevents wrapped phase interpolation from changing a physical OPD ramp.
