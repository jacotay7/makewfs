# Configuration

TOML is the canonical configuration format. A complete example is shipped at
`examples/configs/shack_hartmann_minimal.toml`.

The root requires `schema_version = 1` and has these tables:

| Table | Purpose |
| --- | --- |
| `input` | quantity, units, array shape, physical extent, static OPD |
| `telescope` | pupil diameter, obstruction, spiders, custom mask |
| `source` | NGS/LGS, photon rate or magnitude, field angle, throughput |
| `sensor` | sensor kind and sensing wavelength |
| `shack_hartmann` | lenslet and spot sampling |
| `pyramid` | pupil separation and modulation |
| `detector` | `getframes` preset, exposure, temperature, binning |
| `numerics` | dtype, FFT oversampling/workers, internal sampling |

Unknown keys and conflicting normalization choices are errors. File paths are
resolved relative to the TOML file. The normalized configuration has a short
SHA-256 `digest` used in frame provenance.

## Source normalization and morphology

For laboratory or already-calibrated flux, use:

```toml
[source]
kind = "ngs"
normalization = "detector_photon_rate"
detector_photon_rate_per_s = 2.0e6
```

The rate is photons/s at the detector surface before detector QE. For an NGS,
the alternative is `normalization = "magnitude"` with `magnitude`, `band`, and
`magnitude_system = "vega"` or `"ab"`; this uses `getframes` radiometry.
LGS return flux must be supplied directly because laser return prediction is out
of scope.

Wavelength and source-size quadrature are optional. Weights are normalized, and
intensities—not complex fields—are summed:

```toml
[source]
kind = "ngs"
normalization = "detector_photon_rate"
detector_photon_rate_per_s = 2.0e6
wavelengths_m = [650e-9, 700e-9, 750e-9]
wavelength_weights = [0.2, 0.5, 0.3]
angular_fwhm_arcsec = 0.25
angular_quadrature_order = 3
```

`field_angle_arcsec = [x, y]` is the source centroid. A finite FWHM uses a
deterministic Gaussian quadrature around that centroid. The detector still
receives one summed photon-rate map, so wavelength-dependent detector QE is
represented by the configured scalar camera QE until the conditional getframes
gate in the roadmap is met.

For a sodium LGS, configure a detector-surface return rate and optional range
profile. The range model is currently Shack–Hartmann-specific:

```toml
[source]
kind = "lgs"
normalization = "detector_photon_rate"
detector_photon_rate_per_s = 5.0e5
lgs_ranges_m = [89000.0, 90000.0, 91000.0]
lgs_range_weights = [0.25, 0.5, 0.25]
lgs_launch_position_m = [0.0, 0.0]
```

The supplied OPD is interpreted at the weighted mean range. Each range slice
gets the geometric perspective offset for each lenslet; this captures image
elongation but does not create range-resolved turbulent OPD. See the
[guide-star guide](guide-stars.md) for the approximation and `pyturb` handoff.

## Detector choices

`detector.preset` names any installed `getframes` preset, including OCAM2K,
SAPHIRA, EMCCD, sCMOS, and generic teaching cameras. The optical mosaic is
configured as a detector ROI when its dimensions differ from the preset's full
resolution. Detector binning and all noise physics remain in `getframes`.
