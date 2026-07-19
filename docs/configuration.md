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

## Source normalization

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

## Detector choices

`detector.preset` names any installed `getframes` preset, including OCAM2K,
SAPHIRA, EMCCD, sCMOS, and generic teaching cameras. The optical mosaic is
configured as a detector ROI when its dimensions differ from the preset's full
resolution. Detector binning and all noise physics remain in `getframes`.
