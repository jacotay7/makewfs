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
| `numerics` | device, dtype, FFT oversampling/workers, internal sampling |

Unknown keys and conflicting normalization choices are errors. File paths are
resolved relative to the TOML file. The normalized configuration has a short
SHA-256 `digest` used in frame provenance. All lengths are metres unless the
key explicitly says `arcsec`, `deg`, or `pixels`.

## Complete key reference

The loader rejects omitted required keys, unknown keys, non-finite values, and
values outside the ranges below. Defaults are shown in parentheses.

At the root, `schema_version = 1` is required. An optional `[metadata]` table
accepts arbitrary designer notes and is preserved in the normalized
configuration; it does not affect optical propagation except through the
configuration digest.

### `input`

| Key | Meaning and constraints |
| --- | --- |
| `quantity` | `"opd"` or `"phase"` (default `"opd"`). |
| `unit` | Must be `"m"` for OPD or `"rad"` for phase; units are never inferred. |
| `shape` | Required `[height, width]`, positive integers. Every dynamic input has this shape. |
| `grid_extent_m` | Required physical width/height of the input grid, positive. |
| `reference_wavelength_m` | Required positive wavelength when `quantity = "phase"`; otherwise unused. |
| `static_opd_path` | Optional `.npy`, `.npz`, or FITS OPD map with exactly `shape`; resolved relative to the TOML file. |

### `telescope`

| Key | Meaning and constraints |
| --- | --- |
| `pupil_diameter_m` | Required positive entrance-pupil diameter. |
| `central_obscuration_ratio` | Inner radius divided by outer radius, `[0, 1)`, default `0`. |
| `spiders` | Array of `{angle_deg, width_fraction}` tables (default `[]`); angle is measured from +x and width is a fraction of a half-turn, `[0, 1]`. |
| `pupil_rotation_deg` | Rotation applied to spider angles, default `0`. |
| `custom_mask_path` | Optional `.npy`, `.npz`, or FITS amplitude mask; values must be finite in `[0, 1]` and match the internal sensor grid. |
| `segments_across_pupil` | Optional square segment count (integer ≥2), used with analytic segment gaps. |
| `segment_gap_fraction` | Gap width as a fraction of segment pitch, `[0, 0.99]` (default `0`); nonzero values require `segments_across_pupil`. |

### `source`

| Key | Meaning and constraints |
| --- | --- |
| `kind` | `"ngs"` or `"lgs"` (default `"ngs"`). |
| `normalization` | `"detector_photon_rate"` or `"magnitude"` (default `"detector_photon_rate"`). |
| `detector_photon_rate_per_s` | Non-negative total photons/s at the detector surface before QE; required for direct-rate mode. |
| `magnitude` | Finite magnitude; required for magnitude mode. LGS must not use magnitude mode. |
| `magnitude_system` | `"vega"` or `"ab"` (default `"vega"`); used with `band`. |
| `band` | Required band name for magnitude mode, passed to `getframes` radiometry. |
| `throughput` | Scalar optical throughput in `[0, 1]` (default `1`), used by magnitude radiometry. |
| `field_angle_arcsec` | Source centroid `[x, y]` in arcsec (default `[0, 0]`). |
| `angular_fwhm_arcsec` | Non-negative Gaussian source FWHM (default `0`); nonzero values require quadrature order at least 2. |
| `angular_quadrature_order` | Positive deterministic source quadrature order (default `3`). |
| `angular_kernel_path` | Optional three-column `x_arcsec y_arcsec weight` source kernel; mutually exclusive with nonzero FWHM and resolved relative to the TOML file. |
| `wavelengths_m` | Optional positive wavelength nodes. Empty means the sensor wavelength. |
| `wavelength_weights` | Optional non-negative weights matching `wavelengths_m`; normalized internally. |
| `lgs_ranges_m` | Optional positive sodium range nodes; only valid for LGS. Empty means a thin layer at the configured mean. |
| `lgs_range_weights` | Optional non-negative weights matching `lgs_ranges_m`; normalized internally. |
| `lgs_launch_position_m` | LGS launch `[x, y]` in the entrance-pupil plane (default `[0, 0]`); only valid for LGS. |
| `sed_path` | Optional two-column `wavelength_nm value` source spectrum, non-negative and increasing. |
| `transmission_path` | Optional two-column `wavelength_nm value` transmission curve in `[0, 1]`. |

SED and transmission curves use trapezoid quadrature on their common wavelength
nodes. The resulting normalized wavelength, source-angle, and (for SH) range
states are recorded in provenance.

An angular kernel is an arbitrary incoherent source morphology. Its offsets are
added to `field_angle_arcsec`, its non-negative weights are normalized, and each
sample is propagated independently. For example:

```text
# x_arcsec  y_arcsec  relative_weight
-0.15       0.00      1
 0.00       0.00      2
 0.15       0.00      1
```

This is useful for measured binary-guide-star or resolved-source profiles. The
kernel file hash and all normalized states are included in frame provenance.

### `sensor`, `detector`, and `numerics`

| Table/key | Meaning and constraints |
| --- | --- |
| `sensor.kind` | Required `"shack_hartmann"` or `"pyramid"`. |
| `sensor.wavelength_m` | Required positive reference sensing wavelength. |
| `detector.preset` | Existing `getframes` camera preset; mutually exclusive with `detector.camera`. |
| `detector.camera` | Inline `getframes.CameraConfig` dictionary; mutually exclusive with `preset`. |
| `detector.exposure_s` | Required non-negative exposure time. |
| `detector.temperature_c` | Optional detector temperature in °C. |
| `detector.binning` | Positive integer passed to `getframes`. |
| `detector.binning_mode` | `"digital"` or `"on_chip"` (default `"digital"`). |
| `detector.precision` | `"float32"` or `"float64"` (default `"float64"`). |
| `detector.include_truth` | Boolean (default `true`) controlling detector truth arrays. |
| `detector.qe_curve_path` | Optional configuration-relative two-column `wavelength_nm qe` curve passed to `getframes`; enables wavelength-resolved QE for broadband optical cubes. |
| `detector.roi` | Optional full-detector ROI table with non-negative `left_px`/`top_px` and positive `width_px`/`height_px`. Its shape must match the optical mosaic. |
| `numerics.dtype` | Optical real precision, `"float32"` or `"float64"` (default `"float64"`). |
| `numerics.device` | Execution device, `"cpu"` (default) or `"gpu"`. GPU requires the `makewfs[gpu]` extra and a GPU-capable `getframes`; optical, truth, and ADU arrays stay device-resident. |
| `numerics.fft_oversampling` | Positive FFT integration oversampling (default `2`). Also scales the pyramid propagation grid so diffraction beyond the detector crop is discarded instead of wrapping onto the pupil rims. |
| `numerics.fft_workers` | Positive `scipy.fft` worker count (default `1`). |
| `numerics.pupil_samples_per_lenslet` | Optional integer ≥4 for SH internal pupil sampling; otherwise derived from input shape. |
| `numerics.pupil_supersampling` | Positive analytic pupil boundary sub-sampling factor (default `1`). |

For a device-resident atmosphere → WFS → detector loop:

```toml
[numerics]
device = "gpu"
dtype = "float32"
fft_oversampling = 2
fft_workers = 1  # CPU-only FFT control; accepted but ignored by CuPy FFTs
```

`[shack_hartmann]` keys are:

| Key | Meaning and constraints |
| --- | --- |
| `lenslets_across_pupil` | Positive square lenslet count. |
| `pixels_per_subaperture` | Native detector pixels per lenslet, at least 2. |
| `spot_sampling_pixels_per_lambda_over_d` | Normalized sampling, at least 0.5; mutually exclusive with physical relay fields. |
| `minimum_illuminated_fraction` | Validity threshold in `[0, 1]`. |
| `lenslet_fill_factor` | Square clear fill fraction in `[0, 1]` (default 1). |
| `lenslet_focal_length_m` | Optional positive physical lenslet focal length. |
| `detector_pixel_pitch_m` | Optional positive detector pitch for physical sampling. |
| `relay_magnification` | Positive relay magnification (default 1). |
| `field_stop_radius_lambda_over_d` | Optional non-negative focal-plane field-stop radius. |
| `optical_blur_fwhm_pixels` | Gaussian blur FWHM in native pixels (default 0). |
| `optical_blur_kernel_path` | Optional measured odd-sized blur kernel path; mutually exclusive with Gaussian blur. |
| `detector_margin_pixels` | Non-negative zero-rate mosaic margin. |
| `lenslet_grid_rotation_deg` | Lenslet-frame rotation in degrees (default 0). |
| `lenslet_grid_offset_fraction` | `[x, y]` offset in lenslet-pitch fractions (default `[0, 0]`). |

`[shack_hartmann]` requires positive `lenslets_across_pupil`, at least two
`pixels_per_subaperture`, and `minimum_illuminated_fraction` in `[0, 1]`.
Either `spot_sampling_pixels_per_lambda_over_d` (at least `0.5`) or both
`lenslet_focal_length_m` and `detector_pixel_pitch_m` must be supplied, but
never both. The optional `lenslet_fill_factor` is in `[0, 1]`,
`relay_magnification` is positive, `field_stop_radius_lambda_over_d` is
non-negative when supplied, `optical_blur_fwhm_pixels` is non-negative, and
`optical_blur_kernel_path` may reference a measured odd-sized, non-negative
`.npy`, `.npz`, or FITS PSF kernel; it is normalized to unit sum and is
mutually exclusive with `optical_blur_fwhm_pixels`. `detector_margin_pixels`
is a non-negative integer. The optional
`lenslet_grid_rotation_deg` rotates the lenslet coordinate frame, and
`lenslet_grid_offset_fraction = [x, y]` shifts its origin by fractions of one
subaperture pitch. These two controls use an explicit physical-coordinate
resampling path; the default zero values retain the fast axis-aligned path.

`[pyramid]` requires `pixels_across_pupil` ≥ 8 and positive
`pupil_separation_pixels`. `modulation_radius_lambda_over_d` is non-negative;
zero radius requires exactly one sample, while nonzero modulation requires at
least four `modulation_samples`. `detector_margin_pixels` is non-negative.

`numerics.pupil_supersampling` sub-samples analytic circular, annular, spider,
and segment-gap boundaries before averaging each pupil pixel. Analytic features
are rotated by `pupil_rotation_deg`; custom masks should be supplied already
rotated because they are measured amplitude products. Custom masks may be
`.npy`, `.npz`, or FITS arrays in `[0, 1]`.

## Source normalization and morphology

For laboratory or already-calibrated flux, use:

```toml
[source]
kind = "ngs"
normalization = "detector_photon_rate"
detector_photon_rate_per_s = 2.0e9
```

The rate is photons/s at the detector surface before detector QE. For an NGS,
the alternative is `normalization = "magnitude"` with `magnitude`, `band`, and
`magnitude_system = "vega"` or `"ab"`; this uses `getframes` radiometry.
LGS return flux must be supplied directly because laser return prediction is out
of scope.

Shack–Hartmann sampling can be expressed either directly in normalized units
or with physical lenslet optics. Do not provide both forms:

```toml
[shack_hartmann]
lenslets_across_pupil = 20
pixels_per_subaperture = 8
minimum_illuminated_fraction = 0.25
lenslet_focal_length_m = 0.020
detector_pixel_pitch_m = 15e-6
relay_magnification = 1.0
```

In physical mode, `makewfs` derives pixels per `lambda / D_subaperture` from
the telescope diameter, lenslet count, sensing wavelength, focal length, pixel
pitch, and relay magnification.

`field_stop_radius_lambda_over_d` clips each subaperture's focal-plane field
stop, `optical_blur_fwhm_pixels` applies a flux-spreading optical Gaussian
before detector sampling, and `detector_margin_pixels` pads the assembled SH
mosaic with zero-rate margins. These are optical settings; detector binning and
noise remain in `[detector]`/`getframes`.

Wavelength and source-size quadrature are optional. Weights are normalized, and
intensities—not complex fields—are summed:

```toml
[source]
kind = "ngs"
normalization = "detector_photon_rate"
detector_photon_rate_per_s = 2.0e9
wavelengths_m = [650e-9, 700e-9, 750e-9]
wavelength_weights = [0.2, 0.5, 0.3]
angular_fwhm_arcsec = 0.25
angular_quadrature_order = 3
# Optional two-column wavelength_nm / relative_value files:
# sed_path = "source_sed.txt"
# transmission_path = "filter_transmission.txt"
```

`field_angle_arcsec = [x, y]` is the source centroid. A finite FWHM uses a
deterministic Gaussian quadrature around that centroid. By default the detector
receives one summed photon-rate map and uses its scalar QE. For broadband
scenes, `detector.qe_curve_path` enables wavelength-resolved exposure through
the released `getframes>=2.1.1` spectral cube API, preserving the incident cube
and wavelength nodes in detector truth.

For a sodium LGS, configure a detector-surface return rate and optional range
profile. The range model is currently Shack–Hartmann-specific:

```toml
[source]
kind = "lgs"
normalization = "detector_photon_rate"
detector_photon_rate_per_s = 2.0e8
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
SAPHIRA, EMCCD, sCMOS, and generic teaching cameras. When the optical mosaic is
a hardware ROI rather than the full sensor, preserve its full-detector origin:

```toml
[detector.roi]
left_px = 4
top_px = 4
width_px = 228
height_px = 228
```

The ROI uses unbinned detector pixels, with `left_px`/`top_px` measured from the
full sensor's upper-left corner. Its `(height_px, width_px)` must equal the ideal
optical output shape. `makewfs` passes this geometry to `getframes`, which retains
the preset's native sensor size, evaluates detector effects in full-detector
coordinates, and returns the cropped image. Without `detector.roi`, a differing
optical shape retains the legacy behavior of replacing the camera resolution.
Detector binning and all noise physics remain in `getframes`.
