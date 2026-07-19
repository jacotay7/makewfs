"""Configuration validation tests."""

from pathlib import Path

import pytest

from makewfs import ConfigError, __version__, load_config
from makewfs.config import (
    DetectorConfig,
    InputConfig,
    NumericsConfig,
    PyramidConfig,
    SensorConfig,
    ShackHartmannConfig,
    SourceConfig,
    SpiderConfig,
    TelescopeConfig,
    WFSConfig,
)

CONFIG = Path(__file__).parents[1] / "examples" / "configs" / "shack_hartmann_minimal.toml"


def test_load_config_and_digest() -> None:
    config = load_config(CONFIG)
    assert config.sensor.kind == "shack_hartmann"
    assert config.input.shape == (128, 128)
    assert len(config.digest) == 16
    assert __version__ == "0.1.0.dev0"


def test_config_round_trip_preserves_digest() -> None:

    config = load_config(CONFIG)
    round_tripped = WFSConfig.from_dict(config.to_dict())
    assert round_tripped.digest == config.digest


def _minimal_tables() -> dict[str, object]:
    return {
        "schema_version": 1,
        "input": {"quantity": "opd", "unit": "m", "shape": [8, 8], "grid_extent_m": 1.0},
        "telescope": {"pupil_diameter_m": 1.0},
        "source": {"normalization": "detector_photon_rate", "detector_photon_rate_per_s": 1.0},
        "sensor": {"kind": "shack_hartmann", "wavelength_m": 700e-9},
        "shack_hartmann": {
            "lenslets_across_pupil": 2,
            "pixels_per_subaperture": 4,
            "spot_sampling_pixels_per_lambda_over_d": 2.0,
            "minimum_illuminated_fraction": 0.25,
        },
        "detector": {"preset": "generic_cmos", "exposure_s": 0.001},
    }


def test_all_shipped_configurations_load() -> None:
    for path in sorted(CONFIG.parent.glob("*.toml")):
        config = load_config(path)
        assert config.source_path == str(path.resolve())


def test_config_rejects_unknown_top_level_key() -> None:
    with pytest.raises(ConfigError, match="unknown key"):
        from makewfs.config import WFSConfig

        WFSConfig.from_dict({"schema_version": 1, "unexpected": True})


def test_phase_input_requires_reference_wavelength() -> None:
    from makewfs.config import WFSConfig

    data = {
        "schema_version": 1,
        "input": {"quantity": "phase", "unit": "rad", "shape": [8, 8], "grid_extent_m": 1},
    }
    with pytest.raises(ConfigError, match="reference_wavelength"):
        WFSConfig.from_dict(data)


def test_direct_rate_and_magnitude_are_mutually_exclusive() -> None:
    from makewfs.config import SourceConfig

    with pytest.raises(ConfigError, match="direct-rate"):
        SourceConfig.from_dict(
            {
                "normalization": "detector_photon_rate",
                "detector_photon_rate_per_s": 1,
                "magnitude": 10,
            }
        )


def test_broadband_source_fields_are_validated() -> None:
    from makewfs.config import SourceConfig

    source = SourceConfig.from_dict(
        {
            "normalization": "detector_photon_rate",
            "detector_photon_rate_per_s": 1,
            "wavelengths_m": [6e-7, 7e-7],
            "wavelength_weights": [1, 3],
            "angular_fwhm_arcsec": 0.4,
            "angular_quadrature_order": 3,
        }
    )
    assert source.wavelength_weights == (1.0, 3.0)
    assert source.angular_quadrature_order == 3


def test_sensor_specific_tables_cannot_be_mixed() -> None:
    from makewfs.config import WFSConfig

    data = load_config(CONFIG).to_dict()
    data["pyramid"] = {
        "pixels_across_pupil": 64,
        "pupil_separation_pixels": 16,
    }
    with pytest.raises(ConfigError, match="pyramid"):
        WFSConfig.from_dict(data)


def test_physical_shack_hartmann_sampling_mode() -> None:
    config = ShackHartmannConfig.from_dict(
        {
            "lenslets_across_pupil": 8,
            "pixels_per_subaperture": 8,
            "minimum_illuminated_fraction": 0.25,
            "lenslet_focal_length_m": 0.02,
            "detector_pixel_pitch_m": 15e-6,
        }
    )
    assert config.spot_sampling_pixels_per_lambda_over_d is None
    assert config.lenslet_focal_length_m == 0.02


@pytest.mark.parametrize(
    ("factory", "data", "message", "call_mode"),
    [
        (
            InputConfig.from_dict,
            {"quantity": "phase", "unit": "m", "shape": [2, 2], "grid_extent_m": 1},
            "requires",
            "base",
        ),
        (
            InputConfig.from_dict,
            {"quantity": "opd", "unit": "m", "shape": [2], "grid_extent_m": 1},
            "shape",
            "base",
        ),
        (SpiderConfig.from_dict, {"angle_deg": 0, "width_fraction": 2}, "width_fraction", "path"),
        (TelescopeConfig.from_dict, {"pupil_diameter_m": 1, "spiders": {}}, "spiders", "base"),
        (SensorConfig.from_dict, {"kind": "roof", "wavelength_m": 700e-9}, "sensor.kind", "plain"),
        (
            SourceConfig.from_dict,
            {"normalization": "unknown", "detector_photon_rate_per_s": 1},
            "normalization",
            "plain",
        ),
        (SourceConfig.from_dict, {"normalization": "detector_photon_rate"}, "direct-rate", "plain"),
        (DetectorConfig.from_dict, {"preset": "generic_cmos"}, "exposure_s", "plain"),
        (NumericsConfig.from_dict, {"dtype": "complex64"}, "dtype", "plain"),
    ],
)
def test_individual_config_tables_reject_invalid_values(factory, data, message, call_mode) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ConfigError, match=message):
        if call_mode == "base":
            factory(data, base=Path.cwd())
        elif call_mode == "path":
            factory(data, "spider")
        else:
            factory(data)


def test_source_magnitude_and_lgs_constraints() -> None:
    with pytest.raises(ConfigError, match="band"):
        SourceConfig.from_dict({"normalization": "magnitude", "magnitude": 12})
    with pytest.raises(ConfigError, match="magnitude_system"):
        SourceConfig.from_dict(
            {
                "normalization": "magnitude",
                "magnitude": 12,
                "band": "R",
                "magnitude_system": "vega2",
            }
        )
    with pytest.raises(ConfigError, match="LGS"):
        SourceConfig.from_dict(
            {
                "kind": "lgs",
                "normalization": "magnitude",
                "magnitude": 12,
                "band": "R",
            }
        )
    with pytest.raises(ConfigError, match="LGS geometry"):
        SourceConfig.from_dict(
            {
                "normalization": "detector_photon_rate",
                "detector_photon_rate_per_s": 1,
                "lgs_ranges_m": [90e3],
            }
        )


def test_source_quadrature_and_curve_fields_are_checked(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="order"):
        SourceConfig.from_dict(
            {
                "normalization": "detector_photon_rate",
                "detector_photon_rate_per_s": 1,
                "angular_fwhm_arcsec": 1,
                "angular_quadrature_order": 1,
            }
        )
    with pytest.raises(ConfigError, match="wavelengths_m"):
        SourceConfig.from_dict(
            {
                "normalization": "detector_photon_rate",
                "detector_photon_rate_per_s": 1,
                "wavelengths_m": [-700e-9],
            }
        )
    with pytest.raises(ConfigError, match="wavelength_weights"):
        SourceConfig.from_dict(
            {
                "normalization": "detector_photon_rate",
                "detector_photon_rate_per_s": 1,
                "wavelengths_m": [700e-9],
                "wavelength_weights": [0, 0],
            }
        )
    source = SourceConfig.from_dict(
        {
            "normalization": "detector_photon_rate",
            "detector_photon_rate_per_s": 1,
            "sed_path": "sed.txt",
            "transmission_path": "filter.txt",
        },
        base=tmp_path,
    )
    assert source.sed_path == str((tmp_path / "sed.txt").resolve())


def test_sensor_specific_validation_and_defaults() -> None:
    with pytest.raises(ConfigError, match="normalized and physical"):
        ShackHartmannConfig.from_dict(
            {
                "lenslets_across_pupil": 2,
                "pixels_per_subaperture": 4,
                "spot_sampling_pixels_per_lambda_over_d": 2,
                "lenslet_focal_length_m": 0.02,
                "detector_pixel_pitch_m": 15e-6,
            }
        )
    with pytest.raises(ConfigError, match="provide"):
        ShackHartmannConfig.from_dict({"lenslets_across_pupil": 2, "pixels_per_subaperture": 4})
    with pytest.raises(ConfigError, match="zero modulation"):
        PyramidConfig.from_dict(
            {"pixels_across_pupil": 8, "pupil_separation_pixels": 2, "modulation_samples": 2}
        )
    with pytest.raises(ConfigError, match="at least four"):
        PyramidConfig.from_dict(
            {
                "pixels_across_pupil": 8,
                "pupil_separation_pixels": 2,
                "modulation_radius_lambda_over_d": 1,
                "modulation_samples": 3,
            }
        )


def test_detector_and_root_tables_reject_conflicts() -> None:
    with pytest.raises(ConfigError, match="preset or inline"):
        DetectorConfig.from_dict(
            {"preset": "generic_cmos", "camera": {"resolution": [8, 8]}, "exposure_s": 1}
        )
    with pytest.raises(ConfigError, match="binning_mode"):
        DetectorConfig.from_dict({"preset": "generic_cmos", "exposure_s": 1, "binning_mode": "bad"})
    with pytest.raises(ConfigError, match="camera: expected a table"):
        DetectorConfig.from_dict({"camera": [], "exposure_s": 1})
    missing_sensor_table = _minimal_tables()
    missing_sensor_table.pop("shack_hartmann")
    with pytest.raises(ConfigError, match="required"):
        WFSConfig.from_dict(missing_sensor_table)
    with pytest.raises(ConfigError, match="expected a table"):
        WFSConfig.from_dict(_minimal_tables() | {"input": []})
