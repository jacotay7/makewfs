"""Configuration validation tests."""

from pathlib import Path

import pytest

from makewfs import ConfigError, __version__, load_config

CONFIG = Path(__file__).parents[1] / "examples" / "configs" / "shack_hartmann_minimal.toml"


def test_load_config_and_digest() -> None:
    config = load_config(CONFIG)
    assert config.sensor.kind == "shack_hartmann"
    assert config.input.shape == (128, 128)
    assert len(config.digest) == 16
    assert __version__ == "0.1.0.dev0"


def test_config_round_trip_preserves_digest() -> None:
    from makewfs.config import WFSConfig

    config = load_config(CONFIG)
    round_tripped = WFSConfig.from_dict(config.to_dict())
    assert round_tripped.digest == config.digest


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
    from makewfs.config import ShackHartmannConfig

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
