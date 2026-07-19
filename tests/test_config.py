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


def test_broadband_source_is_explicitly_gated() -> None:
    from makewfs.config import WFSConfig

    data = {
        "schema_version": 1,
        "input": {"quantity": "opd", "unit": "m", "shape": [8, 8], "grid_extent_m": 1},
        "telescope": {"pupil_diameter_m": 1},
        "source": {
            "normalization": "detector_photon_rate",
            "detector_photon_rate_per_s": 1,
            "wavelengths_m": [6e-7, 7e-7],
        },
        "sensor": {"kind": "shack_hartmann", "wavelength_m": 7e-7},
        "shack_hartmann": {
            "lenslets_across_pupil": 1,
            "pixels_per_subaperture": 8,
            "spot_sampling_pixels_per_lambda_over_d": 1,
            "minimum_illuminated_fraction": 0,
        },
        "detector": {
            "camera": {"name": "generic", "pixel_size_um": 15, "read_noise_e": 0},
            "exposure_s": 1,
        },
    }
    with pytest.raises(NotImplementedError, match="broadband"):
        from makewfs import WavefrontSensor

        WavefrontSensor(WFSConfig.from_dict(data))
