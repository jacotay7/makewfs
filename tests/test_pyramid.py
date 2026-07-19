"""Pyramid optical and detector integration tests."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from makewfs import WavefrontSensor

CONFIG = Path(__file__).parents[1] / "examples" / "configs" / "pyramid_minimal.toml"


def test_unmodulated_reference_has_four_equal_pupils() -> None:
    sensor = WavefrontSensor.from_toml(CONFIG)
    reference = sensor.reference()
    assert reference.shape == (84, 84)
    assert np.all(reference >= 0)
    assert np.isclose(reference.sum(), 2.0e6, rtol=1e-6)
    quadrants = [
        reference[:42, :42],
        reference[:42, 42:],
        reference[42:, :42],
        reference[42:, 42:],
    ]
    assert np.allclose([quadrant.sum() for quadrant in quadrants], 5.0e5, rtol=1e-5)


def test_modulation_is_deterministic_and_flux_preserving() -> None:
    config = WavefrontSensor.from_toml(CONFIG).config
    assert config.pyramid is not None
    modulated = replace(
        config,
        pyramid=replace(config.pyramid, modulation_radius_lambda_over_d=2.0, modulation_samples=8),
    )
    sensor = WavefrontSensor(modulated)
    phase = np.zeros(modulated.input.shape)
    first = sensor.photon_rate(phase)
    second = sensor.photon_rate(phase)
    assert np.array_equal(first, second)
    assert np.isclose(first.sum(), modulated.source.detector_photon_rate_per_s, rtol=1e-6)


def test_pyramid_detector_seed_repeats() -> None:
    sensor = WavefrontSensor.from_toml(CONFIG)
    phase = np.zeros(sensor.config.input.shape)
    first = np.asarray(sensor.expose(phase, seed=17))
    second = np.asarray(sensor.expose(phase, seed=17))
    assert np.array_equal(first, second)


def test_pyramid_frame_records_face_order() -> None:
    sensor = WavefrontSensor.from_toml(CONFIG)
    frame = sensor.expose(np.zeros(sensor.config.input.shape), seed=2)
    assert frame.metadata["wfs_pyramid_face_order"] == [
        "upper_left",
        "upper_right",
        "lower_left",
        "lower_right",
    ]
    assert frame.metadata["wfs_source_state_count"] == 1
    assert frame.metadata["wfs_source_wavelengths_m"] == [7.0e-7]


def test_pyramid_allows_overlapping_pupil_images() -> None:
    config = WavefrontSensor.from_toml(CONFIG).config
    assert config.pyramid is not None
    overlap = replace(config.pyramid, pupil_separation_pixels=4)
    sensor = WavefrontSensor(replace(config, pyramid=overlap))
    image = sensor.photon_rate(np.zeros(config.input.shape))
    assert image.shape == (68, 68)
    assert np.isclose(image.sum(), config.source.detector_photon_rate_per_s, rtol=0.1)


def test_pyramid_rejects_range_resolved_lgs() -> None:
    config = WavefrontSensor.from_toml(CONFIG).config
    source = replace(
        config.source,
        kind="lgs",
        lgs_ranges_m=(89e3, 91e3),
        lgs_range_weights=(0.5, 0.5),
    )
    with pytest.raises(NotImplementedError, match="Shack-Hartmann"):
        WavefrontSensor(replace(config, source=source))
