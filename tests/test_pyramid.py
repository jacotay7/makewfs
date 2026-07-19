"""Pyramid optical and detector integration tests."""

from dataclasses import replace
from pathlib import Path

import numpy as np

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
