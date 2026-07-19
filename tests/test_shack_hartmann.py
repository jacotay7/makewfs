"""Shack-Hartmann optical and detector integration tests."""

from dataclasses import replace

import numpy as np
import pytest

from makewfs import WavefrontSensor


def _sensor() -> WavefrontSensor:
    from pathlib import Path

    return WavefrontSensor.from_toml(
        Path(__file__).parents[1] / "examples" / "configs" / "shack_hartmann_minimal.toml"
    )


def test_reference_is_nonnegative_and_flux_bounded() -> None:
    sensor = _sensor()
    reference = sensor.reference()
    assert reference.shape == (64, 64)
    assert np.all(reference >= 0)
    assert reference.sum() <= 2.0e6 * (1.0 + 1e-12)
    assert reference.sum() > 0.0


def test_lenslet_illumination_and_validity_maps_are_explicit() -> None:
    sensor = _sensor()
    illumination = sensor.engine.lenslet_illumination
    valid = sensor.engine.lenslet_valid
    assert illumination.shape == (8, 8)
    assert valid.shape == illumination.shape
    assert np.all((illumination >= 0) & (illumination <= 1))
    assert np.array_equal(
        valid, illumination >= sensor.config.shack_hartmann.minimum_illuminated_fraction
    )


def test_piston_invariance() -> None:
    sensor = _sensor()
    phase = np.zeros((128, 128))
    assert np.allclose(
        sensor.photon_rate(phase), sensor.photon_rate(phase + 2.3e-6), rtol=1e-11, atol=1e-12
    )


def test_nonfinite_opd_inside_pupil_is_rejected() -> None:
    sensor = _sensor()
    phase = np.zeros((128, 128))
    phase[64, 64] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        sensor.photon_rate(phase)


def test_seeded_detector_frame_repeats() -> None:
    sensor = _sensor()
    phase = np.zeros((128, 128))
    first = np.asarray(sensor.expose(phase, seed=4))
    second = np.asarray(sensor.expose(phase, seed=4))
    assert np.array_equal(first, second)


def test_detector_binning_changes_frame_shape_but_preserves_optical_truth() -> None:
    sensor = _sensor()
    configured = WavefrontSensor(
        replace(sensor.config, detector=replace(sensor.config.detector, binning=2))
    )
    frame = configured.expose(np.zeros(configured.config.input.shape), seed=4)
    assert np.asarray(frame).shape == (32, 32)
    assert frame.truth is not None
    assert frame.truth.photon_rate.shape == (64, 64)
    assert frame.metadata["detector_binning"] == 2


def test_temporal_integration_averages_ideal_maps() -> None:
    sensor = _sensor()
    zero = np.zeros((128, 128))
    tilted = zero.copy()
    tilted[:, 64:] = 1e-7
    expected = (sensor.photon_rate(zero) + sensor.photon_rate(tilted)) / 2.0
    result = sensor.expose_integrated(np.stack([zero, tilted]), seed=5)
    assert result.truth is not None
    assert result.metadata["wfs_temporal_samples"] == 2
    assert np.allclose(result.truth.photon_rate, expected, rtol=1e-5, atol=1e-8)


def test_expose_many_is_streaming_and_seeded() -> None:
    sensor = _sensor()
    phases = [np.zeros(sensor.config.input.shape) for _ in range(2)]
    frames = list(sensor.expose_many(phases, seeds=[11, 12]))
    assert len(frames) == 2
    assert not np.array_equal(np.asarray(frames[0]), np.asarray(frames[1]))
    repeated = list(sensor.expose_many(phases, seeds=[11, 12]))
    assert np.array_equal(np.asarray(frames[0]), np.asarray(repeated[0]))
