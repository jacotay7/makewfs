"""Pyramid optical and detector integration tests."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from makewfs import WavefrontSensor

CONFIG = Path(__file__).parents[1] / "examples" / "configs" / "pyramid_minimal.toml"


def test_unmodulated_reference_has_four_equal_pupils() -> None:
    sensor = WavefrontSensor.from_toml(CONFIG)
    settings = sensor.config.pyramid
    assert settings is not None
    rate = sensor.config.source.detector_photon_rate_per_s
    assert rate is not None
    size = settings.pixels_across_pupil + settings.pupil_separation_pixels
    half = size // 2
    reference = sensor.reference()
    assert reference.shape == (size, size)
    assert np.all(reference >= 0)
    total = reference.sum()
    assert 0.75 * rate < total <= rate * (1.0 + 1e-9)
    quadrants = [
        reference[:half, :half],
        reference[:half, half:],
        reference[half:, :half],
        reference[half:, half:],
    ]
    assert np.allclose([quadrant.sum() for quadrant in quadrants], total / 4.0, rtol=1e-5)


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
    rate = modulated.source.detector_photon_rate_per_s
    assert rate is not None
    assert 0.75 * rate < first.sum() <= rate * (1.0 + 1e-9)


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


def _cds_pyramid_config(**detector_changes: object):
    """The pyramid example on a C-RED One, which is how a real PWFS is read."""
    config = WavefrontSensor.from_toml(CONFIG).config
    detector = replace(
        config.detector,
        preset="first_light_imaging_cred_one",
        exposure_s=1.0 / 1750.0,
        temperature_c=-188.55,
        **detector_changes,  # type: ignore[arg-type]
    )
    return replace(config, detector=detector)


def test_pyramid_cds_returns_a_signed_bias_free_difference() -> None:
    """CDS must differ from an integrating read in sign convention and pedestal."""
    flat = np.zeros(WavefrontSensor.from_toml(CONFIG).config.input.shape)
    integrating = WavefrontSensor(_cds_pyramid_config(readout_mode="integrate"))
    cds = WavefrontSensor(_cds_pyramid_config(readout_mode="cds"))

    raw = np.asarray(integrating.expose(flat, seed=3).data)
    difference = np.asarray(cds.expose(flat, seed=3).data)

    assert raw.dtype == np.uint32
    assert difference.dtype == np.int32
    assert difference.shape == raw.shape
    # The measured C-RED One pedestal is ~21,000 ADU; differencing removes it and
    # the fixed structure that rides on it, leaving a signed near-zero frame.
    assert np.median(raw) > 15000.0
    assert abs(np.median(difference)) < 250.0
    assert difference.min() < 0
    assert difference.std() < 0.2 * raw.std()


def test_pyramid_cds_preserves_the_four_pupils_and_repeats_on_seed() -> None:
    config = _cds_pyramid_config(readout_mode="cds")
    sensor = WavefrontSensor(config)
    settings = config.pyramid
    assert settings is not None
    tilt_free = np.zeros(config.input.shape)

    first = np.asarray(sensor.expose(tilt_free, seed=11).data)
    second = np.asarray(sensor.expose(tilt_free, seed=11).data)
    np.testing.assert_array_equal(first, second)

    half = (settings.pixels_across_pupil + settings.pupil_separation_pixels) // 2
    quadrants = [
        first[:half, :half].sum(),
        first[:half, half:].sum(),
        first[half:, :half].sum(),
        first[half:, half:].sum(),
    ]
    # A flat wavefront still splits equally across the four faces after CDS.
    assert np.allclose(quadrants, np.mean(quadrants), rtol=0.05)


def test_pyramid_cds_records_readout_mode_in_frame_metadata() -> None:
    sensor = WavefrontSensor(_cds_pyramid_config(readout_mode="cds"))
    frame = sensor.expose(np.zeros(sensor.config.input.shape), seed=5)
    assert frame.metadata["detector_readout_mode"] == "cds"
    assert frame.metadata["readout_mode"] == "global_reset_cds"


def test_pyramid_cds_refuses_caller_owned_output() -> None:
    sensor = WavefrontSensor(_cds_pyramid_config(readout_mode="cds"))
    shape = sensor.expose(np.zeros(sensor.config.input.shape), seed=1).data.shape
    with pytest.raises(RuntimeError, match="caller-owned"):
        sensor.expose(
            np.zeros(sensor.config.input.shape),
            seed=1,
            out=np.zeros(shape, dtype=np.uint32),
        )
