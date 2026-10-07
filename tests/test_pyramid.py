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


def test_pyramid_temporal_integration_averages_ideal_maps() -> None:
    """Pyramid sensors have no batched engine, so this is the sequential path."""
    sensor = WavefrontSensor.from_toml(CONFIG)
    assert not callable(getattr(sensor.engine, "render_integrated", None))
    zero = np.zeros(sensor.config.input.shape)
    tilted = zero.copy()
    tilted[:, zero.shape[1] // 2 :] = 1e-7
    expected = (sensor.photon_rate(zero) + sensor.photon_rate(tilted)) / 2.0
    result = sensor.expose_integrated(np.stack([zero, tilted]), seed=5)
    assert result.truth is not None
    assert result.metadata["wfs_temporal_samples"] == 2
    assert np.allclose(result.truth.photon_rate, expected, rtol=1e-5, atol=1e-8)
    repeated = sensor.expose_integrated(np.stack([zero, tilted]), seed=5)
    assert np.array_equal(np.asarray(result), np.asarray(repeated))


def test_pyramid_temporal_integration_rejects_an_empty_exposure() -> None:
    sensor = WavefrontSensor.from_toml(CONFIG)
    with pytest.raises(ValueError, match="at least one sample"):
        sensor.expose_integrated([])


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


def test_pyramid_background_adds_signal_and_its_shot_noise() -> None:
    """Sky background is light: it collects charge and carries shot noise."""
    dark = _cds_pyramid_config(readout_mode="cds")
    lit = replace(dark, detector=replace(dark.detector, background_photon_rate_per_s=2.0e6))
    flat = np.zeros(dark.input.shape)
    without = np.asarray(WavefrontSensor(dark).expose(flat, seed=7).data, dtype=float)
    with_sky = np.asarray(WavefrontSensor(lit).expose(flat, seed=7).data, dtype=float)
    # 2e6 photons/s/pixel over 1/3500 s at QE ~0.8 is a few hundred electrons,
    # so the level rises everywhere, including outside the four pupils.
    assert np.median(with_sky) > np.median(without) + 10.0
    # And it is light, not an offset: the extra charge brings extra shot noise.
    assert with_sky.std() > without.std()


@pytest.mark.parametrize(
    "method",
    [
        "valid_subapertures",
        "subaperture_plate_scale_arcsec",
        "subaperture_field_of_view_arcsec",
    ],
)
def test_pyramid_rejects_shack_hartmann_only_geometry(method: str) -> None:
    sensor = WavefrontSensor.from_toml(CONFIG)
    with pytest.raises(ValueError, match="Shack--Hartmann"):
        getattr(sensor, method)()


def test_simulate_accepts_a_config_path_and_matches_a_seeded_exposure() -> None:
    from makewfs import simulate

    sensor = WavefrontSensor.from_toml(CONFIG)
    phase = np.zeros(sensor.config.input.shape)
    assert np.array_equal(
        np.asarray(simulate(phase, CONFIG, seed=3)),
        np.asarray(sensor.expose(phase, seed=3)),
    )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize(
    ("pixels", "separation", "samples", "margin"), [(16, 6, 1, 0), (17, 5, 4, 2)]
)
def test_unshifted_pruned_propagation_matches_centered_reference(
    dtype: str, pixels: int, separation: int, samples: int, margin: int
) -> None:
    """Cancelling the inner shifts and pruning the FFTs keeps the optical result."""
    from makewfs.backend import centered_fft2, centered_ifft2
    from makewfs.sampling import crop_center, pad_center

    config = WavefrontSensor.from_toml(CONFIG).config
    assert config.pyramid is not None
    config = replace(
        config,
        numerics=replace(config.numerics, dtype=dtype),
        pyramid=replace(
            config.pyramid,
            pixels_across_pupil=pixels,
            pupil_separation_pixels=separation,
            modulation_radius_lambda_over_d=2.0 if samples > 1 else 0.0,
            modulation_samples=samples,
            detector_margin_pixels=margin,
        ),
    )
    engine = WavefrontSensor(config).engine
    rng = np.random.default_rng(3)
    opd = rng.normal(0.0, 5.0e-8, config.input.shape)
    internal = engine.wavefront.opd(opd, target_shape=engine.internal_shape)
    fields = engine._fields(internal, engine.source_states[0], 0)
    padded = pad_center(fields, (engine.nfft, engine.nfft))
    exit_pupil = centered_ifft2(centered_fft2(padded) * engine._mask[None, ...])
    mosaic = np.mean(crop_center(np.abs(exit_pupil) ** 2, engine._base_output_shape), axis=0)
    expected = np.zeros(engine.output_shape)
    base = engine._base_output_shape[0]
    expected[margin : margin + base, margin : margin + base] = mosaic
    expected *= engine.source_rate / engine._total_field_flux

    result = engine.render(opd)

    tolerance = 2e-6 if dtype == "float32" else 1e-12
    np.testing.assert_allclose(
        result.photon_rate, expected, rtol=tolerance, atol=tolerance * expected.max()
    )


@pytest.mark.parametrize(
    ("path", "size"),
    [
        (CONFIG, 288),
        (CONFIG.parents[2] / "benchmarks" / "configs" / "pyramid_40_float32.toml", 108),
        (CONFIG.parents[2] / "benchmarks" / "configs" / "pyramid_60_mod8_float32.toml", 160),
        (CONFIG.parents[2] / "benchmarks" / "configs" / "pyramid_80_mod32_float64.toml", 216),
    ],
)
def test_pyramid_fft_size_is_recorded_and_unchanged_for_shipped_configs(
    path: Path, size: int
) -> None:
    sensor = WavefrontSensor.from_toml(path)
    assert sensor.engine.nfft == size
    frame = sensor.expose(np.zeros(sensor.config.input.shape), seed=1)
    assert frame.metadata["wfs_pyramid_fft_size_px"] == size
