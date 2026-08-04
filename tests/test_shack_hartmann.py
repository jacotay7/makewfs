"""Shack-Hartmann optical and detector integration tests."""

from dataclasses import replace
from pathlib import Path

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
    rate = sensor.config.source.detector_photon_rate_per_s
    assert rate is not None
    assert reference.sum() <= rate * (1.0 + 1e-12)
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


def test_caller_owned_detector_output_is_exact_and_seeded() -> None:
    sensor = _sensor()
    phase = np.zeros(sensor.config.input.shape)
    reference = sensor.expose(phase, seed=4)
    out = np.empty(sensor.engine.output_shape, dtype=np.uint32)

    frame = sensor.expose(phase, seed=4, out=out)

    assert frame.data is out
    np.testing.assert_array_equal(frame.data, reference.data)


def test_frame_records_optical_and_detector_timings() -> None:
    sensor = _sensor()
    frame = sensor.expose(np.zeros(sensor.config.input.shape), seed=3)
    assert frame.metadata["wfs_optical_render_s"] >= 0.0
    assert frame.metadata["wfs_detector_expose_s"] >= 0.0
    assert frame.metadata["wfs_total_expose_s"] >= frame.metadata["wfs_optical_render_s"]


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


def test_wavelength_resolved_qe_applies_per_pixel_spectral_weights(tmp_path: Path) -> None:
    config = WavefrontSensor.from_toml(
        Path(__file__).parents[1]
        / "benchmarks"
        / "configs"
        / "shack_hartmann_quadrature_9sample.toml"
    ).config
    qe_path = tmp_path / "qe.txt"
    qe_path.write_text("580 0.1\n590 0.5\n600 0.9\n", encoding="utf-8")
    sensor = WavefrontSensor(
        replace(config, detector=replace(config.detector, qe_curve_path=str(qe_path)))
    )
    frame = sensor.expose(np.zeros(config.input.shape), seed=17)
    assert frame.truth is not None
    spectral_photon_rate = frame.truth.spectral_photon_rate
    wavelengths_nm = frame.truth.wavelengths_nm
    assert spectral_photon_rate is not None
    assert wavelengths_nm is not None
    assert frame.metadata["spectral"] is True
    np.testing.assert_allclose(
        frame.truth.photon_rate,
        np.sum(spectral_photon_rate, axis=0),
    )
    qe = np.interp(
        wavelengths_nm,
        [580.0, 590.0, 600.0],
        [0.1, 0.5, 0.9],
    )
    expected_electrons = (
        np.sum(spectral_photon_rate * qe[:, None, None], axis=0) * config.detector.exposure_s
    )
    np.testing.assert_allclose(frame.truth.mean_photoelectrons, expected_electrons, rtol=1e-6)

    integrated = sensor.expose_integrated(
        np.stack([np.zeros(config.input.shape), np.zeros(config.input.shape)]), seed=18
    )
    assert integrated.truth is not None
    assert integrated.metadata["spectral"] is True
    np.testing.assert_allclose(
        integrated.truth.spectral_photon_rate,
        spectral_photon_rate,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        integrated.truth.mean_photoelectrons,
        expected_electrons,
        rtol=1e-6,
    )


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


def test_batched_render_matches_rendering_each_sample_separately() -> None:
    """The batched exposure path must be the sequential one, not an approximation.

    Rendering every temporal sample in one call is a performance change only:
    it exists because the per-call dispatch cost dominates the arithmetic, not
    because the arithmetic differs. Everything downstream of the spot
    intensities is linear in them, which is what makes averaging them early
    legitimate -- and this asserts that the linearity argument actually holds
    through the mosaic, the flux scaling and the captured-rate accounting,
    rather than only in principle.

    The tolerance is float round-off from a different summation order, not a
    modelling allowance.
    """
    sensor = _sensor()
    engine = sensor.engine
    generator = np.random.default_rng(11)
    samples = [2e-7 * generator.standard_normal(sensor.config.input.shape) for _ in range(4)]

    separate = [engine.render(sample) for sample in samples]
    batched = engine.render_integrated(samples)

    expected_rate = np.mean([np.asarray(r.photon_rate) for r in separate], axis=0)
    np.testing.assert_allclose(np.asarray(batched.photon_rate), expected_rate, rtol=1e-5, atol=0.0)
    expected_spectral = np.mean([np.asarray(r.spectral_photon_rate) for r in separate], axis=0)
    np.testing.assert_allclose(
        np.asarray(batched.spectral_photon_rate), expected_spectral, rtol=1e-5, atol=0.0
    )
    expected_captured = np.mean([float(r.captured_rate_per_s) for r in separate])
    assert float(batched.captured_rate_per_s) == pytest.approx(expected_captured, rel=1e-5)
    # The reported OPD is the exposure mean, matching the mean photon rate.
    np.testing.assert_allclose(
        np.asarray(batched.opd_m), np.mean(samples, axis=0), rtol=1e-6, atol=0.0
    )


def test_batched_render_of_one_sample_is_the_single_render() -> None:
    sensor = _sensor()
    engine = sensor.engine
    sample = np.zeros(sensor.config.input.shape)
    np.testing.assert_allclose(
        np.asarray(engine.render_integrated([sample]).photon_rate),
        np.asarray(engine.render(sample).photon_rate),
    )


def test_batched_render_rejects_an_empty_exposure() -> None:
    with pytest.raises(ValueError, match="at least one sample"):
        _sensor().engine.render_integrated([])
