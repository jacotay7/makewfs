"""Input-OPD RMS frame metadata (aocore CONVENTIONS 4.1)."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from aocore import rms as aocore_rms
from aocore import rms_unweighted as aocore_rms_unweighted
from aocore.conformance import check_rms

from makewfs import WavefrontSensor, load_config
from makewfs.pupil import make_pupil
from makewfs.sampling import area_rebin
from makewfs.wavefront import grid_rms, pupil_rms, pupil_weights

CONFIGS = Path(__file__).parents[1] / "examples" / "configs"
SENSORS = ["shack_hartmann_minimal.toml", "pyramid_minimal.toml"]


def _sensor(name: str) -> WavefrontSensor:
    return WavefrontSensor.from_toml(CONFIGS / name)


def _x_m(sensor: WavefrontSensor) -> np.ndarray:
    """Centred physical x coordinate of every input pixel, in metres."""
    height, width = sensor.config.input.shape
    extent = sensor.config.input.grid_extent_m
    x = (np.arange(width) - (width - 1) / 2) * extent / width
    return np.broadcast_to(x, (height, width)).copy()


def _pupil(sensor: WavefrontSensor) -> np.ndarray:
    """The configured pupil amplitude on the input grid, in float64."""
    config = sensor.config
    return make_pupil(
        config.telescope,
        config.input.shape,
        config.input.grid_extent_m,
        supersampling=config.numerics.pupil_supersampling,
    )


def test_weighted_rms_helper_passes_the_aocore_rms_check() -> None:
    def makewfs_rms(opd: np.ndarray, amplitude: np.ndarray) -> float:
        return float(pupil_rms(opd, pupil_weights(amplitude**2)))

    check_rms(makewfs_rms, rel_tol=1e-12)


@pytest.mark.parametrize("name", SENSORS)
def test_piston_only_input_has_zero_rms(name: str) -> None:
    sensor = _sensor(name)
    piston = np.full(sensor.config.input.shape, 2.5e-6)
    metadata = sensor.expose(piston, seed=1).metadata
    assert metadata["wfs_input_opd_rms_m"] == pytest.approx(0.0, abs=1e-20)
    assert metadata["wfs_input_opd_rms_unweighted_m"] == pytest.approx(2.5e-6, rel=1e-12)


@pytest.mark.parametrize("name", SENSORS)
def test_tilt_matches_the_analytic_pupil_weighted_rms(name: str) -> None:
    # On an annulus of outer radius R and obscuration ratio e, <x> = 0 and
    # <x^2> = R^2 (1 + e^2) / 4, so a slope s gives s R sqrt(1 + e^2) / 2. A
    # piston is added to show that it is removed.
    sensor = _sensor(name)
    telescope = sensor.config.telescope
    assert not telescope.spiders and telescope.custom_mask_path is None
    slope = 1.0e-7
    opd = slope * _x_m(sensor) + 3.0e-6
    radius = telescope.pupil_diameter_m / 2
    analytic = slope * radius * math.sqrt(1 + telescope.central_obscuration_ratio**2) / 2

    metadata = sensor.expose(opd, seed=1).metadata
    # The sampled 128-pixel pupil reproduces the continuous annulus to 0.1 %.
    assert metadata["wfs_input_opd_rms_m"] == pytest.approx(analytic, rel=2e-3)
    # It is exactly CONVENTIONS 4.1 on the pupil the sensor uses.
    assert metadata["wfs_input_opd_rms_m"] == pytest.approx(
        aocore_rms(opd, _pupil(sensor)), rel=1e-12
    )
    assert metadata["wfs_input_opd_rms_unweighted_m"] == pytest.approx(
        aocore_rms_unweighted(opd), rel=1e-12
    )


@pytest.mark.parametrize("name", SENSORS)
def test_only_the_unweighted_rms_sees_opd_outside_the_pupil(name: str) -> None:
    sensor = _sensor(name)
    inside = 4.0e-8 * _x_m(sensor)
    outside = _pupil(sensor) == 0.0
    assert outside.any()
    rng = np.random.default_rng(3)
    polluted = inside.copy()
    polluted[outside] = rng.normal(0.0, 1.0e-5, int(outside.sum()))

    clean = sensor.expose(inside, seed=1).metadata
    dirty = sensor.expose(polluted, seed=1).metadata
    assert dirty["wfs_input_opd_rms_m"] == clean["wfs_input_opd_rms_m"]
    assert dirty["wfs_input_opd_rms_unweighted_m"] > 10 * clean["wfs_input_opd_rms_unweighted_m"]


@pytest.mark.parametrize("name", SENSORS)
def test_integrated_exposure_reports_the_rms_of_the_mean_opd(name: str) -> None:
    sensor = _sensor(name)
    tilt = 1.0e-7 * _x_m(sensor)
    samples = np.stack([tilt + 1.0e-6, 3.0 * tilt - 1.0e-6])
    metadata = sensor.expose_integrated(samples, seed=1).metadata
    mean_opd = samples.mean(axis=0)
    assert metadata["wfs_input_opd_rms_m"] == pytest.approx(
        aocore_rms(mean_opd, _pupil(sensor)), rel=1e-12
    )
    assert metadata["wfs_input_opd_rms_unweighted_m"] == pytest.approx(
        aocore_rms_unweighted(mean_opd), rel=1e-12
    )


@pytest.mark.parametrize("name", SENSORS)
def test_custom_mask_weights_are_area_averaged_onto_the_input_grid(
    name: str, tmp_path: Path
) -> None:
    # A custom mask exists only on the engine's pupil grid, so each input pixel
    # takes the mean intensity of the mask cells it covers. The grid is 128 x 128
    # for the Shack-Hartmann, the same as the input, and 64 x 64 for the
    # pyramid, half of it.
    config = load_config(CONFIGS / name)
    probe = WavefrontSensor(config)
    engine_shape = tuple(probe.engine.configured_pupil.shape)
    mask = np.zeros(engine_shape)
    mask[8:40, 16:56] = 1.0
    mask[40:44, 16:56] = 0.5
    path = tmp_path / "mask.npy"
    np.save(path, mask)
    sensor = WavefrontSensor(
        replace(config, telescope=replace(config.telescope, custom_mask_path=str(path)))
    )
    factor = config.input.shape[0] // engine_shape[0]
    assert factor * engine_shape[0] == config.input.shape[0]
    amplitude_on_input = np.sqrt(np.kron(mask**2, np.ones((factor, factor))))

    rng = np.random.default_rng(5)
    opd = rng.normal(0.0, 5.0e-8, config.input.shape) + 1.0e-6
    metadata = sensor.expose(opd, seed=1).metadata
    assert metadata["wfs_input_opd_rms_m"] == pytest.approx(
        aocore_rms(opd, amplitude_on_input), rel=1e-12
    )


@pytest.mark.parametrize("name", SENSORS)
def test_phase_input_rms_is_reported_in_opd_metres(name: str) -> None:
    # Phase input is converted at the reference wavelength before anything
    # downstream sees it, including both input-RMS keys and OpticalResult.
    opd_sensor = _sensor(name)
    wavelength = 1.0e-6
    phase_input = replace(
        opd_sensor.config.input,
        quantity="phase",
        unit="rad",
        reference_wavelength_m=wavelength,
    )
    phase_sensor = WavefrontSensor(replace(opd_sensor.config, input=phase_input))
    opd = 3.0e-8 * _x_m(opd_sensor) / opd_sensor.config.input.grid_extent_m + 1.0e-8
    phase = 2.0 * np.pi * opd / wavelength

    weighted = aocore_rms(opd, _pupil(opd_sensor))
    unweighted = aocore_rms_unweighted(opd)
    for sensor, wavefront in ((opd_sensor, opd), (phase_sensor, phase)):
        metadata = sensor.expose(wavefront, seed=1).metadata
        assert metadata["wfs_input_opd_rms_m"] == pytest.approx(weighted, rel=1e-12)
        assert metadata["wfs_input_opd_rms_unweighted_m"] == pytest.approx(unweighted, rel=1e-12)
    np.testing.assert_allclose(phase_sensor._render(phase).opd_m, opd, rtol=1e-12)
    np.testing.assert_allclose(
        phase_sensor.photon_rate(phase), opd_sensor.photon_rate(opd), rtol=1e-6, atol=0.0
    )


def test_empty_pupil_on_the_input_grid_is_rejected() -> None:
    with pytest.raises(ValueError, match="no illuminated pixels"):
        pupil_weights(np.zeros((4, 4)))


def test_grid_rms_keeps_piston_and_counts_every_pixel() -> None:
    opd = np.array([[1.0, 1.0], [1.0, 5.0]])
    assert float(grid_rms(opd)) == pytest.approx(math.sqrt(28.0 / 4.0))
    assert float(grid_rms(opd)) == pytest.approx(aocore_rms_unweighted(opd), rel=1e-15)


@pytest.mark.parametrize(
    ("source", "target"),
    [((64, 64), (128, 128)), ((128, 128), (64, 64)), ((40, 48), (64, 30)), ((7, 7), (7, 7))],
)
def test_area_rebin_preserves_uniform_maps_and_the_area_integral(
    source: tuple[int, int], target: tuple[int, int]
) -> None:
    rng = np.random.default_rng(7)
    array = rng.uniform(0.0, 1.0, source)
    rebinned = area_rebin(array, target)
    assert rebinned.shape == target
    source_area = 1.0 / (source[0] * source[1])
    target_area = 1.0 / (target[0] * target[1])
    assert float(rebinned.sum()) * target_area == pytest.approx(
        float(array.sum()) * source_area, rel=1e-12
    )
    np.testing.assert_allclose(area_rebin(np.ones(source), target), 1.0, rtol=1e-12)


def test_area_rebin_downsampling_is_a_block_mean() -> None:
    array = np.arange(36.0).reshape(6, 6)
    expected = array.reshape(3, 2, 3, 2).mean(axis=(1, 3))
    np.testing.assert_allclose(area_rebin(array, (3, 3)), expected, rtol=1e-12)
    # A non-integer ratio splits a source cell between two target cells.
    np.testing.assert_allclose(area_rebin(np.array([[0.0, 3.0, 6.0]]), (1, 2)), [[1.0, 5.0]])
