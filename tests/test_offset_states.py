"""Configured Shack-Hartmann lenslet-grid offset states (issue #19)."""

from __future__ import annotations

import itertools
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from makewfs import ConfigError, WavefrontSensor, load_config
from makewfs.config import WFSConfig

ROOT = Path(__file__).parents[1]
SH_CONFIG = ROOT / "examples" / "configs" / "shack_hartmann_minimal.toml"
PYRAMID_CONFIG = ROOT / "examples" / "configs" / "pyramid_minimal.toml"
STATES = ((0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5))


def _with_states(
    states: tuple[tuple[float, float], ...],
    *,
    base_offset: tuple[float, float] = (0.0, 0.0),
    dtype: str = "float64",
    supersampling: int = 1,
) -> WFSConfig:
    config = load_config(SH_CONFIG)
    assert config.shack_hartmann is not None
    return replace(
        config,
        numerics=replace(config.numerics, dtype=dtype, pupil_supersampling=supersampling),
        shack_hartmann=replace(
            config.shack_hartmann,
            lenslet_grid_offset_fraction=base_offset,
            lenslet_grid_offset_states_fraction=states,
        ),
    )


def _static(offset: tuple[float, float], *, dtype: str = "float64") -> WFSConfig:
    config = load_config(SH_CONFIG)
    assert config.shack_hartmann is not None
    return replace(
        config,
        numerics=replace(config.numerics, dtype=dtype),
        shack_hartmann=replace(config.shack_hartmann, lenslet_grid_offset_fraction=offset),
    )


def _opd(shape: tuple[int, int]) -> np.ndarray:
    rng = np.random.default_rng(19)
    return rng.normal(size=shape) * 40e-9


def test_offset_states_are_validated_and_serialized() -> None:
    data = load_config(SH_CONFIG).to_dict()
    assert "lenslet_grid_offset_states_fraction" not in data["shack_hartmann"]
    data["shack_hartmann"]["lenslet_grid_offset_states_fraction"] = [[0.0, 0.0], [0.5, -0.25]]
    config = WFSConfig.from_dict(data)
    assert config.shack_hartmann is not None
    assert config.shack_hartmann.lenslet_grid_offset_states_fraction == (
        (0.0, 0.0),
        (0.5, -0.25),
    )
    assert WFSConfig.from_dict(config.to_dict()).digest == config.digest
    # The states are physical intent, so they move the digest.
    assert config.digest != load_config(SH_CONFIG).digest
    for value, match in (
        (0.5, "expected a list"),
        ([[0.5]], r"states_fraction\[0\]: expected \[x, y\]"),
        ([[0.0, 0.0], [0.1, 0.2, 0.3]], r"states_fraction\[1\]: expected \[x, y\]"),
        ([[0.0, float("nan")]], "finite"),
        ([0.5, 0.5], r"states_fraction\[0\]: expected an array"),
    ):
        data["shack_hartmann"]["lenslet_grid_offset_states_fraction"] = value
        with pytest.raises(ConfigError, match=match):
            WFSConfig.from_dict(data)


def test_effective_offsets_add_the_static_registration() -> None:
    config = _with_states(((0.0, 0.0), (0.5, 0.0)), base_offset=(0.1, -0.2))
    assert config.shack_hartmann is not None
    assert config.shack_hartmann.lenslet_grid_offsets_fraction == ((0.1, -0.2), (0.6, -0.2))
    sensor = WavefrontSensor(config)
    assert sensor.lenslet_grid_offsets_fraction() == ((0.1, -0.2), (0.6, -0.2))
    single = WavefrontSensor(_static((0.1, -0.2)))
    assert single.lenslet_grid_offsets_fraction() == ((0.1, -0.2),)


def test_each_state_equals_a_sensor_built_with_that_static_offset() -> None:
    sensor = WavefrontSensor(_with_states(STATES))
    opd = _opd(sensor.config.input.shape)
    for index, offset in enumerate(STATES):
        reference = WavefrontSensor(_static(offset))
        np.testing.assert_array_equal(
            sensor.photon_rate(opd, offset_state=index), reference.photon_rate(opd)
        )
        np.testing.assert_array_equal(
            sensor.valid_subapertures(offset_state=index), reference.valid_subapertures()
        )
        np.testing.assert_array_equal(sensor.reference(offset_state=index), reference.reference())
    # Half-pitch states genuinely differ from the aligned grid.
    assert not np.allclose(sensor.photon_rate(opd), sensor.photon_rate(opd, offset_state=3))


def test_state_selection_is_validated() -> None:
    sensor = WavefrontSensor(_with_states(STATES))
    opd = np.zeros(sensor.config.input.shape)
    with pytest.raises(ValueError, match=r"out of range.*4 lenslet-grid offset"):
        sensor.photon_rate(opd, offset_state=4)
    with pytest.raises(ValueError, match="out of range"):
        sensor.expose(opd, offset_state=-1)
    with pytest.raises(TypeError, match="integer"):
        sensor.photon_rate(opd, offset_state=True)
    with pytest.raises(TypeError, match="integer"):
        sensor.valid_subapertures(offset_state=1.0)  # type: ignore[arg-type]
    single = WavefrontSensor(load_config(SH_CONFIG))
    with pytest.raises(ValueError, match="1 lenslet-grid offset"):
        single.photon_rate(opd, offset_state=1)


def test_pyramid_accepts_only_state_zero() -> None:
    pyramid = WavefrontSensor(load_config(PYRAMID_CONFIG))
    opd = np.zeros(pyramid.config.input.shape)
    np.testing.assert_array_equal(
        pyramid.photon_rate(opd, offset_state=0), pyramid.photon_rate(opd)
    )
    with pytest.raises(ValueError, match="only for Shack--Hartmann"):
        pyramid.photon_rate(opd, offset_state=1)
    with pytest.raises(ValueError, match="only for Shack--Hartmann"):
        pyramid.expose_integrated([opd], offset_state=1)
    with pytest.raises(ValueError, match="only for Shack--Hartmann"):
        pyramid.lenslet_grid_offsets_fraction()
    assert "wfs_lenslet_offset_state" not in pyramid.expose(opd, seed=1).metadata


def test_frames_record_the_offset_state() -> None:
    sensor = WavefrontSensor(_with_states(STATES))
    opd = _opd(sensor.config.input.shape)
    frame = sensor.expose(opd, seed=3, offset_state=2)
    assert frame.metadata["wfs_lenslet_offset_state"] == 2
    assert frame.metadata["wfs_lenslet_grid_offset_fraction"] == [0.0, 0.5]
    assert frame.metadata["wfs_lenslet_offset_state_count"] == 4
    np.testing.assert_allclose(
        frame.truth.photon_rate, sensor.photon_rate(opd, offset_state=2), rtol=1e-12
    )
    default = sensor.expose(opd, seed=3)
    assert default.metadata["wfs_lenslet_offset_state"] == 0


def test_expose_many_cycles_states_and_integration_uses_one_state() -> None:
    sensor = WavefrontSensor(_with_states(STATES))
    opd = _opd(sensor.config.input.shape)
    frames = list(
        sensor.expose_many(
            [opd] * 6, seeds=range(6), offset_states=itertools.cycle(range(len(STATES)))
        )
    )
    assert [frame.metadata["wfs_lenslet_offset_state"] for frame in frames] == [0, 1, 2, 3, 0, 1]
    integrated = sensor.expose_integrated([opd, 0.5 * opd], seed=0, offset_state=3)
    assert integrated.metadata["wfs_lenslet_offset_state"] == 3
    expected = 0.5 * (
        sensor.photon_rate(opd, offset_state=3) + sensor.photon_rate(0.5 * opd, offset_state=3)
    )
    np.testing.assert_allclose(integrated.truth.photon_rate, expected, rtol=1e-10, atol=1e-6)


def _exact_illumination(
    offset: tuple[float, float], *, lenslets: int, diameter_m: float, obscuration: float
) -> np.ndarray:
    """Area fraction of each translated lenslet inside the analytic annulus."""
    pitch = diameter_m / lenslets
    centres = (np.arange(lenslets) - (lenslets - 1) / 2) * pitch
    local = (np.arange(256) + 0.5) / 256 * pitch - pitch / 2
    result = np.zeros((lenslets, lenslets))
    for row, centre_y in enumerate(centres):
        for column, centre_x in enumerate(centres):
            x = centre_x + offset[0] * pitch + local[None, :]
            y = centre_y + offset[1] * pitch + local[:, None]
            radius = np.hypot(x, y)
            inside = (radius <= diameter_m / 2) & (radius >= obscuration * diameter_m / 2)
            result[row, column] = np.mean(inside)
    return result


def test_translated_pupil_illumination_matches_the_analytic_area() -> None:
    offsets = ((0.0, 0.0), (0.13, 0.0), (0.5, 0.25), (-0.31, 0.47))
    sensor = WavefrontSensor(_with_states(offsets, supersampling=8))
    config = sensor.config
    assert config.shack_hartmann is not None
    for index, offset in enumerate(offsets):
        expected = _exact_illumination(
            offset,
            lenslets=config.shack_hartmann.lenslets_across_pupil,
            diameter_m=config.telescope.pupil_diameter_m,
            obscuration=config.telescope.central_obscuration_ratio,
        )
        illumination = np.asarray(sensor.engine.geometry(index).lenslet_illumination)
        # The residual is the 16-sample pupil grid's own area error, not the
        # translation: the aligned state shows the same order.
        np.testing.assert_allclose(illumination, expected, atol=6e-3)


def test_sub_sample_translation_moves_edge_illumination_continuously() -> None:
    """Linear translation of an edge lenslet gives linear illumination change.

    The right-hand lenslet of a pupil that fills the grid loses illumination
    linearly as it slides off the pupil edge. Before 2.2 the edge column of
    samples dropped to zero at once, a staircase of 3.5% steps.
    """
    steps = tuple((0.01 * index, 0.0) for index in range(11))
    sensor = WavefrontSensor(_with_states(steps, supersampling=4))
    edge = np.array(
        [
            float(np.asarray(sensor.engine.geometry(index).lenslet_illumination)[3, 7])
            for index in range(len(steps))
        ]
    )
    differences = np.diff(edge)
    assert np.all(differences < 0)
    np.testing.assert_allclose(differences, differences.mean(), atol=2e-4)
    exact = [
        _exact_illumination(offset, lenslets=8, diameter_m=8.0, obscuration=0.14)[3, 7]
        for offset in steps
    ]
    np.testing.assert_allclose(differences, np.diff(exact), atol=5e-4)


def test_float32_states_match_static_offsets_in_integrated_renders() -> None:
    sensor = WavefrontSensor(_with_states(STATES, dtype="float32"))
    reference = WavefrontSensor(_static((0.5, 0.5), dtype="float32"))
    opd = _opd(sensor.config.input.shape)
    np.testing.assert_array_equal(
        sensor.engine.render_integrated([opd, -opd], offset_state=3).photon_rate,
        reference.engine.render_integrated([opd, -opd]).photon_rate,
    )
