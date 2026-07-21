from __future__ import annotations

import importlib.util
import sys
from dataclasses import fields
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

import makewfs

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "keck_haka"


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("keck_haka_simulate", EXAMPLE / "simulate.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_camera_lookup_boundaries_and_limits() -> None:
    example = _module()
    modes = example.load_camera_modes()
    assert len(modes) == 20
    assert example.select_camera_mode(5.0, modes).watao == 15
    assert example.select_camera_mode(10.0, modes).watao == 10
    assert example.select_camera_mode(15.0, modes).watao == 1
    assert example.select_camera_mode(14.999, modes).watao == 2
    assert example.select_camera_mode(11.0, modes).frame_rate_hz("WSFRRT2") == 500
    assert max(mode.em_gain for mode in modes) == 600
    assert max(mode.frame_rate_1_hz for mode in modes) <= 2067
    with pytest.raises(ValueError, match="matches 0"):
        example.select_camera_mode(25.0, modes)
    with pytest.raises(ValueError, match="frame-rate column"):
        modes[0].frame_rate_hz("invalid")


def test_keck_pupil_and_haka_geometry(tmp_path: Path) -> None:
    example = _module()
    pupil = example.make_keck_pupil(supersampling=2)
    config = makewfs.load_config(EXAMPLE / "keck_haka.toml")
    assert pupil.shape == (228, 228)
    assert np.all((pupil >= 0) & (pupil <= 1))
    assert len(example._keck_segment_centres(1.0)) == 36
    assert pupil[114, 114] == 0.0  # the Keck primary has no central segment
    assert pytest.approx(1.2835, abs=1e-4) == example.KECK_SECONDARY_CIRCLE_RADIUS_M
    assert np.sqrt(3.0) * example.KECK_SECONDARY_HEX_CIRCUMRADIUS_M == pytest.approx(
        2.5287, abs=1e-4
    )
    # Each component adds a distinct part of the union: the circle extends
    # beyond a hex flat and the hex extends beyond the circle at a vertex.
    circle_only_x = 1.275
    assert circle_only_x < example.KECK_SECONDARY_CIRCLE_RADIUS_M
    assert not example._inside_hexagon(
        np.asarray([circle_only_x]),
        np.asarray([0.0]),
        example.KECK_SECONDARY_HEX_CIRCUMRADIUS_M,
    )[0]
    hex_only_y = 1.40
    assert hex_only_y > example.KECK_SECONDARY_CIRCLE_RADIUS_M
    assert example._inside_hexagon(
        np.asarray([0.0]),
        np.asarray([hex_only_y]),
        example.KECK_SECONDARY_HEX_CIRCUMRADIUS_M,
    )[0]
    assert pupil[:, 0].max() == 0.0 and pupil[:, -1].max() == 0.0
    hex_membership = example._inside_hexagon(np.asarray([0.0, 0.95]), np.asarray([0.95, 0.0]), 1.0)
    assert hex_membership.tolist() == [True, False]

    without_spiders = example.make_keck_pupil(spider_width_m=0.0, supersampling=2)
    spider_loss = without_spiders - pupil
    center = pupil.shape[0] // 2
    vertical_loss = float(np.sum(spider_loss[:, center - 1 : center + 1]))
    horizontal_loss = float(np.sum(spider_loss[center - 1 : center + 1, :]))
    assert vertical_loss > 2.0 * horizontal_loss
    assert config.shack_hartmann is not None
    assert (
        config.shack_hartmann.lenslets_across_pupil * config.shack_hartmann.pixels_per_subaperture
        == 228
    )
    assert config.input.shape == (228, 228)
    assert config.input.grid_extent_m == pytest.approx(10.95 * 57 / 54)
    assert config.source.throughput == 1.0
    assert example._atmosphere_display_gap_s(1.0 / 2000.0, 1.0 / 30.0) == pytest.approx(
        1.0 / 30.0 - 1.0 / 2000.0
    )
    assert example._atmosphere_display_gap_s(1.0 / 30.0, 1.0 / 2000.0) == 0.0

    pupil_path = tmp_path / "keck-primary.npy"
    np.save(pupil_path, pupil)
    import getframes

    required_camera_fields = {
        "output_full_well_e",
        "amplifier_boundaries_y_px",
        "amplifier_boundaries_x_px",
        "amplifier_gain_factors",
        "amplifier_offsets_adu",
    }
    available_camera_fields = {field.name for field in fields(getframes.CameraConfig)}
    if not required_camera_fields <= available_camera_fields:
        pytest.skip("Keck HAKA detector integration requires the pending getframes release")
    mode = example.select_camera_mode(5.0, example.load_camera_modes())
    sensor = example.configured_sensor(
        config,
        magnitude=5.0,
        mode=mode,
        frame_rate_column="WSFRRT1",
        pupil_path=pupil_path,
    )
    assert sensor.engine.output_shape == (228, 228)
    assert sensor.config.detector.exposure_s == 1.0 / 2000.0
    assert sensor.detector.camera.config.em_gain == 8.0
    assert sensor.detector.camera.config.amplifier_layout == (4, 2)
    assert sensor.detector.camera.config.amplifier_boundaries_y_px == (54, 114, 174)
    assert sensor.detector.camera.config.amplifier_boundaries_x_px == (114,)
    assert len(sensor.detector.camera.config.amplifier_gain_factors) == 8
