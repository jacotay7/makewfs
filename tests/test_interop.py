"""Optional pyturb contract smoke test."""

from pathlib import Path

import numpy as np
import pytest

from makewfs import WavefrontSensor


def test_shack_hartmann_exposes_static_valid_subaperture_mask() -> None:
    config = Path(__file__).parents[1] / "examples" / "configs" / "shack_hartmann_minimal.toml"
    sensor = WavefrontSensor.from_toml(config)
    mask = np.asarray(sensor.valid_subapertures(), dtype=bool)
    assert sensor.config.shack_hartmann is not None
    count = sensor.config.shack_hartmann.lenslets_across_pupil
    assert mask.shape == (count, count)
    assert mask.any()
    assert np.array_equal(mask, np.asarray(sensor.valid_subapertures(), dtype=bool))


@pytest.mark.interop
def test_pyturb_opd_frames_feed_wavefront_sensor() -> None:
    pyturb = pytest.importorskip("pyturb")
    config = Path(__file__).parents[1] / "examples" / "configs" / "shack_hartmann_minimal.toml"
    sensor = WavefrontSensor.from_toml(config)
    atmosphere = pyturb.Atmosphere.from_profile(
        "paranal-median",
        seeing=0.8,
        diameter=8.0,
        n=128,
        seed=5,
    )
    _, opd = next(atmosphere.frames(dt=0.001, steps=1))
    frame = sensor.expose(np.asarray(pyturb.to_numpy(opd)), seed=5)
    assert np.asarray(frame).shape == sensor.engine.output_shape
