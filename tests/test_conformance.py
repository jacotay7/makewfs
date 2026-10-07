"""makewfs follows the AO stack conventions (aocore CONVENTIONS.md)."""

from __future__ import annotations

import numpy as np
import pytest

from makewfs import WavefrontSensor, load_config

conformance = pytest.importorskip("aocore.conformance")
SH_CONFIG = "examples/configs/shack_hartmann_minimal.toml"
PYRAMID_CONFIG = "examples/configs/pyramid_minimal.toml"


def _sensor(path: str):
    config = load_config(path)
    sensor = WavefrontSensor(config)
    return config, sensor, (lambda opd: np.asarray(sensor.photon_rate(opd)))


def test_shack_hartmann_spots_centre_between_pixels_and_follow_tilt() -> None:
    from aocore import ARCSEC_TO_RAD

    config, sensor, image = _sensor(SH_CONFIG)
    shape = config.input.shape
    conformance.check_image_centring(image, pupil_shape=shape)
    # One detector pixel of tilt keeps the 8-pixel subaperture spots in their windows.
    conformance.check_tilt_direction(
        image,
        pupil_shape=shape,
        pitch=config.input.grid_extent_m / shape[1],
        pixel_scale=sensor.subaperture_plate_scale_arcsec() * ARCSEC_TO_RAD,
        pixels=1.0,
    )


def test_pyramid_images_are_centred() -> None:
    config, _, image = _sensor(PYRAMID_CONFIG)
    conformance.check_image_centring(image, pupil_shape=config.input.shape)
