"""Optional independent Fourier-optics cross-checks using HCIPy."""

from pathlib import Path

import numpy as np
import pytest

import makewfs


@pytest.mark.validation
def test_fixed_pyramid_reference_tracks_hcipy() -> None:
    hcipy = pytest.importorskip("hcipy")
    config = makewfs.load_config(
        Path(__file__).parents[1] / "examples" / "configs" / "pyramid_minimal.toml"
    )
    sensor = makewfs.WavefrontSensor(config)
    ours = sensor.photon_rate(np.zeros(config.input.shape, dtype=np.float64))

    pupil_pixels = config.pyramid.pixels_across_pupil  # type: ignore[union-attr]
    output_pixels = ours.shape[0]
    diameter = config.telescope.pupil_diameter_m
    input_grid = hcipy.make_pupil_grid(pupil_pixels, diameter=diameter)
    output_grid = hcipy.make_pupil_grid(
        output_pixels, diameter=output_pixels * diameter / pupil_pixels
    )
    aperture = hcipy.make_obstructed_circular_aperture(
        diameter, config.telescope.central_obscuration_ratio
    )(input_grid)
    wavefront = hcipy.Wavefront(
        hcipy.Field(aperture, input_grid), wavelength=config.sensor.wavelength_m
    )
    optics = hcipy.PyramidWavefrontSensorOptics(
        input_grid,
        output_grid,
        separation=config.pyramid.pupil_separation_pixels * diameter / pupil_pixels,  # type: ignore[union-attr]
        pupil_diameter=diameter,
        wavelength_0=config.sensor.wavelength_m,
        q=2,
        num_airy=pupil_pixels / 2,
    )
    reference = np.asarray(optics.forward(wavefront).intensity).reshape(ours.shape)
    ours /= ours.sum()
    reference /= reference.sum()
    assert np.sqrt(np.mean((ours - reference) ** 2)) < 1.0e-4
    assert np.corrcoef(ours.ravel(), reference.ravel())[0, 1] > 0.9
