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
        q=max(2, int(np.ceil(output_pixels / pupil_pixels))),
        num_airy=pupil_pixels / 2,
    )
    reference = np.asarray(optics.forward(wavefront).intensity).reshape(ours.shape)
    ours /= ours.sum()
    reference /= reference.sum()
    assert np.sqrt(np.mean((ours - reference) ** 2)) < 1.0e-4
    assert np.corrcoef(ours.ravel(), reference.ravel())[0, 1] > 0.9


@pytest.mark.validation
def test_shack_hartmann_tilt_direction_and_scale_track_hcipy() -> None:
    """Compare a small SH centroid response to HCIPy's independent optics."""
    hcipy = pytest.importorskip("hcipy")
    from makewfs.config import WFSConfig

    pixels = 32
    wavelength = 1.0 / pixels
    data = {
        "schema_version": 1,
        "input": {"quantity": "opd", "unit": "m", "shape": [pixels, pixels], "grid_extent_m": 1.0},
        "telescope": {"pupil_diameter_m": 1.0},
        "source": {"normalization": "detector_photon_rate", "detector_photon_rate_per_s": 1.0},
        "sensor": {"kind": "shack_hartmann", "wavelength_m": wavelength},
        "shack_hartmann": {
            "lenslets_across_pupil": 2,
            "pixels_per_subaperture": 8,
            "spot_sampling_pixels_per_lambda_over_d": 1.0,
            "minimum_illuminated_fraction": 0.0,
        },
        "detector": {"preset": "generic_cmos", "exposure_s": 0.0},
        "numerics": {
            "dtype": "float64",
            "fft_oversampling": 1,
            "fft_workers": 1,
            "pupil_samples_per_lenslet": 16,
        },
    }
    config = WFSConfig.from_dict(data)
    sensor = makewfs.WavefrontSensor(config)
    grid = hcipy.make_pupil_grid(pixels, diameter=1.0)
    centers = np.asarray([-0.25, 0.25])
    lenslet_grid = hcipy.CartesianGrid(hcipy.SeparatedCoords((centers, centers)))

    def square_lenslet(input_grid: object) -> np.ndarray:
        return (
            (np.abs(np.asarray(input_grid.x)) <= 0.25) & (np.abs(np.asarray(input_grid.y)) <= 0.25)
        ).astype(float)

    mla = hcipy.MicroLensArray(grid, lenslet_grid, 0.5, lenslet_shape=square_lenslet)
    optics = hcipy.ShackHartmannWavefrontSensorOptics(grid, mla)
    aperture = np.asarray(hcipy.make_circular_aperture(1.0)(grid)).reshape(pixels, pixels)
    coordinates = np.asarray(grid.x).reshape(pixels, pixels)

    def hcipy_image(cycles_per_aperture: float) -> np.ndarray:
        field = aperture * np.exp(2j * np.pi * cycles_per_aperture * coordinates)
        wavefront = hcipy.Wavefront(hcipy.Field(field.ravel(), grid), wavelength=wavelength)
        image = np.asarray(optics.forward(wavefront).intensity).reshape(pixels, pixels)
        return image.reshape(16, 2, 16, 2).sum(axis=(1, 3))

    def centroids(image: np.ndarray) -> np.ndarray:
        values: list[tuple[float, float]] = []
        for lenslet_y in range(2):
            for lenslet_x in range(2):
                subimage = image[
                    lenslet_y * 8 : (lenslet_y + 1) * 8, lenslet_x * 8 : (lenslet_x + 1) * 8
                ]
                yy, xx = np.indices(subimage.shape)
                values.append(
                    (
                        float(np.sum(subimage * xx) / np.sum(subimage)),
                        float(np.sum(subimage * yy) / np.sum(subimage)),
                    )
                )
        return np.asarray(values)

    zero = np.zeros(config.input.shape)
    ours_zero = centroids(sensor.photon_rate(zero))
    ours_tilt = centroids(sensor.photon_rate(wavelength * 0.2 * coordinates))
    hcipy_zero = centroids(hcipy_image(0.0))
    hcipy_tilt = centroids(hcipy_image(0.2))
    ours_delta = ours_tilt - ours_zero
    hcipy_delta = hcipy_tilt - hcipy_zero
    assert np.all(ours_delta[:, 0] > 0)
    assert np.all(hcipy_delta[:, 0] > 0)
    assert np.max(np.abs(ours_delta[:, 1])) < 0.02
    assert np.max(np.abs(hcipy_delta[:, 1])) < 0.02
    scale_ratio = np.median(ours_delta[:, 0] / hcipy_delta[:, 0])
    assert 0.4 < scale_ratio < 1.8
