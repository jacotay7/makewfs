"""Optional independent Fourier-optics cross-checks using HCIPy."""

from pathlib import Path

import numpy as np
import pytest

import makewfs


def _normalized(image: np.ndarray) -> np.ndarray:
    return image / np.sum(image)


def _hcipy_pyramid_image(
    hcipy: object,
    optics: object,
    aperture: np.ndarray,
    input_grid: object,
    mode: np.ndarray,
    wavelength: float,
    amplitude: float,
    output_shape: tuple[int, int],
    sign: float,
) -> np.ndarray:
    field = aperture * np.exp(2j * np.pi * sign * amplitude * mode / wavelength)
    wavefront = hcipy.Wavefront(hcipy.Field(field, input_grid), wavelength=wavelength)
    return _normalized(np.asarray(optics.forward(wavefront).intensity).reshape(output_shape))


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
    ours = _normalized(ours)
    reference = _normalized(reference)
    assert np.sqrt(np.mean((ours - reference) ** 2)) < 1.0e-4
    assert np.corrcoef(ours.ravel(), reference.ravel())[0, 1] > 0.9


@pytest.mark.validation
def test_pyramid_low_order_response_maps_track_hcipy() -> None:
    """Compare tip, tilt, and focus push/pull maps, not only the flat image."""
    hcipy = pytest.importorskip("hcipy")
    config = makewfs.load_config(
        Path(__file__).parents[1] / "examples" / "configs" / "pyramid_minimal.toml"
    )
    sensor = makewfs.WavefrontSensor(config)
    assert config.pyramid is not None
    pupil_pixels = config.pyramid.pixels_across_pupil
    output_pixels = sensor.engine.output_shape[0]
    diameter = config.telescope.pupil_diameter_m
    wavelength = config.sensor.wavelength_m
    input_grid = hcipy.make_pupil_grid(pupil_pixels, diameter=diameter)
    output_grid = hcipy.make_pupil_grid(
        output_pixels, diameter=output_pixels * diameter / pupil_pixels
    )
    aperture = np.asarray(
        hcipy.make_obstructed_circular_aperture(
            diameter, config.telescope.central_obscuration_ratio
        )(input_grid)
    )
    optics = hcipy.PyramidWavefrontSensorOptics(
        input_grid,
        output_grid,
        separation=config.pyramid.pupil_separation_pixels * diameter / pupil_pixels,
        pupil_diameter=diameter,
        wavelength_0=wavelength,
        q=max(2, int(np.ceil(output_pixels / pupil_pixels))),
        num_airy=pupil_pixels / 2,
    )

    input_y, input_x = np.indices(config.input.shape, dtype=np.float64)
    input_x = (input_x + 0.5 - config.input.shape[1] / 2) * diameter / config.input.shape[1]
    input_y = (input_y + 0.5 - config.input.shape[0] / 2) * diameter / config.input.shape[0]
    modes = {
        "tip": (input_x / diameter, np.asarray(input_grid.x) / diameter),
        "tilt": (input_y / diameter, np.asarray(input_grid.y) / diameter),
        "focus": (
            (input_x**2 + input_y**2) / diameter**2,
            (np.asarray(input_grid.x) ** 2 + np.asarray(input_grid.y) ** 2) / diameter**2,
        ),
    }
    amplitude = 0.02 * wavelength
    for name, (our_mode, hcipy_mode) in modes.items():
        ours = _normalized(sensor.photon_rate(amplitude * our_mode)) - _normalized(
            sensor.photon_rate(-amplitude * our_mode)
        )
        reference = _hcipy_pyramid_image(
            hcipy,
            optics,
            aperture,
            input_grid,
            hcipy_mode,
            wavelength,
            amplitude,
            ours.shape,
            1.0,
        ) - _hcipy_pyramid_image(
            hcipy,
            optics,
            aperture,
            input_grid,
            hcipy_mode,
            wavelength,
            amplitude,
            ours.shape,
            -1.0,
        )
        # The two packages name the detector faces from opposite viewing sides.
        # A 180-degree detector-frame rotation makes those fixed conventions coincide.
        reference = np.rot90(reference, 2)
        correlation = float(np.corrcoef(ours.ravel(), reference.ravel())[0, 1])
        gain_ratio = float(np.linalg.norm(ours) / np.linalg.norm(reference))
        assert correlation > 0.95, name
        assert 0.8 < gain_ratio < 1.25, name


@pytest.mark.validation
def test_shack_hartmann_tilt_response_curves_track_hcipy() -> None:
    """Compare two-axis SH centroid curves to HCIPy's independent Fresnel optics."""
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
            "fft_oversampling": 2,
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

    coordinate_maps = {
        "x": np.asarray(grid.x).reshape(pixels, pixels),
        "y": np.asarray(grid.y).reshape(pixels, pixels),
    }

    def hcipy_image(cycles_per_aperture: float, axis: str) -> np.ndarray:
        field = aperture * np.exp(2j * np.pi * cycles_per_aperture * coordinate_maps[axis])
        wavefront = hcipy.Wavefront(hcipy.Field(field.ravel(), grid), wavelength=wavelength)
        image = np.asarray(optics.forward(wavefront).intensity).reshape(pixels, pixels)
        # Integrate HCIPy's two-times-finer detector grid onto the configured pixels.
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
    hcipy_zero = centroids(hcipy_image(0.0, "x"))
    cycles = np.asarray([-0.3, -0.15, 0.15, 0.3])
    gains: list[float] = []
    for axis, component in (("x", 0), ("y", 1)):
        ours_curve: list[float] = []
        hcipy_curve: list[float] = []
        for cycle in cycles:
            ours_delta = (
                centroids(sensor.photon_rate(wavelength * cycle * coordinate_maps[axis]))
                - ours_zero
            )
            hcipy_delta = centroids(hcipy_image(float(cycle), axis)) - hcipy_zero
            cross_component = 1 - component
            assert np.max(np.abs(ours_delta[:, cross_component])) < 0.01
            assert np.max(np.abs(hcipy_delta[:, cross_component])) < 0.01
            ours_curve.append(float(np.mean(ours_delta[:, component])))
            hcipy_curve.append(float(np.mean(hcipy_delta[:, component])))
        ours_values = np.asarray(ours_curve)
        hcipy_values = np.asarray(hcipy_curve)
        assert np.all(np.sign(ours_values) == np.sign(cycles))
        assert np.all(np.sign(hcipy_values) == np.sign(cycles))
        assert np.corrcoef(ours_values, hcipy_values)[0, 1] > 0.999
        gains.append(
            float(np.vdot(hcipy_values, ours_values) / np.vdot(hcipy_values, hcipy_values))
        )
    # Both paths integrate a two-times-finer focal grid onto centered detector
    # cells, so their tilt-response gain should now agree quantitatively.
    assert all(0.98 < gain < 1.03 for gain in gains), gains
    assert np.isclose(gains[0], gains[1], rtol=0.01)
