"""Supplementary independent optical cross-checks using a local OOPAO checkout."""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest

from makewfs import WavefrontSensor
from makewfs.config import WFSConfig


def _normalized(image: np.ndarray) -> np.ndarray:
    return image / np.sum(image)


def _centroids(image: np.ndarray, lenslets: int, pixels_per_subaperture: int) -> np.ndarray:
    values: list[tuple[float, float]] = []
    yy, xx = np.indices((pixels_per_subaperture, pixels_per_subaperture))
    for lenslet_y in range(lenslets):
        for lenslet_x in range(lenslets):
            subimage = image[
                lenslet_y * pixels_per_subaperture : (lenslet_y + 1) * pixels_per_subaperture,
                lenslet_x * pixels_per_subaperture : (lenslet_x + 1) * pixels_per_subaperture,
            ]
            values.append(
                (
                    float(np.sum(subimage * xx) / np.sum(subimage)),
                    float(np.sum(subimage * yy) / np.sum(subimage)),
                )
            )
    return np.asarray(values)


def _config(kind: str, wavelength_m: float) -> WFSConfig:
    sensor_table: dict[str, object]
    if kind == "shack_hartmann":
        sensor_table = {
            "shack_hartmann": {
                "lenslets_across_pupil": 2,
                "pixels_per_subaperture": 8,
                "spot_sampling_pixels_per_lambda_over_d": 2.0,
                "minimum_illuminated_fraction": 0.0,
            },
            "numerics": {
                "dtype": "float64",
                "fft_oversampling": 1,
                "pupil_samples_per_lenslet": 16,
            },
        }
    else:
        sensor_table = {
            "pyramid": {
                "pixels_across_pupil": 32,
                "pupil_separation_pixels": 32,
                "modulation_radius_lambda_over_d": 0.0,
                "modulation_samples": 1,
            },
            "numerics": {"dtype": "float64", "fft_oversampling": 2},
        }
    data = {
        "schema_version": 1,
        "input": {"quantity": "opd", "unit": "m", "shape": [32, 32], "grid_extent_m": 1.0},
        "telescope": {"pupil_diameter_m": 1.0},
        "source": {"normalization": "detector_photon_rate", "detector_photon_rate_per_s": 1.0},
        "sensor": {"kind": kind, "wavelength_m": wavelength_m},
        "detector": {"preset": "generic_cmos", "exposure_s": 0.0},
        **sensor_table,
    }
    return WFSConfig.from_dict(data)


@pytest.mark.validation
def test_shack_hartmann_tilt_response_tracks_oopao() -> None:
    """Compare SH response gain and axis isolation with OOPAO's diffractive path."""
    pytest.importorskip("OOPAO")
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        from OOPAO.ShackHartmann import ShackHartmann
        from OOPAO.Source import Source
        from OOPAO.Telescope import Telescope

        source = Source(optBand="V", magnitude=0, display_properties=False)
        telescope = Telescope(resolution=32, diameter=1.0, samplingTime=1.0)
        source * telescope
        oopao = ShackHartmann(
            nSubap=2,
            telescope=telescope,
            lightRatio=0.0,
            shannon_sampling=True,
            is_geometric=False,
            n_pixel_per_subaperture=8,
        )

    config = _config("shack_hartmann", float(source.wavelength))
    sensor = WavefrontSensor(config)
    yy, xx = np.indices(config.input.shape, dtype=np.float64)
    coordinate_maps = {
        "x": (xx + 0.5 - config.input.shape[1] / 2) / config.input.shape[1],
        "y": (yy + 0.5 - config.input.shape[0] / 2) / config.input.shape[0],
    }
    ours_zero = _centroids(sensor.reference(), 2, 8)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        oopao.wfs_measure(phase_in=np.zeros(config.input.shape))
    oopao_zero = _centroids(np.asarray(oopao.raw_data), 2, 8)
    cycles = np.asarray([-0.2, -0.1, 0.1, 0.2])
    gains: list[float] = []
    for axis, our_component, oopao_component in (("x", 0, 1), ("y", 1, 0)):
        ours_curve: list[float] = []
        oopao_curve: list[float] = []
        for cycle in cycles:
            mode = coordinate_maps[axis]
            ours_delta = (
                _centroids(sensor.photon_rate(config.sensor.wavelength_m * cycle * mode), 2, 8)
                - ours_zero
            )
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                oopao.wfs_measure(phase_in=2 * np.pi * cycle * mode)
            oopao_delta = _centroids(np.asarray(oopao.raw_data), 2, 8) - oopao_zero
            assert np.max(np.abs(ours_delta[:, 1 - our_component])) < 0.01
            assert np.max(np.abs(oopao_delta[:, 1 - oopao_component])) < 0.01
            ours_curve.append(float(np.mean(ours_delta[:, our_component])))
            oopao_curve.append(float(np.mean(oopao_delta[:, oopao_component])))
        ours_values = np.asarray(ours_curve)
        oopao_values = np.asarray(oopao_curve)
        assert np.corrcoef(ours_values, oopao_values)[0, 1] > 0.999
        gains.append(
            float(np.vdot(oopao_values, ours_values) / np.vdot(oopao_values, oopao_values))
        )
    # OOPAO stores its raw detector axes transposed relative to makewfs, but the
    # physical x/y response and sampling-adjusted gains agree after that mapping.
    assert all(0.85 < gain < 1.1 for gain in gains)
    assert np.isclose(gains[0], gains[1], rtol=0.01)


@pytest.mark.validation
def test_pyramid_low_order_response_maps_track_oopao() -> None:
    """Compare unmodulated pyramid push/pull maps with OOPAO's Fourier model."""
    pytest.importorskip("OOPAO")
    try:
        import cupy

        if cupy.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("this OOPAO Pyramid version selects CuPy when installed and needs a GPU")
    except ImportError:
        pass
    except Exception as exc:
        pytest.skip(f"OOPAO's optional CuPy backend is unavailable: {exc}")

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        from OOPAO.Pyramid import Pyramid
        from OOPAO.Source import Source
        from OOPAO.Telescope import Telescope

        source = Source(optBand="V", magnitude=0, display_properties=False)
        telescope = Telescope(resolution=32, diameter=1.0, samplingTime=1.0)
        source * telescope
        oopao = Pyramid(
            nSubap=32,
            telescope=telescope,
            modulation=0.0,
            lightRatio=0.0,
            postProcessing="fullFrame_incidence_flux",
            psfCentering=True,
            n_pix_separation=0,
            n_pix_edge=0,
        )

    config = _config("pyramid", float(source.wavelength))
    sensor = WavefrontSensor(config)
    yy, xx = np.indices(config.input.shape, dtype=np.float64)
    x = (xx + 0.5 - config.input.shape[1] / 2) / config.input.shape[1]
    y = (yy + 0.5 - config.input.shape[0] / 2) / config.input.shape[0]
    modes = {"tip": x, "tilt": y, "focus": x**2 + y**2}
    amplitude_waves = 0.02
    for name, mode in modes.items():
        ours = _normalized(
            sensor.photon_rate(config.sensor.wavelength_m * amplitude_waves * mode)
        ) - _normalized(sensor.photon_rate(-config.sensor.wavelength_m * amplitude_waves * mode))
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            oopao.wfs_measure(phase_in=2 * np.pi * amplitude_waves * mode)
        plus = _normalized(np.asarray(oopao.raw_data))
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            oopao.wfs_measure(phase_in=-2 * np.pi * amplitude_waves * mode)
        reference = plus - _normalized(np.asarray(oopao.raw_data))
        correlation = float(np.corrcoef(ours.ravel(), reference.ravel())[0, 1])
        gain_ratio = float(np.linalg.norm(ours) / np.linalg.norm(reference))
        assert correlation > 0.9, name
        assert 0.75 < gain_ratio < 1.3, name
