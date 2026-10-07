"""Shack-Hartmann lenslet-field sampling: flux conservation and spot aliasing.

A lenslet field sampled at ``s`` points across its pitch ``d`` has a far field
that repeats every ``s`` lenslet ``lambda / d``. When the detector window,
``pixels / sampling`` lenslet ``lambda / d``, is wider than that period, a
sampled transform sums the replicas as light. These tests fix the physics
against the continuous square-lenslet result rather than against the engine.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.special import sici

from makewfs import WavefrontSensor
from makewfs.config import WFSConfig, load_config
from makewfs.pupil import make_pupil
from makewfs.sampling import lenslet_field_upsampling

LENSLETS = 4
EXTENT_M = 2.0
WAVELENGTH_M = 5.0e-7


def _config(
    sampling: float,
    pixels: int,
    *,
    samples: int = 6,
    pupil_diameter_m: float = 3.0,
    dtype: str = "float64",
    device: str = "cpu",
    shack_hartmann: dict[str, Any] | None = None,
    source: dict[str, Any] | None = None,
    telescope: dict[str, Any] | None = None,
) -> WFSConfig:
    """Square lenslets; a pupil wider than the grid diagonal fills every one."""
    return WFSConfig.from_dict(
        {
            "schema_version": 1,
            "input": {
                "quantity": "opd",
                "unit": "m",
                "shape": [LENSLETS * 6, LENSLETS * 6],
                "grid_extent_m": EXTENT_M,
            },
            "telescope": {"pupil_diameter_m": pupil_diameter_m, **(telescope or {})},
            "source": {
                "kind": "ngs",
                "normalization": "detector_photon_rate",
                "detector_photon_rate_per_s": 1.0,
                **(source or {}),
            },
            "sensor": {"kind": "shack_hartmann", "wavelength_m": WAVELENGTH_M},
            "shack_hartmann": {
                "lenslets_across_pupil": LENSLETS,
                "pixels_per_subaperture": pixels,
                "spot_sampling_pixels_per_lambda_over_d": sampling,
                **(shack_hartmann or {}),
            },
            "detector": {"preset": "generic_cmos", "exposure_s": 1e-3, "temperature_c": 20.0},
            "numerics": {
                "dtype": dtype,
                "fft_oversampling": 4,
                "pupil_samples_per_lenslet": samples,
                "device": device,
            },
        }
    )


def _square_lenslet_captured_fraction(window_lambda_over_d: float) -> float:
    """Return the flux of a square lenslet's ``sinc^2`` spot in a centred window.

    ``int_{-W/2}^{W/2} sinc^2(u) du = (2/pi) [Si(pi W) - sin^2(pi W/2) / (pi W/2)]``
    per axis, from integrating by parts; the 2-D fraction is its square.
    """
    x = math.pi * window_lambda_over_d
    one_axis = (2.0 / math.pi) * (float(sici(x)[0]) - math.sin(x / 2.0) ** 2 / (x / 2.0))
    return one_axis**2


def _tilt_opd(tilt_lambda_over_d: float, shape: tuple[int, int]) -> np.ndarray:
    """Return an x OPD ramp tilting every lenslet spot by ``tilt`` lambda/d."""
    pitch_m = EXTENT_M / LENSLETS
    x = (np.arange(shape[1]) - (shape[1] - 1) / 2.0) * EXTENT_M / shape[1]
    slope = tilt_lambda_over_d * WAVELENGTH_M / pitch_m
    return np.broadcast_to(x * slope, shape).copy()


@pytest.mark.parametrize(
    ("samples", "window", "expected"),
    [
        (6, 0.0, 1),
        (6, 4.0, 1),
        (6, 6.0, 1),
        (6, 6.0 * (1 + 1e-12), 1),
        (6, 6.01, 2),
        (6, 16.0, 3),
        (4, 7.2, 2),
    ],
)
def test_lenslet_field_upsampling_is_the_smallest_adequate_integer(
    samples: int, window: float, expected: int
) -> None:
    factor = lenslet_field_upsampling(
        pupil_samples_per_lenslet=samples, window_lambda_over_d=window
    )
    assert factor == expected
    assert factor * samples >= window * (1 - 1e-9)
    assert factor == 1 or (factor - 1) * samples < window


@pytest.mark.parametrize(("samples", "window"), [(0, 4.0), (6, -1.0), (6, math.inf), (6, math.nan)])
def test_lenslet_field_upsampling_rejects_invalid_geometry(samples: int, window: float) -> None:
    with pytest.raises(ValueError):
        lenslet_field_upsampling(pupil_samples_per_lenslet=samples, window_lambda_over_d=window)


@pytest.mark.parametrize(
    ("sampling", "pixels"),
    [(0.25, 4), (0.32, 4), (0.5, 4), (0.75, 4), (0.91, 4), (2.0, 4), (0.32, 16), (0.91, 16)],
)
def test_flux_is_bounded_and_matches_the_continuous_square_lenslet(
    sampling: float, pixels: int
) -> None:
    # Before the fix, 0.25 px with 4 px returned 8.6x the configured rate and
    # 0.32 px with 16 px returned 79x, because the window spans several periods
    # of a field sampled six times per lenslet.
    sensor = WavefrontSensor(_config(sampling, pixels))
    engine = sensor.engine
    window = pixels / sampling
    assert engine.detector_window_lambda_over_d == pytest.approx(window)
    assert engine.samples_per_lenslet >= window
    assert engine.samples_per_lenslet == engine.pupil_samples_per_lenslet * engine.field_upsampling

    captured = float(np.sum(sensor.photon_rate(np.zeros(sensor.config.input.shape))))
    assert captured <= 1.0 + 1e-9
    # The residual is the documented sampled-field wing error, largest when
    # the grid only just reaches the window (0.75 px keeps its adequate six
    # samples and is 7% high); it is percent-level, not the multiples the
    # aliasing produced, and converges with pupil_samples_per_lenslet below.
    exact = _square_lenslet_captured_fraction(window)
    assert captured == pytest.approx(exact, rel=0.08)


def test_finer_pupil_sampling_converges_to_the_continuous_square_lenslet() -> None:
    window = 4 / 0.32
    exact = _square_lenslet_captured_fraction(window)
    errors = []
    for samples in (6, 32, 64):
        sensor = WavefrontSensor(_config(0.32, 4, samples=samples))
        captured = float(np.sum(sensor.photon_rate(np.zeros(sensor.config.input.shape))))
        errors.append(abs(captured / exact - 1.0))
    assert errors[0] > errors[1] > errors[2]
    assert errors[-1] < 2e-3


def test_tilted_spot_lands_where_the_continuous_field_puts_it() -> None:
    # A 5 lambda/d tilt is inside the 16 lambda/d window but beyond the
    # +-3 lambda/d that six samples per lenslet can represent: the old engine
    # put the spot at the alias 5 - 6 = -1 lambda/d, on the wrong side.
    sampling, pixels, tilt = 0.25, 4, 5.0
    sensor = WavefrontSensor(_config(sampling, pixels))
    reference = WavefrontSensor(_config(sampling, pixels, samples=48))
    opd = _tilt_opd(tilt, sensor.config.input.shape)

    image = sensor.photon_rate(opd)
    expected = reference.photon_rate(opd)
    spots = image.reshape(LENSLETS, pixels, LENSLETS, pixels).sum(axis=(0, 2))
    # Pixel centres sit at -1.5, -0.5, 0.5, 1.5 px; the spot is at +1.25 px.
    assert np.argmax(spots.sum(axis=0)) == 3
    assert spots[:, :2].sum() < 0.05 * spots.sum()
    assert np.abs(image - expected).sum() < 0.05 * expected.sum()


def test_shortest_wavelength_sets_the_refinement_and_every_node_is_bounded() -> None:
    # A HAKA-like spectral geometry: 4 px at 0.91 px/(lambda/d) at 673 nm and a
    # quadrature reaching 411 nm, where the window is 7.2 lambda/d.
    wavelengths = (4.11e-7, 5.3e-7, 6.73e-7, 9.39e-7)
    sensor = WavefrontSensor(
        _config(
            0.91 * WAVELENGTH_M / 6.73e-7,
            4,
            samples=4,
            source={"wavelengths_m": list(wavelengths), "wavelength_weights": [1, 1, 1, 1]},
        )
    )
    engine = sensor.engine
    assert engine.detector_window_lambda_over_d == pytest.approx(4 / (0.91 * 411 / 673))
    assert engine.field_upsampling == 2
    result = engine.render(np.zeros(sensor.config.input.shape))
    per_node = np.asarray(result.spectral_photon_rate).sum(axis=(1, 2)) / 0.25
    assert np.all(per_node <= 1.0 + 1e-9)
    # Shorter wavelengths see a wider window in lambda/d, so capture more.
    assert np.all(np.diff(per_node) < 0)


def test_field_stop_bounds_the_window_that_needs_sampling() -> None:
    sensor = WavefrontSensor(
        _config(0.25, 4, shack_hartmann={"field_stop_radius_lambda_over_d": 2.5})
    )
    assert sensor.engine.detector_window_lambda_over_d == pytest.approx(5.0)
    assert sensor.engine.field_upsampling == 1


def test_closed_field_stop_needs_no_refinement() -> None:
    sensor = WavefrontSensor(
        _config(0.25, 4, shack_hartmann={"field_stop_radius_lambda_over_d": 0.0})
    )
    assert sensor.engine.field_upsampling == 1
    assert float(np.sum(sensor.photon_rate(np.zeros(sensor.config.input.shape)))) == 0.0


def test_rotated_lenslet_grid_is_refined_and_flux_bounded() -> None:
    sensor = WavefrontSensor(
        _config(
            0.25,
            4,
            pupil_diameter_m=1.8,
            shack_hartmann={
                "lenslet_grid_rotation_deg": 10.0,
                "lenslet_grid_offset_fraction": [0.1, -0.2],
            },
        )
    )
    reference = WavefrontSensor(
        _config(
            0.25,
            4,
            samples=48,
            pupil_diameter_m=1.8,
            shack_hartmann={
                "lenslet_grid_rotation_deg": 10.0,
                "lenslet_grid_offset_fraction": [0.1, -0.2],
            },
        )
    )
    assert sensor.engine.field_upsampling == 3
    opd = _tilt_opd(3.0, sensor.config.input.shape)
    image = sensor.photon_rate(opd)
    expected = reference.photon_rate(opd)
    assert float(image.sum()) <= 1.0
    assert float(image.sum()) == pytest.approx(float(expected.sum()), rel=0.05)


def test_adequate_sampling_keeps_the_configured_grid_unchanged() -> None:
    config = load_config(
        Path(__file__).parents[1] / "examples" / "configs" / "shack_hartmann_minimal.toml"
    )
    engine = WavefrontSensor(config).engine
    assert engine.field_upsampling == 1
    assert engine.samples_per_lenslet == engine.pupil_samples_per_lenslet == 16
    assert engine.internal_shape == engine.pupil_shape
    np.testing.assert_array_equal(
        engine.pupil,
        make_pupil(
            config.telescope,
            engine.internal_shape,
            config.input.grid_extent_m,
            supersampling=config.numerics.pupil_supersampling,
            dtype=np.float32,
        ),
    )


def test_refined_grid_preserves_custom_mask_and_lenslet_illumination(tmp_path: Path) -> None:
    analytic = _config(0.25, 4, pupil_diameter_m=1.8, telescope={"central_obscuration_ratio": 0.3})
    reference = WavefrontSensor(analytic).engine
    assert reference.pupil_shape == (LENSLETS * 6, LENSLETS * 6)
    mask = np.asarray(
        make_pupil(analytic.telescope, reference.pupil_shape, EXTENT_M), dtype=np.float64
    )
    path = tmp_path / "mask.npy"
    np.save(path, mask)
    custom = replace(analytic, telescope=replace(analytic.telescope, custom_mask_path=str(path)))
    # A custom mask is supplied on the configured pupil grid, not the refined one.
    engine = WavefrontSensor(custom).engine
    assert engine.field_upsampling == 3
    assert engine.internal_shape == (LENSLETS * 18, LENSLETS * 18)
    np.testing.assert_array_equal(engine.pupil, np.repeat(np.repeat(mask, 3, axis=0), 3, axis=1))
    coarse_illumination = mask.reshape(LENSLETS, 6, LENSLETS, 6).mean(axis=(1, 3))
    np.testing.assert_allclose(engine.lenslet_illumination, coarse_illumination, rtol=1e-12)
    np.testing.assert_allclose(reference.lenslet_illumination, coarse_illumination, rtol=1e-12)
    assert float(engine._total_field_flux) == pytest.approx(9.0 * float(np.sum(mask**2)))


def test_float32_matches_float64_on_a_refined_grid() -> None:
    double = WavefrontSensor(_config(0.32, 4))
    single = WavefrontSensor(_config(0.32, 4, dtype="float32"))
    opd = _tilt_opd(2.0, double.config.input.shape)
    np.testing.assert_allclose(
        single.photon_rate(opd), double.photon_rate(opd), rtol=2e-4, atol=1e-6
    )


def _cupy() -> Any:
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CuPy is installed but no CUDA device is available")
    except Exception as exc:  # pragma: no cover - depends on local CUDA runtime
        pytest.skip(f"CUDA runtime is unavailable: {exc}")
    return cupy


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("sampling", "pixels", "compiled"),
    # 0.25 px refines to an exact integer FFT grid and 0.32/0.91 px to sampled
    # DFTs; on a device the compiled CUDA executor evaluates all of them.
    [(0.25, 4, True), (0.32, 4, True), (0.91, 16, True)],
)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_gpu_refined_grid_matches_cpu_and_conserves_flux(
    sampling: float, pixels: int, compiled: bool, dtype: str
) -> None:  # pragma: no cover - optional CUDA execution
    cupy = _cupy()
    cpu = WavefrontSensor(_config(sampling, pixels, dtype=dtype))
    gpu = WavefrontSensor(_config(sampling, pixels, dtype=dtype, device="gpu"))
    assert gpu.engine.samples_per_lenslet == cpu.engine.samples_per_lenslet
    opd = _tilt_opd(2.0, cpu.config.input.shape)
    expected = cpu.photon_rate(opd)
    actual = cupy.asnumpy(gpu.photon_rate(cupy.asarray(opd)))
    assert bool(gpu.engine._compiled_executors) is compiled
    tolerance = 5e-5 if dtype == "float32" else 1e-10
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance * expected.max())
    assert float(actual.sum()) <= 1.0 + 1e-6
