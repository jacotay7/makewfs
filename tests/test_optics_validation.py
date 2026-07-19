"""Independent numerical checks for the optical kernels."""

from dataclasses import replace
from pathlib import Path

import numpy as np

from makewfs import WavefrontSensor, load_config
from makewfs.backend import centered_fft2
from makewfs.sampling import spot_intensity

SH_CONFIG = Path(__file__).parents[1] / "examples" / "configs" / "shack_hartmann_minimal.toml"
PYRAMID_CONFIG = Path(__file__).parents[1] / "examples" / "configs" / "pyramid_minimal.toml"


def _direct_centered_dft(array: np.ndarray) -> np.ndarray:
    """Small independent unitary DFT used only for a reference assertion."""
    shifted = np.fft.ifftshift(array)
    height, width = shifted.shape
    result = np.zeros_like(shifted, dtype=np.complex128)
    for output_y in range(height):
        frequency_y = (output_y - height // 2) % height
        for output_x in range(width):
            frequency_x = (output_x - width // 2) % width
            total = 0j
            for input_y in range(height):
                for input_x in range(width):
                    total += shifted[input_y, input_x] * np.exp(
                        -2j
                        * np.pi
                        * (frequency_y * input_y / height + frequency_x * input_x / width)
                    )
            result[output_y, output_x] = total / np.sqrt(height * width)
    return result


def test_centered_fft_matches_independent_small_dft() -> None:
    rng = np.random.default_rng(4)
    array = rng.normal(size=(5, 5)) + 1j * rng.normal(size=(5, 5))
    assert np.allclose(centered_fft2(array), _direct_centered_dft(array), atol=1e-12)


def test_known_opd_ramp_moves_a_spot_by_the_analytic_amount() -> None:
    samples = 16
    physical_width_m = 1.0
    wavelength_m = 700e-9
    sampling = 2.0
    slope_rad = 1e-7
    x = (np.arange(samples, dtype=np.float64) + 0.5 - samples / 2) * physical_width_m / samples
    flat = np.ones((samples, samples), dtype=np.complex128)
    tilted = np.exp(2j * np.pi * slope_rad * x[None, :] / wavelength_m)
    flat_spot = spot_intensity(
        flat[None],
        pixels=32,
        samples_per_lenslet=samples,
        sampling=sampling,
        oversampling=2,
        workers=1,
    )[0]
    tilted_spot = spot_intensity(
        tilted[None],
        pixels=32,
        samples_per_lenslet=samples,
        sampling=sampling,
        oversampling=2,
        workers=1,
    )[0]
    _, pixels = np.indices(flat_spot.shape)
    flat_centroid = float(np.sum(flat_spot * pixels) / flat_spot.sum())
    tilted_centroid = float(np.sum(tilted_spot * pixels) / tilted_spot.sum())
    expected = slope_rad * physical_width_m / wavelength_m * sampling
    assert np.isclose(tilted_centroid - flat_centroid, expected, rtol=0.05, atol=0.01)


def test_source_rate_scales_ideal_image_linearly() -> None:
    config = load_config(SH_CONFIG)
    low = WavefrontSensor(config).photon_rate(np.zeros(config.input.shape))
    source = replace(config.source, detector_photon_rate_per_s=3.0e6)
    high = WavefrontSensor(replace(config, source=source)).photon_rate(np.zeros(config.input.shape))
    assert np.allclose(high, low * 1.5, rtol=1e-12, atol=1e-12)


def test_float32_and_float64_optics_agree() -> None:
    config = load_config(SH_CONFIG)
    float64 = WavefrontSensor(config).photon_rate(np.zeros(config.input.shape))
    float32_config = replace(config, numerics=replace(config.numerics, dtype="float32"))
    float32 = WavefrontSensor(float32_config).photon_rate(np.zeros(config.input.shape))
    assert np.allclose(float32, float64, rtol=3e-5, atol=1e-8)


def test_pyramid_piston_invariance_and_push_pull_antisymmetry() -> None:
    sensor = WavefrontSensor.from_toml(PYRAMID_CONFIG)
    zero = np.zeros(sensor.config.input.shape)
    x = np.linspace(-1.0, 1.0, sensor.config.input.shape[1])[None, :]
    tilt = np.broadcast_to(5e-8 * x, sensor.config.input.shape)
    assert np.allclose(sensor.photon_rate(zero), sensor.photon_rate(zero + 2e-6))
    plus = sensor.photon_rate(tilt)
    minus = sensor.photon_rate(-tilt)
    plus_response = plus - sensor.photon_rate(zero)
    minus_response = minus - sensor.photon_rate(zero)
    assert np.linalg.norm(plus_response) > 0.0
    assert float(np.vdot(plus_response, minus_response).real) < 0.0
