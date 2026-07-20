"""Independent numerical checks for the optical kernels."""

from dataclasses import replace
from pathlib import Path

import numpy as np

from makewfs import WavefrontSensor, load_config
from makewfs.backend import centered_fft2
from makewfs.config import WFSConfig
from makewfs.sampling import spot_intensity
from makewfs.wavefront import _coordinates

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


def _direct_centered_idft(array: np.ndarray) -> np.ndarray:
    """Small independent unitary inverse DFT used by the pyramid reference."""
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
                        2j
                        * np.pi
                        * (frequency_y * input_y / height + frequency_x * input_x / width)
                    )
            result[output_y, output_x] = total / np.sqrt(height * width)
    return result


def test_centered_fft_matches_independent_small_dft() -> None:
    rng = np.random.default_rng(4)
    array = rng.normal(size=(5, 5)) + 1j * rng.normal(size=(5, 5))
    assert np.allclose(centered_fft2(array), _direct_centered_dft(array), atol=1e-12)


def test_batched_shack_spots_match_direct_random_small_grid() -> None:
    rng = np.random.default_rng(19)
    phase = rng.normal(size=(3, 4, 4))
    fields = np.exp(1j * phase)
    spots = spot_intensity(
        fields,
        pixels=4,
        samples_per_lenslet=4,
        sampling=1.0,
        oversampling=1,
        workers=1,
    )
    expected = np.stack([np.abs(_direct_centered_dft(field)) ** 2 for field in fields])
    assert np.allclose(spots, expected, atol=1e-12, rtol=1e-12)


def test_small_shack_mosaic_matches_independent_random_phase_reference() -> None:
    data = {
        "schema_version": 1,
        "input": {"quantity": "opd", "unit": "m", "shape": [8, 8], "grid_extent_m": 1.0},
        "telescope": {"pupil_diameter_m": 1.0},
        "source": {"normalization": "detector_photon_rate", "detector_photon_rate_per_s": 1.0},
        "sensor": {"kind": "shack_hartmann", "wavelength_m": 700e-9},
        "shack_hartmann": {
            "lenslets_across_pupil": 2,
            "pixels_per_subaperture": 4,
            "spot_sampling_pixels_per_lambda_over_d": 1.0,
            "minimum_illuminated_fraction": 0.0,
        },
        "detector": {"preset": "generic_cmos", "exposure_s": 0.0},
        "numerics": {
            "dtype": "float64",
            "fft_oversampling": 1,
            "pupil_samples_per_lenslet": 4,
        },
    }
    config = WFSConfig.from_dict(data)
    rng = np.random.default_rng(23)
    opd = rng.normal(scale=0.04e-6, size=config.input.shape)
    actual = WavefrontSensor(config).photon_rate(opd)

    xx, yy = _coordinates(config.input.shape, config.input.grid_extent_m)
    pupil = (np.hypot(xx, yy) <= config.telescope.pupil_diameter_m / 2).astype(float)
    field = pupil * np.exp(2j * np.pi * opd / config.sensor.wavelength_m)
    subapertures = field.reshape(2, 4, 2, 4).transpose(0, 2, 1, 3).reshape(4, 4, 4)
    direct_spots = np.stack([np.abs(_direct_centered_dft(subap)) ** 2 for subap in subapertures])
    expected = direct_spots.reshape(2, 2, 4, 4).transpose(0, 2, 1, 3).reshape(8, 8)
    expected /= np.sum(pupil)
    assert np.allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_small_pyramid_matches_independent_direct_dft_reference() -> None:
    """Verify both pyramid propagation legs without using the FFT implementation."""
    data = {
        "schema_version": 1,
        "input": {"quantity": "opd", "unit": "m", "shape": [8, 8], "grid_extent_m": 1.0},
        "telescope": {"pupil_diameter_m": 1.0},
        "source": {"normalization": "detector_photon_rate", "detector_photon_rate_per_s": 1.0},
        "sensor": {"kind": "pyramid", "wavelength_m": 700e-9},
        "pyramid": {
            "pixels_across_pupil": 8,
            "pupil_separation_pixels": 2,
            "modulation_radius_lambda_over_d": 0.0,
            "modulation_samples": 1,
        },
        "detector": {"preset": "generic_cmos", "exposure_s": 0.0},
        "numerics": {"dtype": "float64", "fft_oversampling": 1},
    }
    config = WFSConfig.from_dict(data)
    sensor = WavefrontSensor(config)
    rng = np.random.default_rng(29)
    opd = rng.normal(scale=0.025 * config.sensor.wavelength_m, size=config.input.shape)
    actual = sensor.photon_rate(opd)

    pupil = np.asarray(sensor.engine.pupil)
    field = pupil * np.exp(2j * np.pi * opd / config.sensor.wavelength_m)
    nfft = sensor.engine.nfft
    padded = np.zeros((nfft, nfft), dtype=np.complex128)
    start = (nfft - field.shape[0]) // 2
    padded[start : start + field.shape[0], start : start + field.shape[1]] = field
    focal = _direct_centered_dft(padded)
    frequencies = np.fft.fftshift(np.fft.fftfreq(nfft)) * nfft
    fy, fx = np.meshgrid(frequencies, frequencies, indexing="ij")
    sign_x = np.where(fx >= 0.0, 1.0, -1.0)
    sign_y = np.where(fy >= 0.0, 1.0, -1.0)
    half_separation = config.pyramid.pupil_separation_pixels / 2  # type: ignore[union-attr]
    mask = np.exp(-2j * np.pi * half_separation * (sign_x * fx + sign_y * fy) / nfft)
    exit_pupil = _direct_centered_idft(focal * mask)
    output_size = actual.shape[0]
    crop_start = (nfft - output_size) // 2
    expected = (
        np.abs(
            exit_pupil[
                crop_start : crop_start + output_size,
                crop_start : crop_start + output_size,
            ]
        )
        ** 2
    )
    expected /= np.sum(pupil)
    assert np.allclose(actual, expected, atol=2e-12, rtol=2e-12)


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
    assert config.source.detector_photon_rate_per_s is not None
    source = replace(
        config.source, detector_photon_rate_per_s=1.5 * config.source.detector_photon_rate_per_s
    )
    high = WavefrontSensor(replace(config, source=source)).photon_rate(np.zeros(config.input.shape))
    assert np.allclose(high, low * 1.5, rtol=1e-12, atol=1e-12)


def test_float32_and_float64_optics_agree() -> None:
    config = load_config(SH_CONFIG)
    float64 = WavefrontSensor(config).photon_rate(np.zeros(config.input.shape))
    float32_config = replace(config, numerics=replace(config.numerics, dtype="float32"))
    float32 = WavefrontSensor(float32_config).photon_rate(np.zeros(config.input.shape))
    assert np.allclose(float32, float64, rtol=3e-5, atol=1e-8)


def test_physical_lenslet_sampling_matches_normalized_mode() -> None:
    config = load_config(SH_CONFIG)
    assert config.shack_hartmann is not None
    physical = replace(
        config,
        shack_hartmann=replace(
            config.shack_hartmann,
            spot_sampling_pixels_per_lambda_over_d=None,
            lenslet_focal_length_m=0.02,
            detector_pixel_pitch_m=15e-6,
        ),
    )
    normalized = replace(
        config,
        shack_hartmann=replace(
            config.shack_hartmann,
            spot_sampling_pixels_per_lambda_over_d=0.02 * 700e-9 / 15e-6,
        ),
    )
    phase = np.zeros(config.input.shape)
    assert np.allclose(
        WavefrontSensor(physical).photon_rate(phase),
        WavefrontSensor(normalized).photon_rate(phase),
        rtol=1e-12,
        atol=1e-12,
    )


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


def test_pyramid_focus_response_and_modulation_sensitivity_trade() -> None:
    config = load_config(PYRAMID_CONFIG)
    sensor = WavefrontSensor(config)
    yy, xx = np.mgrid[: config.input.shape[0], : config.input.shape[1]]
    x = (xx - (config.input.shape[1] - 1) / 2) / config.input.shape[1]
    y = (yy - (config.input.shape[0] - 1) / 2) / config.input.shape[0]
    focus = 1.0e-7 * (x**2 + y**2)
    reference = sensor.photon_rate(np.zeros(config.input.shape))
    plus = sensor.photon_rate(focus) - reference
    minus = sensor.photon_rate(-focus) - reference
    assert float(np.vdot(plus, minus).real) < 0.0

    assert config.pyramid is not None
    modulated_config = replace(
        config,
        pyramid=replace(
            config.pyramid,
            modulation_radius_lambda_over_d=2.0,
            modulation_samples=8,
        ),
    )
    modulated = WavefrontSensor(modulated_config)
    tilt = 1.0e-8 * x
    unmodulated_response = np.linalg.norm(sensor.photon_rate(tilt) - reference)
    modulated_reference = modulated.photon_rate(np.zeros(config.input.shape))
    modulated_response = np.linalg.norm(modulated.photon_rate(tilt) - modulated_reference)
    assert modulated_response < unmodulated_response


def test_shack_hartmann_field_stop_blur_and_margin_are_optical_settings() -> None:
    config = load_config(SH_CONFIG)
    assert config.shack_hartmann is not None
    base = WavefrontSensor(config).photon_rate(np.zeros(config.input.shape))
    settings = replace(
        config.shack_hartmann,
        field_stop_radius_lambda_over_d=0.6,
        optical_blur_fwhm_pixels=1.0,
        detector_margin_pixels=2,
    )
    configured = WavefrontSensor(replace(config, shack_hartmann=settings)).photon_rate(
        np.zeros(config.input.shape)
    )
    assert configured.shape == (68, 68)
    assert configured.sum() < base.sum()
    assert np.all(configured >= 0)


def test_measured_shack_blur_kernel_is_applied(tmp_path: Path) -> None:
    config = load_config(SH_CONFIG)
    assert config.shack_hartmann is not None
    kernel = np.zeros((3, 3), dtype=float)
    kernel[1, 1] = 0.5
    kernel[1, 0] = 0.25
    kernel[1, 2] = 0.25
    path = tmp_path / "blur.npy"
    np.save(path, kernel)
    base = WavefrontSensor(config).photon_rate(np.zeros(config.input.shape))
    measured = WavefrontSensor(
        replace(
            config,
            shack_hartmann=replace(config.shack_hartmann, optical_blur_kernel_path=str(path)),
        )
    ).photon_rate(np.zeros(config.input.shape))
    assert np.all(measured >= 0)
    assert measured.shape == base.shape
    assert not np.array_equal(measured, base)


def test_rotated_offset_lenslet_grid_uses_physical_resampling_path() -> None:
    config = load_config(SH_CONFIG)
    assert config.shack_hartmann is not None
    settings = replace(
        config.shack_hartmann,
        lenslet_grid_rotation_deg=11.0,
        lenslet_grid_offset_fraction=(0.18, -0.13),
    )
    sensor = WavefrontSensor(replace(config, shack_hartmann=settings))
    zero = sensor.photon_rate(np.zeros(config.input.shape))
    piston = sensor.photon_rate(np.full(config.input.shape, 2.0e-6))
    assert zero.shape == (64, 64)
    assert np.all(zero >= 0)
    assert np.allclose(zero, piston, rtol=1e-10, atol=1e-10)


def test_explicit_zero_lenslet_transform_matches_default() -> None:
    config = load_config(SH_CONFIG)
    assert config.shack_hartmann is not None
    explicit = replace(
        config.shack_hartmann,
        lenslet_grid_rotation_deg=0.0,
        lenslet_grid_offset_fraction=(0.0, 0.0),
    )
    phase = np.zeros(config.input.shape)
    default = WavefrontSensor(config).photon_rate(phase)
    unchanged = WavefrontSensor(replace(config, shack_hartmann=explicit)).photon_rate(phase)
    assert np.array_equal(default, unchanged)


def test_pyramid_detector_margin_is_zero_filled() -> None:
    config = load_config(PYRAMID_CONFIG)
    assert config.pyramid is not None
    settings = replace(config.pyramid, detector_margin_pixels=3)
    sensor = WavefrontSensor(replace(config, pyramid=settings))
    image = sensor.photon_rate(np.zeros(config.input.shape))
    size = settings.pixels_across_pupil + settings.pupil_separation_pixels + 6
    assert image.shape == (size, size)
    assert np.all(image[:3] == 0)
    assert np.all(image[-3:] == 0)
