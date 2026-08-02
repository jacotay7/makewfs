"""Small numerical helper and radiometry tests."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from makewfs.backend import (
    centered_fft2,
    centered_fft_intensity,
    centered_ifft2,
    complex_dtype,
    cpu_backend,
    real_dtype,
)
from makewfs.config import SourceConfig, SpiderConfig, TelescopeConfig
from makewfs.pupil import make_pupil
from makewfs.radiometry import source_rate_per_s
from makewfs.sampling import (
    _SpotPropagationPlan,
    block_sum,
    crop_center,
    load_blur_kernel,
    pad_center,
    spot_intensity,
)
from makewfs.sensors._shack_hartmann_cuda import _fused_integration_weights


def test_centered_fft_roundtrip_and_dtype() -> None:
    array = np.zeros((8, 8), dtype=np.complex64)
    array[3, 4] = 1.0
    transformed = centered_fft2(array)
    assert transformed.dtype == np.complex64
    assert np.allclose(centered_ifft2(transformed), array, atol=1e-6)


def test_centered_fft_intensity_omits_only_irrelevant_fourier_phase() -> None:
    rng = np.random.default_rng(4)
    field = (rng.normal(size=(3, 8, 8)) + 1j * rng.normal(size=(3, 8, 8))).astype(np.complex64)
    expected = np.abs(centered_fft2(field)) ** 2
    actual = centered_fft_intensity(field)
    assert actual.dtype == np.float32
    assert np.allclose(actual, expected, rtol=2e-6, atol=2e-6)


def test_fft_worker_count_is_deterministic() -> None:
    rng = np.random.default_rng(31)
    array = (rng.normal(size=(16, 16)) + 1j * rng.normal(size=(16, 16))).astype(np.complex64)
    assert np.array_equal(centered_fft2(array, workers=1), centered_fft2(array, workers=2))


def test_spot_integration_preserves_float32_batch_precision() -> None:
    field = np.ones((1, 8, 8), dtype=np.complex64)
    spots = spot_intensity(
        field,
        pixels=8,
        samples_per_lenslet=8,
        sampling=1.0,
        oversampling=1,
        workers=1,
    )
    assert spots.dtype == np.float32


def test_fused_diffusion_and_pixel_integration_matches_reference() -> None:
    from scipy.ndimage import convolve

    pixels = 3
    oversampling = 2
    high_pixels = pixels * oversampling
    intensity = np.arange(high_pixels**2, dtype=np.float64).reshape(high_pixels, high_pixels)
    # Deliberately asymmetric: this verifies convolution orientation as well as
    # the zero-padded focal-plane boundary and native-pixel collection.
    kernel = np.arange(1.0, 10.0).reshape(3, 3)
    weights = _fused_integration_weights(
        kernel,
        pixels=pixels,
        oversampling=oversampling,
        dtype=np.dtype(np.float64),
    )

    expected = block_sum(convolve(intensity, kernel, mode="constant"), oversampling)
    actual = (weights @ intensity.ravel()).reshape(pixels, pixels)

    np.testing.assert_array_equal(actual, expected)


def test_cached_spot_propagation_plan_preserves_optics() -> None:
    rng = np.random.default_rng(19)
    field = (rng.normal(size=(3, 8, 8)) + 1j * rng.normal(size=(3, 8, 8))).astype(np.complex128)
    kwargs = {
        "pixels": 4,
        "samples_per_lenslet": 8,
        "sampling": 0.91,
        "oversampling": 2,
        "workers": 1,
        "field_stop_radius_lambda_over_d": 0.8,
    }
    plan = _SpotPropagationPlan.build(
        field_dtype=field.dtype,
        optical_blur_kernel=None,
        charge_diffusion_kernel=None,
        backend=cpu_backend(),
        pixels=kwargs["pixels"],
        samples_per_lenslet=kwargs["samples_per_lenslet"],
        sampling=kwargs["sampling"],
        oversampling=kwargs["oversampling"],
        field_stop_radius_lambda_over_d=kwargs["field_stop_radius_lambda_over_d"],
    )
    reference = spot_intensity(field, **kwargs)
    cached = spot_intensity(field, _plan=plan, **kwargs)
    np.testing.assert_allclose(cached, reference, rtol=1e-13, atol=1e-13)
    with pytest.raises(ValueError, match="does not match"):
        spot_intensity(field, _plan=plan, **{**kwargs, "field_stop_radius_lambda_over_d": None})


def test_even_quadcell_is_centered_on_four_pixels_for_flat_wavefront() -> None:
    field = np.ones((1, 8, 8), dtype=np.complex128)
    spot = spot_intensity(
        field,
        pixels=4,
        samples_per_lenslet=8,
        sampling=0.91,
        oversampling=2,
        workers=1,
    )[0]
    yy, xx = np.indices(spot.shape, dtype=np.float64)
    assert np.sum(spot * xx) / np.sum(spot) == pytest.approx(1.5, abs=1e-12)
    assert np.sum(spot * yy) / np.sum(spot) == pytest.approx(1.5, abs=1e-12)
    np.testing.assert_allclose(spot[1:3, 1:3], spot[1, 1], rtol=1e-12, atol=1e-12)


def test_backend_rejects_unknown_dtype() -> None:
    with pytest.raises(ValueError, match="unsupported dtype"):
        real_dtype("float16")
    with pytest.raises(ValueError, match="unsupported dtype"):
        complex_dtype("float16")


def test_padding_cropping_and_block_sum_preserve_flux() -> None:
    array = np.ones((4, 4))
    padded = pad_center(array, (8, 8))
    assert padded.sum() == array.sum()
    assert np.array_equal(crop_center(padded, (4, 4)), array)
    assert np.array_equal(block_sum(array, 1), array)
    assert block_sum(array, 2).sum() == array.sum()
    batch = np.arange(3 * 8 * 8, dtype=np.float32).reshape(3, 8, 8)
    expected = batch.reshape(3, 4, 2, 4, 2).sum(axis=(-1, -3))
    np.testing.assert_array_equal(block_sum(batch, 2), expected)


def test_sampling_helpers_reject_invalid_geometry() -> None:
    array = np.ones((4, 4))
    with pytest.raises(ValueError, match="cannot crop"):
        pad_center(array, (2, 4))
    with pytest.raises(ValueError, match="cannot enlarge"):
        crop_center(array, (5, 4))
    with pytest.raises(ValueError, match="positive"):
        block_sum(array, 0)
    with pytest.raises(ValueError, match="divisible"):
        block_sum(array, 3)


def test_custom_pupil_and_spider_masks(tmp_path: Path) -> None:
    custom_path = tmp_path / "mask.npy"
    mask = np.ones((16, 16))
    np.save(custom_path, mask)
    custom = TelescopeConfig(1.0, custom_mask_path=str(custom_path))
    assert np.array_equal(make_pupil(custom, (16, 16), 1.0), mask)
    spidered = replace(custom, custom_mask_path=None, central_obscuration_ratio=0.2)
    spidered = replace(spidered, spiders=())
    assert make_pupil(spidered, (16, 16), 1.0).sum() > 0
    spidered = replace(spidered, spiders=(SpiderConfig(0.0, 0.1),))
    assert (
        make_pupil(spidered, (16, 16), 1.0).sum()
        < make_pupil(replace(spidered, spiders=()), (16, 16), 1.0).sum()
    )


def test_rotated_segmented_analytic_pupil_has_explicit_gaps() -> None:
    plain = TelescopeConfig(1.0)
    segmented = TelescopeConfig(
        1.0,
        pupil_rotation_deg=23.0,
        segments_across_pupil=4,
        segment_gap_fraction=0.08,
    )
    plain_mask = make_pupil(plain, (128, 128), 1.0, supersampling=4)
    segmented_mask = make_pupil(segmented, (128, 128), 1.0, supersampling=4)
    assert np.all(segmented_mask >= 0)
    assert np.all(segmented_mask <= 1)
    assert segmented_mask.sum() < plain_mask.sum()
    assert not np.array_equal(segmented_mask, make_pupil(plain, (128, 128), 1.0))


def test_custom_pupil_archives_and_validation(tmp_path: Path) -> None:
    from makewfs.pupil import make_pupil

    archive = tmp_path / "mask.npz"
    np.savez(archive, first=np.ones((4, 4)))
    config = TelescopeConfig(1.0, custom_mask_path=str(archive))
    assert make_pupil(config, (4, 4), 1.0).sum() == 16
    invalid = tmp_path / "bad.npy"
    np.save(invalid, np.full((4, 4), 2.0))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        make_pupil(TelescopeConfig(1.0, custom_mask_path=str(invalid)), (4, 4), 1.0)
    with pytest.raises(ValueError, match="supersampling"):
        make_pupil(TelescopeConfig(1.0), (4, 4), 1.0, supersampling=0)


def test_measured_blur_kernel_is_normalized_and_validated(tmp_path: Path) -> None:
    path = tmp_path / "blur.npy"
    kernel = np.zeros((3, 3))
    kernel[1, 1] = 2.0
    kernel[1, 2] = 1.0
    np.save(path, kernel)
    loaded = load_blur_kernel(str(path))
    assert np.isclose(loaded.sum(), 1.0)
    assert loaded[1, 2] > loaded[0, 0]
    even = tmp_path / "even.npy"
    np.save(even, np.ones((2, 2)))
    with pytest.raises(ValueError, match="odd-sized"):
        load_blur_kernel(str(even))
    bad_suffix = tmp_path / "blur.dat"
    bad_suffix.write_text("0", encoding="utf-8")
    with pytest.raises(ValueError, match="support"):
        load_blur_kernel(str(bad_suffix))
    empty = tmp_path / "empty.npz"
    np.savez(empty)
    with pytest.raises(ValueError, match="contains no arrays"):
        load_blur_kernel(str(empty))
    negative = tmp_path / "negative.npy"
    np.save(negative, -np.ones((3, 3)))
    with pytest.raises(ValueError, match="non-negative"):
        load_blur_kernel(str(negative))


def test_spot_blur_modes_are_mutually_exclusive() -> None:
    field = np.ones((1, 4, 4), dtype=np.complex128)
    with pytest.raises(ValueError, match="either"):
        spot_intensity(
            field,
            pixels=4,
            samples_per_lenslet=4,
            sampling=1.0,
            oversampling=1,
            workers=1,
            optical_blur_fwhm_pixels=1.0,
            optical_blur_kernel=np.ones((3, 3)) / 9.0,
        )


def test_magnitude_rate_follows_pogson_scaling() -> None:
    telescope = TelescopeConfig(8.0)
    bright = SourceConfig.from_dict({"normalization": "magnitude", "magnitude": 10, "band": "R"})
    faint = replace(bright, magnitude=12.0)
    ratio = source_rate_per_s(bright, telescope) / source_rate_per_s(faint, telescope)
    assert np.isclose(ratio, 10.0**0.8, rtol=1e-12)
