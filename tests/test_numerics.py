"""Small numerical helper and radiometry tests."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from makewfs.backend import centered_fft2, centered_ifft2, complex_dtype, real_dtype
from makewfs.config import SourceConfig, SpiderConfig, TelescopeConfig
from makewfs.pupil import make_pupil
from makewfs.radiometry import source_rate_per_s
from makewfs.sampling import block_sum, crop_center, pad_center, spot_intensity


def test_centered_fft_roundtrip_and_dtype() -> None:
    array = np.zeros((8, 8), dtype=np.complex64)
    array[3, 4] = 1.0
    transformed = centered_fft2(array)
    assert transformed.dtype == np.complex64
    assert np.allclose(centered_ifft2(transformed), array, atol=1e-6)


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
    assert block_sum(array, 2).sum() == array.sum()


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


def test_magnitude_rate_follows_pogson_scaling() -> None:
    telescope = TelescopeConfig(8.0)
    bright = SourceConfig.from_dict({"normalization": "magnitude", "magnitude": 10, "band": "R"})
    faint = replace(bright, magnitude=12.0)
    ratio = source_rate_per_s(bright, telescope) / source_rate_per_s(faint, telescope)
    assert np.isclose(ratio, 10.0**0.8, rtol=1e-12)
