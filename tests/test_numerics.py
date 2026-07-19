"""Small numerical helper and radiometry tests."""

from dataclasses import replace
from pathlib import Path

import numpy as np

from makewfs.backend import centered_fft2, centered_ifft2
from makewfs.config import SourceConfig, TelescopeConfig
from makewfs.pupil import make_pupil
from makewfs.radiometry import source_rate_per_s
from makewfs.sampling import block_sum, crop_center, pad_center


def test_centered_fft_roundtrip_and_dtype() -> None:
    array = np.zeros((8, 8), dtype=np.complex64)
    array[3, 4] = 1.0
    transformed = centered_fft2(array)
    assert transformed.dtype == np.complex64
    assert np.allclose(centered_ifft2(transformed), array, atol=1e-6)


def test_padding_cropping_and_block_sum_preserve_flux() -> None:
    array = np.ones((4, 4))
    padded = pad_center(array, (8, 8))
    assert padded.sum() == array.sum()
    assert np.array_equal(crop_center(padded, (4, 4)), array)
    assert block_sum(array, 2).sum() == array.sum()


def test_custom_pupil_and_spider_masks(tmp_path: Path) -> None:
    custom_path = tmp_path / "mask.npy"
    mask = np.ones((16, 16))
    np.save(custom_path, mask)
    custom = TelescopeConfig(1.0, custom_mask_path=str(custom_path))
    assert np.array_equal(make_pupil(custom, (16, 16), 1.0), mask)
    spidered = replace(custom, custom_mask_path=None, spiders=())
    assert make_pupil(spidered, (16, 16), 1.0).sum() > 0


def test_magnitude_rate_follows_pogson_scaling() -> None:
    telescope = TelescopeConfig(8.0)
    bright = SourceConfig.from_dict({"normalization": "magnitude", "magnitude": 10, "band": "R"})
    faint = replace(bright, magnitude=12.0)
    ratio = source_rate_per_s(bright, telescope) / source_rate_per_s(faint, telescope)
    assert np.isclose(ratio, 10.0**0.8, rtol=1e-12)
