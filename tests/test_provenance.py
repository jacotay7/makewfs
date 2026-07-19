"""Provenance and referenced-file hashing tests."""

from dataclasses import replace
from pathlib import Path

import numpy as np

from makewfs import WavefrontSensor, load_config


def test_source_curve_digest_is_recorded(tmp_path: Path) -> None:
    path = tmp_path / "sed.txt"
    np.savetxt(path, [[600.0, 1.0], [700.0, 1.0]])
    config = load_config(
        Path(__file__).parents[1] / "examples" / "configs" / "shack_hartmann_minimal.toml"
    )
    source = replace(config.source, wavelengths_m=(), wavelength_weights=(), sed_path=str(path))
    sensor = WavefrontSensor(replace(config, source=source))
    frame = sensor.expose(np.zeros(config.input.shape), seed=1)
    assert len(frame.metadata["wfs_source_sed_sha256"]) == 16
    assert frame.metadata["wfs_source_kind"] == "ngs"
    assert np.isclose(frame.metadata["wfs_source_states"][0]["wavelength_m"], 6.0e-7)


def test_angular_kernel_digest_is_recorded(tmp_path: Path) -> None:
    path = tmp_path / "kernel.txt"
    np.savetxt(path, [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]])
    config = load_config(
        Path(__file__).parents[1] / "examples" / "configs" / "shack_hartmann_minimal.toml"
    )
    source = replace(config.source, angular_kernel_path=str(path))
    frame = WavefrontSensor(replace(config, source=source)).expose(
        np.zeros(config.input.shape), seed=2
    )
    assert len(frame.metadata["wfs_source_angular_kernel_sha256"]) == 16


def test_optical_blur_kernel_digest_is_recorded(tmp_path: Path) -> None:
    path = tmp_path / "blur.npy"
    kernel = np.zeros((3, 3))
    kernel[1, 1] = 1.0
    np.save(path, kernel)
    config = load_config(
        Path(__file__).parents[1] / "examples" / "configs" / "shack_hartmann_minimal.toml"
    )
    settings = replace(config.shack_hartmann, optical_blur_kernel_path=str(path))
    frame = WavefrontSensor(replace(config, shack_hartmann=settings)).expose(
        np.zeros(config.input.shape), seed=4
    )
    assert len(frame.metadata["wfs_shack_hartmann_optical_blur_kernel_sha256"]) == 16
