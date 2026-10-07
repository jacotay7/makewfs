"""Provenance and referenced-file hashing tests."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from makewfs import WavefrontSensor, load_config

CONFIGS = Path(__file__).parents[1] / "examples" / "configs"


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


@pytest.mark.parametrize("name", ["shack_hartmann_minimal.toml", "pyramid_minimal.toml"])
def test_phase_input_reports_its_opd_in_metres(name: str) -> None:
    # Phase input is converted at the reference wavelength before anything
    # downstream sees it, including the input-RMS metadata and OpticalResult.
    opd_sensor = WavefrontSensor.from_toml(CONFIGS / name)
    wavelength = 1.0e-6
    phase_input = replace(
        opd_sensor.config.input,
        quantity="phase",
        unit="rad",
        reference_wavelength_m=wavelength,
    )
    phase_sensor = WavefrontSensor(replace(opd_sensor.config, input=phase_input))
    shape = opd_sensor.config.input.shape
    _, xx = np.indices(shape, dtype=np.float64)
    opd = 30e-9 * (xx - (shape[1] - 1) / 2) / shape[1] + 10e-9
    phase = 2.0 * np.pi * opd / wavelength

    expected = float(np.sqrt(np.mean(opd**2)))
    from_opd = opd_sensor.expose(opd, seed=1).metadata["wfs_input_opd_rms_m"]
    from_phase = phase_sensor.expose(phase, seed=1).metadata["wfs_input_opd_rms_m"]
    assert from_opd == pytest.approx(expected, rel=1e-12)
    assert from_phase == pytest.approx(expected, rel=1e-12)
    np.testing.assert_allclose(phase_sensor._render(phase).opd_m, opd, rtol=1e-12)
    np.testing.assert_allclose(
        phase_sensor.photon_rate(phase), opd_sensor.photon_rate(opd), rtol=1e-6, atol=0.0
    )
