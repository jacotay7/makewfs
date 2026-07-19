"""Optional CuPy parity tests for the private optical backend."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from makewfs import WavefrontSensor
from makewfs.backend import cupy_backend
from makewfs.config import load_config


def _cupy() -> object:
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CuPy is installed but no CUDA device is available")
    except Exception as exc:  # pragma: no cover - depends on local CUDA runtime
        pytest.skip(f"CUDA runtime is unavailable: {exc}")
    return cupy


def _config(name: str):
    return load_config(Path(__file__).parents[1] / "examples" / "configs" / name)


@pytest.mark.gpu
def test_shack_hartmann_gpu_matches_cpu_and_returns_host_detector_frame() -> None:
    cupy = _cupy()
    config = _config("shack_hartmann_minimal.toml")
    cpu_sensor = WavefrontSensor(config)
    gpu_sensor = WavefrontSensor(config, _backend=cupy_backend())
    opd = np.zeros(config.input.shape, dtype=np.float64)
    device_opd = cupy.asarray(opd)

    cpu_rate = cpu_sensor.photon_rate(opd)
    gpu_rate = gpu_sensor.photon_rate(device_opd)
    assert isinstance(gpu_rate, cupy.ndarray)
    np.testing.assert_allclose(cupy.asnumpy(gpu_rate), cpu_rate, rtol=5e-5, atol=5e-4)

    frame = gpu_sensor.expose(device_opd, seed=41)
    assert not isinstance(np.asarray(frame), cupy.ndarray)
    assert np.asarray(frame).shape == cpu_sensor.engine.output_shape
    assert frame.metadata["wfs_input_opd_rms_m"] == 0.0


@pytest.mark.gpu
def test_pyramid_gpu_matches_cpu_and_integrated_stack_stays_on_device_until_detector() -> None:
    cupy = _cupy()
    config = _config("pyramid_minimal.toml")
    cpu_sensor = WavefrontSensor(config)
    gpu_sensor = WavefrontSensor(config, _backend=cupy_backend())
    zero = np.zeros(config.input.shape, dtype=np.float64)
    tilt = zero.copy()
    tilt[:, config.input.shape[1] // 2 :] = 1.0e-8
    device_samples = cupy.asarray(np.stack([zero, tilt]))

    cpu_rate = cpu_sensor.photon_rate(zero)
    gpu_rate = gpu_sensor.photon_rate(cupy.asarray(zero))
    np.testing.assert_allclose(cupy.asnumpy(gpu_rate), cpu_rate, rtol=5e-5, atol=5e-4)

    frame = gpu_sensor.expose_integrated(device_samples, seed=42)
    assert np.asarray(frame).shape == cpu_sensor.engine.output_shape
    assert frame.metadata["wfs_temporal_samples"] == 2
