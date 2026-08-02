"""Optional CuPy parity tests for the private optical backend."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from makewfs import WavefrontSensor
from makewfs.backend import cupy_backend
from makewfs.config import WFSConfig, load_config


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


def _gpu_config(name: str) -> WFSConfig:
    data = _config(name).to_dict()
    data["numerics"]["device"] = "gpu"
    return WFSConfig.from_dict(data)


@pytest.mark.gpu
def test_shack_hartmann_gpu_matches_cpu_and_keeps_detector_frame_on_device() -> None:
    cupy = _cupy()
    config = _config("shack_hartmann_minimal.toml")
    cpu_sensor = WavefrontSensor(config)
    gpu_sensor = WavefrontSensor(_gpu_config("shack_hartmann_minimal.toml"))
    opd = np.zeros(config.input.shape, dtype=np.float64)
    device_opd = cupy.asarray(opd)

    cpu_rate = cpu_sensor.photon_rate(opd)
    gpu_rate = gpu_sensor.photon_rate(device_opd)
    assert isinstance(gpu_rate, cupy.ndarray)
    np.testing.assert_allclose(cupy.asnumpy(gpu_rate), cpu_rate, rtol=5e-5, atol=5e-4)

    frame = gpu_sensor.expose(device_opd, seed=41)
    assert isinstance(frame.data, cupy.ndarray)
    assert isinstance(frame.truth.mean_electrons, cupy.ndarray)
    assert not isinstance(np.asarray(frame), cupy.ndarray)
    assert np.asarray(frame).shape == cpu_sensor.engine.output_shape
    assert frame.metadata["wfs_input_opd_rms_m"] == 0.0

    out = cupy.empty(gpu_sensor.engine.output_shape, dtype=cupy.uint32)
    into = gpu_sensor.expose(device_opd, seed=41, out=out)
    assert into.data is out
    assert bool(cupy.array_equal(frame.data, into.data))


@pytest.mark.gpu
def test_pyramid_gpu_matches_cpu_and_integrated_stack_stays_on_device_until_detector() -> None:
    cupy = _cupy()
    config = _config("pyramid_minimal.toml")
    cpu_sensor = WavefrontSensor(config)
    gpu_sensor = WavefrontSensor(_gpu_config("pyramid_minimal.toml"))
    zero = np.zeros(config.input.shape, dtype=np.float64)
    tilt = zero.copy()
    tilt[:, config.input.shape[1] // 2 :] = 1.0e-8
    device_samples = cupy.asarray(np.stack([zero, tilt]))

    cpu_rate = cpu_sensor.photon_rate(zero)
    gpu_rate = gpu_sensor.photon_rate(cupy.asarray(zero))
    np.testing.assert_allclose(cupy.asnumpy(gpu_rate), cpu_rate, rtol=5e-5, atol=5e-4)

    out = cupy.empty(gpu_sensor.engine.output_shape, dtype=cupy.uint32)
    frame = gpu_sensor.expose_integrated(device_samples, seed=42, out=out)
    assert frame.data is out
    assert isinstance(frame.data, cupy.ndarray)
    assert np.asarray(frame).shape == cpu_sensor.engine.output_shape
    assert frame.metadata["wfs_temporal_samples"] == 2


@pytest.mark.gpu
def test_private_backend_hook_also_selects_gpu_detector_for_compatibility() -> None:
    cupy = _cupy()
    config = _config("shack_hartmann_minimal.toml")
    sensor = WavefrontSensor(config, _backend=cupy_backend())

    frame = sensor.expose(cupy.zeros(config.input.shape), seed=43)

    assert isinstance(frame.data, cupy.ndarray)
    assert frame.metadata["wfs_device"] == "gpu"


@pytest.mark.gpu
def test_pyturb_gpu_opd_flows_to_shack_hartmann_adu_without_host_copy() -> None:
    cupy = _cupy()
    pyturb = pytest.importorskip("pyturb")
    config = _gpu_config("shack_hartmann_minimal.toml")
    atmosphere = pyturb.Atmosphere.from_profile(
        "paranal-median",
        seeing=0.8,
        diameter=config.telescope.pupil_diameter_m,
        n=config.input.shape[0],
        seed=5,
        device="gpu",
    )
    sensor = WavefrontSensor(config)

    opd_m = atmosphere.opd()
    frame = sensor.expose(opd_m, seed=44)

    assert isinstance(opd_m, cupy.ndarray)
    assert isinstance(frame.data, cupy.ndarray)
    assert isinstance(frame.truth.photon_rate, cupy.ndarray)


@pytest.mark.gpu
def test_broadband_sh_gpu_state_batch_matches_sequential_execution() -> None:
    cupy = _cupy()
    data = load_config(
        Path(__file__).parents[1] / "benchmarks" / "configs" / "shack_hartmann_broadband_lgs.toml"
    ).to_dict()
    data["numerics"]["device"] = "gpu"
    config = WFSConfig.from_dict(data)
    sensor = WavefrontSensor(config)
    engine = sensor.engine
    assert any(len(group) > 1 for group in engine._state_groups)
    rng = cupy.random.RandomState(17)
    opd = rng.normal(0.0, 8.0e-8, config.input.shape)

    grouped = engine.render(opd)
    engine._state_groups = tuple((index,) for index in range(len(engine.source_states)))
    sequential = engine.render(opd)

    np.testing.assert_allclose(
        cupy.asnumpy(grouped.photon_rate),
        cupy.asnumpy(sequential.photon_rate),
        rtol=5e-6,
        atol=5e-5,
    )
    assert grouped.spectral_photon_rate is not None
    assert sequential.spectral_photon_rate is not None
    np.testing.assert_allclose(
        cupy.asnumpy(grouped.spectral_photon_rate),
        cupy.asnumpy(sequential.spectral_photon_rate),
        rtol=5e-6,
        atol=5e-5,
    )
