"""Measure cold construction and warm optical/detector paths."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shlex
import subprocess
import sys
import tracemalloc
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

import makewfs
from makewfs.source import iter_source_states


def _sync(sensor: makewfs.WavefrontSensor) -> None:
    """Wait for queued GPU work so wall-clock timings are honest."""
    if not sensor.backend.is_cpu:
        sensor.backend.xp.cuda.Stream.null.synchronize()


def _measure(
    path: Path, frames: int, *, device: str = "cpu", measure_memory: bool = False
) -> dict[str, Any]:
    if measure_memory:
        tracemalloc.start()
    start = perf_counter()
    config = makewfs.load_config(path)
    config = replace(config, numerics=replace(config.numerics, device=device))
    sensor = makewfs.WavefrontSensor(config)
    _sync(sensor)
    construction_s = perf_counter() - start
    phase = sensor.backend.zeros(sensor.config.input.shape, dtype=np.float64)
    sensor.photon_rate(phase)  # warm FFT plans and arrays
    sensor.expose(phase, seed=0)  # warm detector RNG/kernels and end-to-end dispatch
    _sync(sensor)
    start = perf_counter()
    for _ in range(frames):
        sensor.photon_rate(phase)
    _sync(sensor)
    optical_s = perf_counter() - start
    start = perf_counter()
    for frame in range(frames):
        sensor.expose(phase, seed=frame)
    _sync(sensor)
    detector_s = perf_counter() - start
    rate = sensor.photon_rate(phase)
    _sync(sensor)
    start = perf_counter()
    for frame in range(frames):
        sensor.detector.expose(rate, metadata={}, seed=frame)
    _sync(sensor)
    detector_only_s = perf_counter() - start
    peak_memory_mib: float | None = None
    if measure_memory:
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_memory_mib = peak_bytes / (1024.0**2)
    source_wavelengths = sensor.config.source.wavelengths_m
    source_ranges = sensor.config.source.lgs_ranges_m
    pyramid = sensor.config.pyramid
    return {
        "config": str(path),
        "sensor": sensor.config.sensor.kind,
        "device": device,
        "shape": list(sensor.engine.output_shape),
        "source_states": len(iter_source_states(sensor.config)),
        "wavelength_samples": len(source_wavelengths) or 1,
        "range_samples": len(source_ranges) or 1,
        "modulation_samples": pyramid.modulation_samples if pyramid is not None else 1,
        "modulation_radius_lambda_over_d": (
            pyramid.modulation_radius_lambda_over_d if pyramid is not None else 0.0
        ),
        "dtype": sensor.config.numerics.dtype,
        "fft_workers": sensor.config.numerics.fft_workers,
        "frames": frames,
        "construction_s": construction_s,
        "warm_optical_total_s": optical_s,
        "warm_optical_frame_s": optical_s / frames,
        "warm_detector_total_s": detector_s,
        "warm_detector_frame_s": detector_s / frames,
        "warm_optical_frames_per_s": frames / optical_s,
        "warm_detector_frames_per_s": frames / detector_s,
        "warm_detector_only_frame_s": detector_only_s / frames,
        "warm_detector_only_frames_per_s": frames / detector_only_s,
        "python_peak_memory_mib": peak_memory_mib,
    }


def _installed_version(name: str) -> str | None:
    """Return an installed package version without making optional deps required."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_state(root: Path) -> tuple[str | None, bool | None]:
    """Return the source revision and whether the checkout has local changes."""
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return revision, dirty


def _gpu_name() -> str | None:
    """Return the first NVIDIA device name without requiring CUDA on CPU hosts."""
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return output.splitlines()[0].strip() if output.splitlines() else None


def _cpu_model() -> str:
    """Return a useful CPU model string on Linux, with a portable fallback."""
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", type=Path)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--device", choices=("cpu", "gpu", "both"), default="cpu")
    parser.add_argument(
        "--measure-memory",
        action="store_true",
        help="record Python-level tracemalloc peak memory (slower; C buffers are not included)",
    )
    parser.add_argument(
        "--representative",
        action="store_true",
        help="also measure the 20x20 float32 and 60x60 float64 SH benchmark configs",
    )
    parser.add_argument("--output", type=Path, default=Path("benchmark-results.json"))
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be >= 1")
    root = Path(__file__).parents[1]
    configs = args.config or [
        root / "examples" / "configs" / "shack_hartmann_minimal.toml",
        root / "examples" / "configs" / "pyramid_minimal.toml",
    ]
    if args.representative:
        configs.extend(
            [
                root / "benchmarks" / "configs" / "shack_hartmann_20x20_float32.toml",
                root / "benchmarks" / "configs" / "shack_hartmann_60x60_float64.toml",
                root / "benchmarks" / "configs" / "shack_hartmann_broadband_lgs.toml",
                root / "benchmarks" / "configs" / "pyramid_40_float32.toml",
                root / "benchmarks" / "configs" / "pyramid_60_mod8_float32.toml",
                root / "benchmarks" / "configs" / "pyramid_80_mod32_float64.toml",
            ]
        )
    devices = ("cpu", "gpu") if args.device == "both" else (args.device,)
    revision, source_dirty = _git_state(root)
    report: dict[str, Any] = {
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "revision": revision,
        "source_dirty": source_dirty,
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu": _cpu_model(),
        "gpu": _gpu_name() if "gpu" in devices else None,
        "dependencies": {
            name: _installed_version(name)
            for name in ("makewfs", "numpy", "scipy", "cupy", "getframes", "pyturb")
        },
        "methodology": {
            "frames_per_cell": args.frames,
            "persistent_sensor": True,
            "warmup_frames": 1,
            "device_resident_opd_rate_and_output": True,
            "include_detector_truth": True,
            "rng": "a distinct deterministic per-frame seed",
            "cuda_synchronization": "before and after each timed region",
            "construction_included_in_throughput": False,
            "host_transfers_included": False,
        },
        "results": [
            _measure(
                path,
                args.frames,
                device=device,
                measure_memory=args.measure_memory,
            )
            for path in configs
            for device in devices
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
