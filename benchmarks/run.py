"""Measure cold construction and warm optical/detector paths."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
import tracemalloc
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

import makewfs
from makewfs.source import iter_source_states


def _measure(path: Path, frames: int, *, measure_memory: bool = False) -> dict[str, Any]:
    if measure_memory:
        tracemalloc.start()
    start = perf_counter()
    sensor = makewfs.WavefrontSensor.from_toml(path)
    construction_s = perf_counter() - start
    phase = np.zeros(sensor.config.input.shape, dtype=np.float64)
    sensor.photon_rate(phase)  # warm FFT plans and arrays
    start = perf_counter()
    for _ in range(frames):
        sensor.photon_rate(phase)
    optical_s = perf_counter() - start
    start = perf_counter()
    for frame in range(frames):
        sensor.expose(phase, seed=frame)
    detector_s = perf_counter() - start
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
        "python_peak_memory_mib": peak_memory_mib,
    }


def _installed_version(name: str) -> str | None:
    """Return an installed package version without making optional deps required."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", type=Path)
    parser.add_argument("--frames", type=int, default=3)
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
    report: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "dependencies": {
            name: _installed_version(name)
            for name in ("makewfs", "numpy", "scipy", "getframes", "pyturb")
        },
        "results": [
            _measure(path, args.frames, measure_memory=args.measure_memory) for path in configs
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
