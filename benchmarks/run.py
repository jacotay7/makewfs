"""Measure cold construction and warm optical/detector paths."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

import makewfs
from makewfs.source import iter_source_states


def _measure(path: Path, frames: int) -> dict[str, Any]:
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
    return {
        "config": str(path),
        "sensor": sensor.config.sensor.kind,
        "shape": list(sensor.engine.output_shape),
        "source_states": len(iter_source_states(sensor.config)),
        "frames": frames,
        "construction_s": construction_s,
        "warm_optical_total_s": optical_s,
        "warm_optical_frame_s": optical_s / frames,
        "warm_detector_total_s": detector_s,
        "warm_detector_frame_s": detector_s / frames,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", type=Path)
    parser.add_argument("--frames", type=int, default=3)
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
            ]
        )
    report: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "results": [_measure(path, args.frames) for path in configs],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
