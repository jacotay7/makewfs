"""Write cProfile summaries for warm optical paths."""

from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import sys
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np

import makewfs


def _profile(path: Path, frames: int) -> dict[str, Any]:
    sensor = makewfs.WavefrontSensor.from_toml(path)
    phase = np.zeros(sensor.config.input.shape, dtype=np.float64)
    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(frames):
        sensor.photon_rate(phase)
    profiler.disable()
    stream = StringIO()
    pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(25)
    return {
        "config": str(path),
        "sensor": sensor.config.sensor.kind,
        "frames": frames,
        "profile": stream.getvalue(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile cached warm optical paths.")
    parser.add_argument("--config", action="append", type=Path)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("profile-results.json"))
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be >= 1")
    root = Path(__file__).parents[1]
    configs = args.config or [
        root / "examples" / "configs" / "shack_hartmann_minimal.toml",
        root / "examples" / "configs" / "pyramid_minimal.toml",
    ]
    report = {"python": sys.version, "results": [_profile(path, args.frames) for path in configs]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
