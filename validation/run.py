"""Produce deterministic optical validation metrics and optional plots."""

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


def _metrics(config_path: Path) -> dict[str, Any]:
    config = makewfs.load_config(config_path)
    sensor = makewfs.WavefrontSensor(config)
    zero = np.zeros(config.input.shape, dtype=np.float64)
    start = perf_counter()
    reference = sensor.photon_rate(zero)
    render_s = perf_counter() - start
    piston = sensor.photon_rate(zero + 2.0e-6)
    result: dict[str, Any] = {
        "config": str(config_path),
        "config_digest": config.digest,
        "sensor": config.sensor.kind,
        "shape": list(reference.shape),
        "source_rate_per_s": sensor.engine.source_rate,
        "reference_rate_sum_per_s": float(reference.sum()),
        "piston_relative_error": float(
            np.max(np.abs(reference - piston)) / max(float(np.max(reference)), 1e-30)
        ),
        "warm_render_s": render_s,
    }
    if config.sensor.kind == "pyramid":
        half = reference.shape[0] // 2
        quadrants = [
            reference[:half, :half],
            reference[:half, half:],
            reference[half:, :half],
            reference[half:, half:],
        ]
        result["pyramid_quadrant_relative_spread"] = float(
            np.std([float(quadrant.sum()) for quadrant in quadrants])
            / max(float(reference.sum()), 1e-30)
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        action="append",
        type=Path,
        help="configuration to validate (repeatable; defaults to shipped SH and pyramid files)",
    )
    parser.add_argument("--output", type=Path, default=Path("validation-metrics.json"))
    parser.add_argument("--plot", type=Path, default=None)
    args = parser.parse_args()
    root = Path(__file__).parents[1]
    configs = args.config or [
        root / "examples" / "configs" / "shack_hartmann_minimal.toml",
        root / "examples" / "configs" / "pyramid_minimal.toml",
    ]
    report: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "metrics": [_metrics(path) for path in configs],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.plot is not None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise SystemExit("--plot requires matplotlib") from exc
        figure, axes = plt.subplots(1, len(configs), figsize=(5 * len(configs), 4))
        axes = np.atleast_1d(axes)
        for axis, path in zip(axes, configs):
            sensor = makewfs.WavefrontSensor.from_toml(path)
            image = sensor.photon_rate(np.zeros(sensor.config.input.shape))
            axis.imshow(image, origin="lower")
            axis.set_title(path.stem)
            axis.set_axis_off()
        figure.tight_layout()
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.plot, dpi=140)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
