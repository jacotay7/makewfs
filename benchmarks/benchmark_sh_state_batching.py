#!/usr/bin/env python3
"""Compare grouped and sequential GPU Shack--Hartmann source-state execution."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from makewfs import WavefrontSensor


def _distribution(samples_ms: list[float]) -> dict[str, float | int]:
    p50, p95, p99 = np.percentile(samples_ms, (50, 95, 99))
    return {
        "count": len(samples_ms),
        "p50_ms": float(p50),
        "p95_ms": float(p95),
        "p99_ms": float(p99),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--temporal-samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.frames < 1 or arguments.temporal_samples < 1:
        parser.error("--frames and --temporal-samples must be positive")

    try:
        import cupy as cp
    except ImportError as exc:
        parser.error(f"CuPy is required: {exc}")
    if cp.cuda.runtime.getDeviceCount() < 1:
        parser.error("a CUDA device is required")

    loaded = WavefrontSensor.from_toml(arguments.config)
    config = replace(
        loaded.config,
        numerics=replace(loaded.config.numerics, device="gpu"),
    )
    sensor = WavefrontSensor(config)
    engine = sensor.engine
    grouped = engine._state_groups
    sequential = tuple((index,) for index in range(len(engine.source_states)))
    if not any(len(group) > 1 for group in grouped):
        parser.error("the configuration has no source states with shared FFT geometry")

    rng = cp.random.RandomState(arguments.seed)
    samples = rng.normal(
        0.0,
        2.0e-7,
        (arguments.temporal_samples, *config.input.shape),
    ).astype(cp.float64)
    outputs: dict[str, Any] = {}
    for name, groups in (("grouped", grouped), ("sequential", sequential)):
        engine._state_groups = groups
        outputs[name] = engine.render(samples[0]).photon_rate.copy()
        sensor.expose_integrated(samples, seed=arguments.seed)
        cp.cuda.get_current_stream().synchronize()

    difference = cp.abs(outputs["grouped"] - outputs["sequential"])
    maximum_absolute_error = float(cp.max(difference).item())
    reference_scale = float(cp.max(cp.abs(outputs["sequential"])).item())

    timings: dict[str, dict[str, list[float]]] = {
        "grouped": {"optical_ms": [], "total_ms": []},
        "sequential": {"optical_ms": [], "total_ms": []},
    }
    started = time.perf_counter()
    for frame_index in range(arguments.frames):
        order = (
            (("grouped", grouped), ("sequential", sequential))
            if frame_index % 2 == 0
            else (("sequential", sequential), ("grouped", grouped))
        )
        for name, groups in order:
            engine._state_groups = groups
            frame = sensor.expose_integrated(
                samples,
                seed=arguments.seed + 1 + frame_index,
            )
            cp.cuda.get_current_stream().synchronize()
            timings[name]["optical_ms"].append(float(frame.metadata["wfs_optical_render_s"] * 1e3))
            timings[name]["total_ms"].append(float(frame.metadata["wfs_total_expose_s"] * 1e3))
    elapsed_s = time.perf_counter() - started
    results = {
        name: {metric: _distribution(values) for metric, values in measurements.items()}
        for name, measurements in timings.items()
    }
    grouped_optical = results["grouped"]["optical_ms"]["p50_ms"]
    sequential_optical = results["sequential"]["optical_ms"]["p50_ms"]
    assert isinstance(grouped_optical, float)
    assert isinstance(sequential_optical, float)
    payload = {
        "schema_version": 1,
        "config": str(arguments.config.resolve()),
        "hardware": {
            "platform": platform.platform(),
            "python": sys.version,
            "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
            "cupy": cp.__version__,
        },
        "method": {
            "frames_per_variant": arguments.frames,
            "temporal_samples": arguments.temporal_samples,
            "seed": arguments.seed,
            "alternating_variant_order": True,
            "cuda_synchronized_after_each_frame": True,
            "elapsed_s": elapsed_s,
        },
        "source_states": len(engine.source_states),
        "grouped_state_indices": [list(group) for group in grouped],
        "sequential_state_indices": [list(group) for group in sequential],
        "equivalence": {
            "maximum_absolute_photon_rate_error": maximum_absolute_error,
            "maximum_relative_photon_rate_error": (
                maximum_absolute_error / reference_scale if reference_scale else 0.0
            ),
        },
        "results": results,
        "optical_p50_reduction_fraction": (1.0 - grouped_optical / sequential_optical),
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
