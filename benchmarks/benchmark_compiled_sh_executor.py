#!/usr/bin/env python3
"""Compare the first-use-JIT and array-reference Shack--Hartmann GPU paths."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
from contextlib import nullcontext
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from makewfs import WavefrontSensor, load_config


def _distribution(samples_ms: list[float]) -> dict[str, float | int]:
    p50, p95, p99 = np.percentile(samples_ms, (50, 95, 99))
    return {
        "count": len(samples_ms),
        "p50_ms": float(p50),
        "p95_ms": float(p95),
        "p99_ms": float(p99),
    }


def _haka_sensor(pupil_path: Path) -> WavefrontSensor:
    from examples.keck_haka.simulate import (
        HAKA_REFERENCE_AIRMASS,
        HAKA_SOURCE_TEMPERATURE_K,
        configured_sensor,
        load_camera_modes,
        make_keck_pupil,
        pupil_collecting_area_m2,
        select_camera_mode,
    )

    root = Path(__file__).parents[1]
    pupil = make_keck_pupil()
    np.save(pupil_path, pupil)
    base = load_config(root / "examples" / "keck_haka" / "keck_haka.toml")
    base = replace(base, numerics=replace(base.numerics, device="gpu"))
    return configured_sensor(
        base,
        magnitude=10.16,
        mode=select_camera_mode(10.16, load_camera_modes()),
        frame_rate_column="WSFRRT1",
        pupil_path=pupil_path,
        collecting_area_m2=pupil_collecting_area_m2(pupil),
        source_temperature_k=HAKA_SOURCE_TEMPERATURE_K,
        airmass=HAKA_REFERENCE_AIRMASS,
    )


def _generic_sensor(config_path: Path) -> WavefrontSensor:
    config = load_config(config_path)
    return WavefrontSensor(replace(config, numerics=replace(config.numerics, device="gpu")))


def _relative_rms(actual: Any, reference: Any, cp: Any) -> float:
    difference = actual - reference
    scale = cp.sqrt(cp.mean(reference * reference))
    if float(scale.item()) == 0.0:
        return float(cp.sqrt(cp.mean(difference * difference)).item())
    return float((cp.sqrt(cp.mean(difference * difference)) / scale).item())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path)
    source.add_argument(
        "--haka",
        action="store_true",
        help="use the physically complete eight-wavelength Keck HAKA example",
    )
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--temporal-samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=91)
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

    temporary = tempfile.TemporaryDirectory(prefix="makewfs-jit-") if arguments.haka else None
    context = temporary if temporary is not None else nullcontext()
    with context:
        sensor = (
            _haka_sensor(Path(temporary.name) / "keck-pupil.npy")
            if temporary is not None
            else _generic_sensor(arguments.config)
        )
        engine = sensor.engine
        rng = cp.random.RandomState(arguments.seed)
        samples = rng.normal(
            0.0,
            1.5e-7,
            (arguments.temporal_samples, *sensor.config.input.shape),
        ).astype(cp.float32)
        wavefronts = [samples[index] for index in range(arguments.temporal_samples)]

        started = perf_counter()
        first_result = engine.render_integrated(wavefronts)
        cp.cuda.get_current_stream().synchronize()
        first_use_ms = (perf_counter() - started) * 1e3
        if arguments.temporal_samples not in engine._compiled_executors:
            reason = engine._compiled_executor_rejections.get(
                arguments.temporal_samples, "unknown rejection"
            )
            parser.error(f"configuration does not support the compiled executor: {reason}")
        signature = engine._compiled_executors[arguments.temporal_samples].signature

        engine._compiled_executor_enabled = False
        reference = engine.render_integrated(wavefronts)
        cp.cuda.get_current_stream().synchronize()
        engine._compiled_executor_enabled = True
        compiled = engine.render_integrated(wavefronts)
        cp.cuda.get_current_stream().synchronize()

        spectral_reference = reference.spectral_photon_rate
        spectral_compiled = compiled.spectral_photon_rate
        assert spectral_reference is not None and spectral_compiled is not None
        equivalence = {
            "photon_rate_relative_rms": _relative_rms(
                compiled.photon_rate, reference.photon_rate, cp
            ),
            "spectral_rate_relative_rms": _relative_rms(spectral_compiled, spectral_reference, cp),
            "captured_rate_relative_error": abs(
                float(compiled.captured_rate_per_s) - float(reference.captured_rate_per_s)
            )
            / abs(float(reference.captured_rate_per_s)),
        }
        if max(equivalence.values()) > 5e-6:
            raise RuntimeError(f"compiled/reference parity failed: {equivalence}")

        timings: dict[str, list[float]] = {"compiled": [], "reference": []}
        for frame_index in range(arguments.frames):
            order = (
                ("compiled", "reference")
                if frame_index % 2 == 0
                else (
                    "reference",
                    "compiled",
                )
            )
            for name in order:
                engine._compiled_executor_enabled = name == "compiled"
                started = perf_counter()
                engine.render_integrated(wavefronts)
                cp.cuda.get_current_stream().synchronize()
                timings[name].append((perf_counter() - started) * 1e3)

        results = {name: _distribution(values) for name, values in timings.items()}
        compiled_p50 = results["compiled"]["p50_ms"]
        reference_p50 = results["reference"]["p50_ms"]
        assert isinstance(compiled_p50, float) and isinstance(reference_p50, float)
        properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
        payload = {
            "schema_version": 1,
            "configuration": {
                "source": "keck_haka_physically_complete"
                if arguments.haka
                else str(arguments.config.resolve()),
                "source_states": len(engine.source_states),
                "signature": asdict(signature),
            },
            "hardware": {
                "platform": platform.platform(),
                "python": sys.version,
                "gpu": properties["name"].decode(),
                "cupy": cp.__version__,
            },
            "method": {
                "frames_per_variant": arguments.frames,
                "temporal_samples": arguments.temporal_samples,
                "seed": arguments.seed,
                "alternating_variant_order": True,
                "cuda_synchronized_after_each_render": True,
                "cupy_cache_dir": os.environ.get("CUPY_CACHE_DIR"),
                "first_use_includes_plan_build_and_raw_kernel_first_launch": True,
            },
            "first_use_ms": first_use_ms,
            "equivalence": equivalence,
            "results": results,
            "steady_state_speedup": reference_p50 / compiled_p50,
            "steady_state_reduction_fraction": 1.0 - compiled_p50 / reference_p50,
            "first_result_captured_rate_per_s": float(first_result.captured_rate_per_s),
        }

    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
