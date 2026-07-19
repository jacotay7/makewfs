"""Check relative performance envelopes in a representative benchmark report.

The check deliberately compares kernels measured in the same process rather than
promising portable absolute wall-clock times.  It is intended to catch an
accidental order-of-magnitude regression while remaining tolerant of shared CI
runners.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"could not read benchmark report {path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise SystemExit(f"benchmark report {path} has no results list")
    return payload


def _results(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    by_name: dict[str, dict[str, Any]] = {}
    for entry in report["results"]:
        if not isinstance(entry, dict):
            raise SystemExit("benchmark result entry is not an object")
        config = entry.get("config")
        if not isinstance(config, str):
            raise SystemExit("benchmark result is missing its config path")
        name = Path(config).name
        if name in by_name:
            raise SystemExit(f"duplicate benchmark config {name}")
        by_name[name] = entry
    return by_name


def _positive(entry: dict[str, Any], key: str, name: str) -> float:
    value = entry.get(key)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value <= 0:
        raise SystemExit(f"{name}: {key} must be a finite positive number")
    return float(value)


def _ratio(
    entries: dict[str, dict[str, Any]],
    numerator: str,
    denominator: str,
    *,
    key: str,
    limit: float,
) -> tuple[float, str]:
    missing = [name for name in (numerator, denominator) if name not in entries]
    if missing:
        raise SystemExit(f"benchmark report is missing {', '.join(missing)}")
    ratio = _positive(entries[numerator], key, numerator) / _positive(
        entries[denominator], key, denominator
    )
    if ratio > limit:
        raise SystemExit(f"{key}: {numerator}/{denominator} ratio {ratio:.3g} exceeds {limit:.3g}")
    return ratio, f"{numerator}/{denominator} {key}={ratio:.3g} (limit {limit:.3g})"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--max-sh-size-ratio", type=float, default=20.0)
    parser.add_argument("--max-source-state-ratio", type=float, default=30.0)
    parser.add_argument("--max-pyramid-modulation-ratio", type=float, default=100.0)
    args = parser.parse_args()
    if (
        min(args.max_sh_size_ratio, args.max_source_state_ratio, args.max_pyramid_modulation_ratio)
        <= 0
    ):
        parser.error("ratio limits must be positive")

    entries = _results(_load(args.report))
    checks = [
        _ratio(
            entries,
            "shack_hartmann_60x60_float64.toml",
            "shack_hartmann_20x20_float32.toml",
            key="warm_optical_frame_s",
            limit=args.max_sh_size_ratio,
        ),
        _ratio(
            entries,
            "shack_hartmann_broadband_lgs.toml",
            "shack_hartmann_minimal.toml",
            key="warm_optical_frame_s",
            limit=args.max_source_state_ratio,
        ),
        _ratio(
            entries,
            "pyramid_80_mod32_float64.toml",
            "pyramid_40_float32.toml",
            key="warm_optical_frame_s",
            limit=args.max_pyramid_modulation_ratio,
        ),
    ]
    for _, message in checks:
        print(f"OK {message}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
