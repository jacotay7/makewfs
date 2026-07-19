"""Render a benchmark JSON report as a portable Markdown table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _format_seconds(value: Any) -> str:
    if not isinstance(value, (int, float)):
        raise ValueError("benchmark timing is not numeric")
    return f"{float(value) * 1e3:.3f}"


def render(report: dict[str, Any], *, source: str) -> str:
    results = report.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError("benchmark report has no results")
    dependencies = report.get("dependencies", {})
    if not isinstance(dependencies, dict):
        raise ValueError("benchmark report dependencies must be an object")
    dependency_text = ", ".join(
        f"{name} {version or 'not installed'}" for name, version in dependencies.items()
    )
    lines = [
        "# Reference benchmark snapshot",
        "",
        f"Generated from `{source}`. Timings are local evidence, not CI promises.",
        "",
        f"- Python: `{report.get('python', 'unknown')}`",
        f"- Platform: `{report.get('platform', 'unknown')}`",
        f"- Dependencies: {dependency_text}",
        "",
        "| Configuration | Sensor | Shape | Dtype | States | Construction (ms) | "
        "Wavelengths | Ranges | Modulation | Warm optics (ms/frame) | "
        "Warm detector (ms/frame) | Optics (frames/s) | Detector (frames/s) | "
        "Python peak (MiB) |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | ---: |",
    ]
    for item in results:
        if not isinstance(item, dict):
            raise ValueError("benchmark result must be an object")
        config = Path(str(item["config"])).name
        shape = "x".join(str(value) for value in item["shape"])
        lines.append(
            (
                "| {config} | {sensor} | {shape} | {dtype} | {states} | {construction} | "
                "{wavelengths} | {ranges} | {modulation} @ {radius} λ/D | {optics} | "
                "{detector} | {optics_fps:.1f} | {detector_fps:.1f} | {memory} |"
            ).format(
                config=config,
                sensor=item["sensor"],
                shape=shape,
                dtype=item.get("dtype", "unknown"),
                states=item["source_states"],
                construction=_format_seconds(item["construction_s"]),
                wavelengths=item.get("wavelength_samples", "?"),
                ranges=item.get("range_samples", "?"),
                modulation=item.get("modulation_samples", "?"),
                radius=float(item.get("modulation_radius_lambda_over_d", 0.0)),
                optics=_format_seconds(item["warm_optical_frame_s"]),
                detector=_format_seconds(item["warm_detector_frame_s"]),
                optics_fps=float(item["warm_optical_frames_per_s"]),
                detector_fps=float(item["warm_detector_frames_per_s"]),
                memory=(
                    f"{float(item['python_peak_memory_mib']):.1f}"
                    if item.get("python_peak_memory_mib") is not None
                    else "not measured"
                ),
            )
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    output = render(report, source=str(args.report))
    if args.output is None:
        print(output, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
