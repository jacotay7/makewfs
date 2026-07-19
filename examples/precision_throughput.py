"""Compare optical precision and warm throughput for a representative SH case."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np

import makewfs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="precision_throughput.png")
    parser.add_argument("--frames", type=int, default=5)
    args = parser.parse_args()
    if args.frames < 1:
        raise SystemExit("--frames must be >= 1")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("install makewfs[examples] to run this example") from exc

    config = makewfs.load_config(Path(__file__).parent / "configs" / "precision_throughput.toml")
    phase = np.zeros(config.input.shape, dtype=np.float64)
    results: dict[str, tuple[float, np.ndarray]] = {}
    for precision in ("float32", "float64"):
        sensor = makewfs.WavefrontSensor(
            replace(config, numerics=replace(config.numerics, dtype=precision))
        )
        reference = sensor.photon_rate(phase)
        start = perf_counter()
        for _ in range(args.frames):
            sensor.photon_rate(phase)
        results[precision] = ((perf_counter() - start) / args.frames, reference)
    reference64 = results["float64"][1]
    error = np.max(np.abs(results["float32"][1] - reference64)) / max(
        float(np.max(reference64)), 1e-30
    )
    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(list(results), [value[0] * 1e3 for value in results.values()])
    axes[0].set_ylabel("warm optical latency (ms/frame)")
    axes[0].set_title(f"{args.frames}-frame average")
    axes[1].bar(["float32 vs float64"], [error])
    axes[1].set_ylabel("max relative image error")
    axes[1].set_title("precision agreement")
    figure.suptitle("CPU precision and throughput")
    figure.tight_layout()
    figure.savefig(args.output, dpi=140)
    print(f"wrote {args.output}; float32 relative error={error:.3e}")


if __name__ == "__main__":
    main()
