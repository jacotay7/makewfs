"""Show what float32 optics buys in speed and costs in accuracy.

The same representative Shack-Hartmann case is rendered in float32 and float64.
The panels make both claims concrete: the difference image shows the float32
error is a tiny fraction of the signal, and the timing bars show the warm
per-frame speedup. Use this to decide whether float32 optics is acceptable for
a given closed-loop budget.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np

import makewfs


def _warm_latency(sensor: makewfs.WavefrontSensor, phase: np.ndarray, frames: int) -> float:
    sensor.photon_rate(phase)  # warm caches
    start = perf_counter()
    for _ in range(frames):
        sensor.photon_rate(phase)
    return (perf_counter() - start) / frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="precision_throughput.png")
    parser.add_argument("--frames", type=int, default=20)
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
    yy, xx = np.mgrid[: config.input.shape[0], : config.input.shape[1]]
    radius = np.hypot(xx - (config.input.shape[1] - 1) / 2, yy - (config.input.shape[0] - 1) / 2)
    phase = 0.1e-6 * np.exp(-((radius / (config.input.shape[0] / 4)) ** 2))

    images: dict[str, np.ndarray] = {}
    latency_ms: dict[str, float] = {}
    for precision in ("float32", "float64"):
        sensor = makewfs.WavefrontSensor(
            replace(config, numerics=replace(config.numerics, dtype=precision))
        )
        images[precision] = np.asarray(sensor.photon_rate(phase), dtype=np.float64)
        latency_ms[precision] = _warm_latency(sensor, phase, args.frames) * 1e3

    reference = images["float64"]
    difference = images["float32"] - reference
    peak = float(np.max(reference)) or 1.0
    rel_error = float(np.max(np.abs(difference))) / peak
    speedup = latency_ms["float64"] / latency_ms["float32"]

    figure, axes = plt.subplots(2, 2, figsize=(12.5, 9.5), constrained_layout=True)
    ref_artist = axes[0, 0].imshow(reference, origin="lower")
    axes[0, 0].set_title("float64 ideal image")
    figure.colorbar(ref_artist, ax=axes[0, 0], fraction=0.046, label="photons/s/pixel")

    diff_scale = float(np.max(np.abs(difference))) or 1.0
    diff_artist = axes[0, 1].imshow(
        difference, origin="lower", cmap="coolwarm", vmin=-diff_scale, vmax=diff_scale
    )
    axes[0, 1].set_title(f"float32 - float64\n(peak error {rel_error:.1e} of max signal)")
    figure.colorbar(diff_artist, ax=axes[0, 1], fraction=0.046, label="photons/s/pixel")
    for axis in (axes[0, 0], axes[0, 1]):
        axis.set_xlabel("detector x (pixel)")
        axis.set_ylabel("detector y (pixel)")

    order = ["float32", "float64"]
    bars = axes[1, 0].bar(order, [latency_ms[p] for p in order], color=["tab:green", "tab:blue"])
    axes[1, 0].set_ylabel("warm optical latency (ms/frame)")
    axes[1, 0].set_title(f"{args.frames}-frame average: float32 is {speedup:.2f}x faster")
    axes[1, 0].bar_label(bars, fmt="%.2f")

    axes[1, 1].axis("off")
    summary = (
        f"Configuration\n"
        f"  lenslets: {config.shack_hartmann.lenslets_across_pupil}"
        f" x {config.shack_hartmann.lenslets_across_pupil}\n"
        f"  input grid: {config.input.shape[0]} x {config.input.shape[1]}\n"
        f"  output frame: {reference.shape[0]} x {reference.shape[1]}\n\n"
        f"Result\n"
        f"  float32 warm latency: {latency_ms['float32']:.3f} ms/frame\n"
        f"  float64 warm latency: {latency_ms['float64']:.3f} ms/frame\n"
        f"  speedup: {speedup:.2f}x\n"
        f"  peak relative image error: {rel_error:.2e}\n\n"
        f"(single-host timing, not a CI promise)"
    )
    axes[1, 1].text(
        0.0,
        0.95,
        summary,
        va="top",
        ha="left",
        family="monospace",
        fontsize=11,
        transform=axes[1, 1].transAxes,
    )

    figure.suptitle("float32 vs float64 optics: accuracy cost and throughput gain", fontsize=14)
    figure.savefig(args.output, dpi=140)
    print(f"wrote {args.output}; float32 relative error={rel_error:.3e}, speedup={speedup:.2f}x")


if __name__ == "__main__":
    main()
