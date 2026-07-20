"""Demonstrate the closed-loop injection boundary without owning a controller.

makewfs never runs a reconstructor or controller. This script stands in for an
external AO loop that measures the frame, estimates a correction, and hands the
next *residual* wavefront back to ``WavefrontSensor.expose``. The residual is a
low-order aberration (defocus + astigmatism) so the Shack-Hartmann spots shift
differently across the pupil and the relaxation toward the flat reference is
visible, and a convergence panel tracks the residual RMS.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import makewfs


def _low_order_residual(shape: tuple[int, int], amplitude_m: float) -> np.ndarray:
    """Return a defocus + astigmatism OPD normalized to ``amplitude_m`` RMS."""
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    x = (xx - (shape[1] - 1) / 2) / (shape[1] / 2)
    y = (yy - (shape[0] - 1) / 2) / (shape[0] / 2)
    defocus = x**2 + y**2
    astigmatism = x**2 - y**2
    combined = defocus + 0.6 * astigmatism
    combined -= combined.mean()
    combined /= combined.std()
    return amplitude_m * combined


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="closed_loop_injection.png")
    parser.add_argument("--steps", type=int, default=6)
    args = parser.parse_args()
    if args.steps < 2:
        raise SystemExit("--steps must be >= 2")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("install makewfs[examples] to run this example") from exc

    sensor = makewfs.WavefrontSensor.from_toml(
        Path(__file__).parent / "configs" / "shack_hartmann_minimal.toml"
    )
    shape = sensor.config.input.shape
    flat_frame = np.asarray(sensor.expose(np.zeros(shape), seed=0)).astype(np.float64)

    frames: list[np.ndarray] = []
    residual_rms_nm: list[float] = []
    for step in range(args.steps):
        residual = _low_order_residual(shape, amplitude_m=200e-9) * (0.6**step)
        residual_rms_nm.append(float(np.std(residual) * 1e9))
        frames.append(np.asarray(sensor.expose(residual, seed=0)).astype(np.float64))

    columns = min(4, args.steps)
    shown = sorted({0, 1, args.steps // 2, args.steps - 1})[:columns]
    figure, axes = plt.subplots(
        2, columns + 1, figsize=(3.3 * (columns + 1), 6.6), constrained_layout=True
    )
    # Bottom row shows each frame minus the flat-wavefront frame on a shared,
    # symmetric scale so the spot displacements shrinking to zero are the story.
    diffs = [frame - flat_frame for frame in frames]
    diff_scale = float(np.abs(diffs[0]).max()) or 1.0
    for column, step in enumerate(shown):
        opd_axis = axes[0, column]
        opd = _low_order_residual(shape, amplitude_m=200e-9) * (0.6**step)
        opd_scale = float(np.abs(_low_order_residual(shape, amplitude_m=200e-9)).max()) * 1e9
        artist = opd_axis.imshow(
            opd * 1e9, origin="lower", cmap="coolwarm", vmin=-opd_scale, vmax=opd_scale
        )
        opd_axis.set_title(f"residual OPD, step {step}\n{residual_rms_nm[step]:.0f} nm RMS")
        opd_axis.set_axis_off()
        figure.colorbar(artist, ax=opd_axis, fraction=0.046, label="OPD (nm)")

        diff_axis = axes[1, column]
        diff_artist = diff_axis.imshow(
            diffs[step], origin="lower", cmap="coolwarm", vmin=-diff_scale, vmax=diff_scale
        )
        diff_axis.set_title(f"SH frame - flat, step {step}")
        diff_axis.set_axis_off()
        figure.colorbar(diff_artist, ax=diff_axis, fraction=0.046, label="signal (ADU)")

    convergence = axes[0, columns]
    convergence.plot(range(args.steps), residual_rms_nm, "o-")
    convergence.set_title("external loop convergence")
    convergence.set_xlabel("loop step")
    convergence.set_ylabel("residual OPD RMS (nm)")
    convergence.grid(True, alpha=0.3)

    frame_rms = [float(np.sqrt(np.mean(diff**2))) for diff in diffs]
    departure = axes[1, columns]
    departure.plot(range(args.steps), frame_rms, "o-", color="tab:red")
    departure.set_title("frame departure from flat")
    departure.set_xlabel("loop step")
    departure.set_ylabel("RMS |frame - flat| (ADU)")
    departure.grid(True, alpha=0.3)

    figure.suptitle(
        "External AO loop drives the residual toward zero; makewfs only forms the frame",
        fontsize=14,
    )
    figure.savefig(args.output, dpi=140)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
