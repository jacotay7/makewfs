"""Demonstrate the closed-loop injection boundary without owning a controller."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import makewfs


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
    yy, xx = np.mgrid[: sensor.config.input.shape[0], : sensor.config.input.shape[1]]
    centred_x = (xx - (sensor.config.input.shape[1] - 1) / 2) / sensor.config.input.shape[1]
    centred_y = (yy - (sensor.config.input.shape[0] - 1) / 2) / sensor.config.input.shape[0]
    residual = 0.15e-6 * (centred_x + 0.5 * centred_y)
    history: list[np.ndarray] = []
    for step in range(args.steps):
        frame = sensor.expose(residual, seed=step)
        history.append(np.asarray(frame))
        # This toy update stands in for an external reconstructor/controller;
        # makewfs only receives the resulting residual on the next iteration.
        residual = residual * 0.72

    figure, axes = plt.subplots(2, 3, figsize=(12, 7))
    for axis, frame, step in zip(axes.flat, history[:6], range(min(6, args.steps))):
        axis.imshow(frame, origin="lower")
        axis.set_title(f"external loop step {step}")
        axis.set_axis_off()
    figure.suptitle("residual OPD → WavefrontSensor.expose (controller remains external)")
    figure.tight_layout()
    figure.savefig(args.output, dpi=140)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
