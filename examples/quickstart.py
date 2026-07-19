"""Render a small configured Shack-Hartmann image and save diagnostic plots."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import makewfs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=str(Path(__file__).parent / "configs" / "shack_hartmann_minimal.toml")
    )
    parser.add_argument("--output", default="quickstart.png")
    args = parser.parse_args()
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("install makewfs[examples] to run this example") from exc

    config = makewfs.load_config(args.config)
    sensor = makewfs.WavefrontSensor(config)
    yy, xx = np.mgrid[: config.input.shape[0], : config.input.shape[1]]
    radius = np.hypot(xx - (config.input.shape[1] - 1) / 2, yy - (config.input.shape[0] - 1) / 2)
    opd = 0.15e-6 * np.exp(-((radius / (config.input.shape[0] / 4)) ** 2))
    ideal = sensor.photon_rate(opd)
    frame = sensor.expose(opd, seed=0)

    figure, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flat
    axes[0].imshow(opd * 1e9, origin="lower")
    axes[0].set_title("input OPD (nm)")
    axes[1].imshow(ideal, origin="lower")
    axes[1].set_title("ideal rate (photons/s/pixel)")
    axes[2].imshow(np.asarray(frame), origin="lower")
    axes[2].set_title("detector frame (ADU)")
    axes[3].plot(ideal[ideal.shape[0] // 2])
    axes[3].set_title("central row (photons/s/pixel)")
    axes[3].set_xlabel("detector x (pixel)")
    axes[3].set_ylabel("photons/s")
    central_axis = axes[3]
    for axis in axes:
        if axis is not central_axis:
            axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(args.output, dpi=140)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
