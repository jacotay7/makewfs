"""Compare SH and pyramid ideal images for one injected OPD."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import makewfs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="compare_sensors.png")
    args = parser.parse_args()
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("install makewfs[examples] to run this example") from exc

    root = Path(__file__).parent
    sh = makewfs.WavefrontSensor.from_toml(root / "configs" / "shack_hartmann_minimal.toml")
    pyramid = makewfs.WavefrontSensor.from_toml(root / "configs" / "pyramid_minimal.toml")
    yy, xx = np.mgrid[: sh.config.input.shape[0], : sh.config.input.shape[1]]
    radius = np.hypot(xx - 63.5, yy - 63.5)
    opd = 0.12e-6 * np.exp(-((radius / 30.0) ** 2)) + 0.03e-6 * (xx - 63.5) / 64.0
    sh_rate = sh.photon_rate(opd)
    pyramid_rate = pyramid.photon_rate(opd)

    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(opd * 1e9, origin="lower")
    axes[0].set_title("injected OPD (nm)")
    axes[1].imshow(sh_rate, origin="lower")
    axes[1].set_title("Shack-Hartmann rate")
    axes[2].imshow(pyramid_rate, origin="lower")
    axes[2].set_title("pyramid rate")
    for axis in axes:
        axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(args.output, dpi=140)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
