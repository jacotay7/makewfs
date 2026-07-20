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
        import matplotlib

        matplotlib.use("Agg")
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

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    panels = (
        (axes[0], opd * 1e9, "injected OPD", "OPD (nm)", "coolwarm"),
        (
            axes[1],
            sh_rate,
            "Shack-Hartmann ideal image",
            "photon rate (photons/s/pixel)",
            "viridis",
        ),
        (axes[2], pyramid_rate, "pyramid ideal image", "photon rate (photons/s/pixel)", "viridis"),
    )
    for axis, image, title, label, cmap in panels:
        artist = axis.imshow(image, origin="lower", cmap=cmap)
        axis.set_title(title)
        axis.set_xlabel("detector x (pixel)")
        axis.set_ylabel("detector y (pixel)")
        figure.colorbar(artist, ax=axis, fraction=0.046, pad=0.04, label=label)
    figure.suptitle("Same wavefront through two sensors (ideal photon rate, before detector)")
    figure.savefig(args.output, dpi=140)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
