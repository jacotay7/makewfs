"""Compare NGS magnitudes while retaining one detector configuration."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

import makewfs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="magnitude_series.png")
    parser.add_argument("--magnitudes", nargs="+", type=float, default=[5.0, 8.0, 11.0])
    args = parser.parse_args()
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("install makewfs[examples] to run this example") from exc

    base = makewfs.load_config(Path(__file__).parent / "configs" / "shack_hartmann_minimal.toml")
    source = replace(
        base.source,
        normalization="magnitude",
        detector_photon_rate_per_s=None,
        magnitude_system="vega",
        band="R",
        throughput=0.35,
    )
    config = replace(base, source=source)
    yy, xx = np.mgrid[: config.input.shape[0], : config.input.shape[1]]
    opd = 0.08e-6 * np.exp(-(((xx - 63.5) ** 2 + (yy - 63.5) ** 2) / 1800.0))

    figure, axes = plt.subplots(1, len(args.magnitudes), figsize=(4 * len(args.magnitudes), 4))
    axes = np.atleast_1d(axes)
    for axis, magnitude in zip(axes, args.magnitudes):
        sensor = makewfs.WavefrontSensor(
            replace(config, source=replace(source, magnitude=magnitude))
        )
        frame = sensor.expose(opd, seed=100 + int(magnitude))
        axis.imshow(np.asarray(frame), origin="lower")
        axis.set_title(f"R={magnitude:g} mag")
        axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(args.output, dpi=140)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
