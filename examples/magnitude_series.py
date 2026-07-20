"""Compare NGS magnitudes while retaining one detector configuration.

The optical configuration and detector are fixed; only the guide-star magnitude
changes, so the panels show how the Shack-Hartmann frame degrades from photon-
rich to read-noise-limited through the getframes radiometry path. A fixed, real
low-noise sCMOS preset is used and named in the figure.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

import makewfs

PRESET = "hamamatsu_orca_fusion"  # real low-noise sCMOS, good for faint WFS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="magnitude_series.png")
    parser.add_argument("--magnitudes", nargs="+", type=float, default=[5.0, 9.0, 13.0])
    args = parser.parse_args()
    try:
        import getframes
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
    config = replace(base, source=source, detector=replace(base.detector, preset=PRESET))
    camera = getframes.load_preset(PRESET)
    yy, xx = np.mgrid[: config.input.shape[0], : config.input.shape[1]]
    opd = 0.08e-6 * np.exp(-(((xx - 63.5) ** 2 + (yy - 63.5) ** 2) / 1800.0))

    figure, axes = plt.subplots(
        1, len(args.magnitudes), figsize=(4.6 * len(args.magnitudes), 5.0), constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    for axis, magnitude in zip(axes, args.magnitudes):
        sensor = makewfs.WavefrontSensor(
            replace(config, source=replace(source, magnitude=magnitude))
        )
        frame = np.asarray(sensor.expose(opd, seed=100 + int(magnitude)))
        vmin, vmax = np.percentile(frame, [2.0, 99.5])
        artist = axis.imshow(frame, origin="lower", vmin=vmin, vmax=vmax)
        axis.set_title(f"R = {magnitude:g} mag")
        axis.set_xlabel("detector x (pixel)")
        axis.set_ylabel("detector y (pixel)")
        figure.colorbar(artist, ax=axis, fraction=0.046, pad=0.04, label="detector signal (ADU)")
    figure.suptitle(
        f"NGS magnitude series on a fixed detector "
        f"({camera.name}, {camera.sensor_type.value}, read noise {camera.read_noise_e:g} e-)",
        fontsize=14,
    )
    figure.savefig(args.output, dpi=140)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
