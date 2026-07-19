"""Compare thin, centre-launched, and side-launched sodium-range images."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

import makewfs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="lgs_elongation.png")
    args = parser.parse_args()
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("install makewfs[examples] to run this example") from exc

    config = makewfs.load_config(Path(__file__).parent / "configs" / "lgs_thin_beacon.toml")
    phase = np.zeros(config.input.shape, dtype=np.float64)
    profile = (89e3, 91e3)
    weights = (0.5, 0.5)
    cases = {
        "thin beacon": replace(config.source, lgs_ranges_m=(), lgs_range_weights=()),
        "centre launch": replace(
            config.source,
            lgs_ranges_m=profile,
            lgs_range_weights=weights,
            lgs_launch_position_m=(0.0, 0.0),
        ),
        "side launch": replace(
            config.source,
            lgs_ranges_m=profile,
            lgs_range_weights=weights,
            lgs_launch_position_m=(2.0, 0.0),
        ),
    }
    figure, axes = plt.subplots(2, len(cases), figsize=(12, 7))
    for column, (label, source) in enumerate(cases.items()):
        sensor = makewfs.WavefrontSensor(replace(config, source=source))
        ideal = sensor.photon_rate(phase)
        frame = sensor.expose(phase, seed=10)
        axes[0, column].imshow(ideal, origin="lower")
        axes[0, column].set_title(f"{label} ideal rate")
        axes[1, column].imshow(np.asarray(frame), origin="lower")
        axes[1, column].set_title(f"{label} ADU")
        axes[0, column].set_axis_off()
        axes[1, column].set_axis_off()
    figure.suptitle("supplied LGS return rate; geometric sodium-range approximation")
    figure.tight_layout()
    figure.savefig(args.output, dpi=140)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
