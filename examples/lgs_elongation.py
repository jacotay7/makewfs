"""Compare thin, centre-launched, and side-launched sodium-range images.

The current Shack-Hartmann model represents a finite sodium layer as a weighted
set of ranges. Each range shifts every subaperture spot by the launch-to-
subaperture parallax, so spots elongate radially away from the launch position
with a length that grows with distance from the launch. This script samples the
layer finely enough to render smooth streaks and uses a wide subaperture field
of view so the elongation is not clipped. It is a geometric approximation:
makewfs does not synthesize range-resolved turbulence or predict return flux.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

import makewfs

# Physical sodium layer sampled as a weighted range profile (mean ~90 km).
RANGES_M = tuple(np.linspace(86e3, 94e3, 11))
RANGE_WEIGHTS = tuple(np.full(len(RANGES_M), 1.0 / len(RANGES_M)))
LAUNCH_EDGE_M = (4.0, 0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="lgs_elongation.png")
    args = parser.parse_args()
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("install makewfs[examples] to run this example") from exc

    config = makewfs.load_config(Path(__file__).parent / "configs" / "lgs_thin_beacon.toml")
    # Widen each subaperture's field of view so multi-pixel streaks are not
    # clipped by the minimal 8-pixel window used elsewhere.
    config = replace(
        config,
        shack_hartmann=replace(
            config.shack_hartmann,
            pixels_per_subaperture=20,
            spot_sampling_pixels_per_lambda_over_d=2.5,
        ),
    )
    phase = np.zeros(config.input.shape, dtype=np.float64)
    cases = {
        "thin beacon\n(point-like)": replace(config.source, lgs_ranges_m=(), lgs_range_weights=()),
        "centre launch\n(radial from centre)": replace(
            config.source,
            lgs_ranges_m=RANGES_M,
            lgs_range_weights=RANGE_WEIGHTS,
            lgs_launch_position_m=(0.0, 0.0),
        ),
        "side launch at (4, 0) m\n(radial from launch)": replace(
            config.source,
            lgs_ranges_m=RANGES_M,
            lgs_range_weights=RANGE_WEIGHTS,
            lgs_launch_position_m=LAUNCH_EDGE_M,
        ),
    }

    figure, axes = plt.subplots(2, len(cases), figsize=(13, 8), constrained_layout=True)
    for column, (label, source) in enumerate(cases.items()):
        sensor = makewfs.WavefrontSensor(replace(config, source=source))
        ideal = sensor.photon_rate(phase)
        frame = np.asarray(sensor.expose(phase, seed=10))
        rate_artist = axes[0, column].imshow(ideal, origin="lower", cmap="inferno")
        axes[0, column].set_title(f"{label}\nideal rate")
        figure.colorbar(rate_artist, ax=axes[0, column], fraction=0.046, label="photons/s/pixel")
        frame_artist = axes[1, column].imshow(frame, origin="lower", cmap="inferno")
        axes[1, column].set_title("detector frame")
        figure.colorbar(frame_artist, ax=axes[1, column], fraction=0.046, label="signal (ADU)")
        for row in (0, 1):
            axes[row, column].set_xlabel("detector x (pixel)")
            axes[row, column].set_ylabel("detector y (pixel)")
    figure.suptitle(
        "Sodium LGS spot elongation: radial from the launch, growing with launch distance\n"
        "(supplied return rate; geometric range approximation, not range-resolved turbulence)",
        fontsize=13,
    )
    figure.savefig(args.output, dpi=140)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
