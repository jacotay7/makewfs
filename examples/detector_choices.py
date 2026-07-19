"""Apply one ideal SH map to several existing getframes camera presets."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

import makewfs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="detector_choices.png")
    args = parser.parse_args()
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("install makewfs[examples] to run this example") from exc

    base = makewfs.load_config(Path(__file__).parent / "configs" / "shack_hartmann_minimal.toml")
    yy, xx = np.mgrid[: base.input.shape[0], : base.input.shape[1]]
    residual = 0.08e-6 * np.exp(-(((xx - 63.5) ** 2 + (yy - 63.5) ** 2) / 1500.0))
    presets = ["generic_ccd", "generic_emccd", "generic_scmos", "generic_eapd"]
    figure, axes = plt.subplots(2, 2, figsize=(9, 8))
    for axis, preset in zip(axes.flat, presets):
        config = replace(base, detector=replace(base.detector, preset=preset))
        frame = makewfs.WavefrontSensor(config).expose(residual, seed=21)
        axis.imshow(np.asarray(frame), origin="lower")
        axis.set_title(preset)
        axis.set_axis_off()
    figure.suptitle("same ideal WFS input, different getframes detector models")
    figure.tight_layout()
    figure.savefig(args.output, dpi=140)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
