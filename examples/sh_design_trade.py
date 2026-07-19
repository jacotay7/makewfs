"""Plot normalized Shack-Hartmann design trade-offs from one config."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

import makewfs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="sh_design_trade.png")
    args = parser.parse_args()
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("install makewfs[examples] to run this example") from exc

    root = Path(__file__).parent
    base = makewfs.load_config(root / "configs" / "shack_hartmann_minimal.toml")
    assert base.shack_hartmann is not None
    phase = np.zeros(base.input.shape, dtype=np.float64)
    cases = {
        "8x8, 8 px": base.shack_hartmann,
        "8x8, field stop": replace(base.shack_hartmann, field_stop_radius_lambda_over_d=0.8),
        "8x8, measured blur": replace(base.shack_hartmann, optical_blur_fwhm_pixels=1.0),
    }
    figure, axes = plt.subplots(1, len(cases), figsize=(12, 4))
    for axis, (label, settings) in zip(axes, cases.items()):
        image = makewfs.WavefrontSensor(replace(base, shack_hartmann=settings)).photon_rate(phase)
        axis.imshow(image, origin="lower")
        axis.set_title(f"{label}\n{image.sum():.3g} photons/s")
        axis.set_xlabel("native detector x")
        axis.set_ylabel("native detector y")
    figure.suptitle("Shack-Hartmann sampling and optical-window trade-offs")
    figure.tight_layout()
    figure.savefig(args.output, dpi=140)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
