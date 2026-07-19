"""Show unmodulated and modulated pyramid response images."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

import makewfs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="pyramid_modulation.png")
    args = parser.parse_args()
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("install makewfs[examples] to run this example") from exc

    base = makewfs.load_config(Path(__file__).parent / "configs" / "pyramid_minimal.toml")
    assert base.pyramid is not None
    _yy, xx = np.mgrid[: base.input.shape[0], : base.input.shape[1]]
    tilt = 2.0e-8 * (xx - (base.input.shape[1] - 1) / 2) / base.input.shape[1]
    cases = {
        "unmodulated": (0.0, 1),
        "radius 1 λ/D": (1.0, 8),
        "radius 3 λ/D": (3.0, 16),
    }
    figure, axes = plt.subplots(2, len(cases), figsize=(12, 7))
    for column, (label, (radius, samples)) in enumerate(cases.items()):
        config = replace(
            base,
            pyramid=replace(
                base.pyramid,
                modulation_radius_lambda_over_d=radius,
                modulation_samples=samples,
            ),
        )
        sensor = makewfs.WavefrontSensor(config)
        reference = sensor.photon_rate(np.zeros(config.input.shape))
        response = sensor.photon_rate(tilt) - reference
        axes[0, column].imshow(reference, origin="lower")
        axes[0, column].set_title(label)
        axes[1, column].imshow(response, origin="lower", cmap="coolwarm")
        axes[1, column].set_title("push response (rate difference)")
        axes[0, column].set_axis_off()
        axes[1, column].set_axis_off()
    figure.suptitle("Pyramid modulation sampling and low-order sensitivity")
    figure.tight_layout()
    figure.savefig(args.output, dpi=140)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
