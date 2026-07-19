"""Compare a segmented pupil at one wavelength and across a source spectrum."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

import makewfs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="realistic_broadband.png")
    args = parser.parse_args()
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("install makewfs[examples] to run this example") from exc

    root = Path(__file__).parent
    base = makewfs.load_config(root / "configs" / "shack_hartmann_extended_source.toml")
    source = replace(
        base.source,
        angular_kernel_path=None,
        wavelengths_m=(600e-9, 700e-9, 800e-9),
        wavelength_weights=(0.25, 0.5, 0.25),
    )
    mono = replace(source, wavelengths_m=(700e-9,), wavelength_weights=(1.0,))
    yy, xx = np.mgrid[: base.input.shape[0], : base.input.shape[1]]
    radius = np.hypot(xx - 63.5, yy - 63.5)
    opd = 0.12e-6 * np.exp(-((radius / 32.0) ** 2))
    mono_image = makewfs.WavefrontSensor(replace(base, source=mono)).photon_rate(opd)
    broadband_image = makewfs.WavefrontSensor(replace(base, source=source)).photon_rate(opd)
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(opd * 1e9, origin="lower")
    axes[0].set_title("segmented/rotated OPD (nm)")
    axes[1].imshow(mono_image, origin="lower")
    axes[1].set_title("700 nm rate")
    axes[2].imshow(broadband_image, origin="lower")
    axes[2].set_title("600/700/800 nm rate")
    for axis in axes:
        axis.set_xlabel("native detector x")
        axis.set_ylabel("native detector y")
    figure.suptitle("realistic pupil and incoherent broadband sensing")
    figure.tight_layout()
    figure.savefig(args.output, dpi=140)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
