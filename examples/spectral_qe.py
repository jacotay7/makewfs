"""Compare scalar-QE and wavelength-resolved detector exposure."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

import makewfs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="spectral_qe.png")
    args = parser.parse_args()
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("install makewfs[examples] to run this example") from exc

    root = Path(__file__).parent
    path = root / "configs" / "shack_hartmann_spectral_qe.toml"
    spectral_config = makewfs.load_config(path)
    scalar_config = replace(
        spectral_config,
        detector=replace(spectral_config.detector, qe_curve_path=None),
    )
    opd = np.zeros(spectral_config.input.shape, dtype=np.float64)
    spectral_sensor = makewfs.WavefrontSensor(spectral_config)
    scalar_sensor = makewfs.WavefrontSensor(scalar_config)
    spectral_frame = spectral_sensor.expose(opd, seed=1200)
    scalar_frame = scalar_sensor.expose(opd, seed=1200)
    assert spectral_frame.truth is not None
    assert scalar_frame.truth is not None
    spectral_truth = np.asarray(spectral_frame.truth.mean_photoelectrons)
    scalar_truth = np.asarray(scalar_frame.truth.mean_photoelectrons)
    difference = spectral_truth - scalar_truth

    figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for axis, image, title, label, cmap in (
        (axes[0], scalar_truth, "scalar QE", "electrons", "viridis"),
        (axes[1], spectral_truth, "wavelength-resolved QE", "electrons", "viridis"),
        (axes[2], difference, "spectral - scalar", "electrons", "coolwarm"),
    ):
        artist = axis.imshow(image, origin="lower", cmap=cmap)
        axis.set_title(title)
        axis.set_xlabel("detector x (pixel)")
        axis.set_ylabel("detector y (pixel)")
        figure.colorbar(artist, ax=axis, label=label)
    figure.suptitle("spatially varying spectral detector response; seed=1200")
    figure.savefig(args.output, dpi=140)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
