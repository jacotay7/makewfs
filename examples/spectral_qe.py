"""Wavelength-resolved detector QE: a blue star versus a red star.

A detector with QE that rolls off toward the red (here the real
``qhy530_pro_ii`` CMOS preset) does not detect every guide star equally. Two
stars with the *same* incident photon rate but different colors are exposed:

* Scalar QE applies one number to both, so it cannot tell them apart.
* Wavelength-resolved QE weights each wavelength by the QE curve, so the red
  star -- whose light lands where the detector is insensitive -- yields fewer
  photoelectrons than the blue star.

getframes 2.0/2.1 lacks the spectral-cube truth API, so makewfs uses the
documented QE-weighted integrated fallback; the comparison below is exact for
total detected electrons.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

import makewfs

PRESET = "qhy530_pro_ii"  # real CMOS with strong red QE rolloff
BAND_M = tuple(np.linspace(400e-9, 900e-9, 6))


def _detected_electrons(config: object, opd: np.ndarray, seed: int) -> np.ndarray:
    frame = makewfs.WavefrontSensor(config).expose(opd, seed=seed)
    assert frame.truth is not None
    return np.asarray(frame.truth.mean_photoelectrons)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="spectral_qe.png")
    args = parser.parse_args()
    try:
        import getframes
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("install makewfs[examples] to run this example") from exc

    base = makewfs.load_config(Path(__file__).parent / "configs" / "shack_hartmann_minimal.toml")
    base = replace(base, detector=replace(base.detector, preset=PRESET, qe_curve_path=None))
    opd = np.zeros(base.input.shape, dtype=np.float64)

    blue_weights = np.linspace(2.0, 0.5, len(BAND_M))
    blue_weights /= blue_weights.sum()
    red_weights = np.linspace(0.5, 2.0, len(BAND_M))
    red_weights /= red_weights.sum()

    stars = {
        "blue star": tuple(blue_weights),
        "red star": tuple(red_weights),
    }
    spectral_electrons: dict[str, np.ndarray] = {}
    spectral_totals: dict[str, float] = {}
    scalar_totals: dict[str, float] = {}
    for name, weights in stars.items():
        source = replace(base.source, wavelengths_m=BAND_M, wavelength_weights=weights)
        spectral = replace(base, source=source)
        scalar = replace(spectral, detector=replace(spectral.detector, qe_curve_path=None))
        # Force the scalar path by collapsing to a single effective wavelength.
        scalar_source = replace(source, wavelengths_m=(700e-9,), wavelength_weights=(1.0,))
        scalar = replace(scalar, source=scalar_source)
        image = _detected_electrons(spectral, opd, seed=1200)
        spectral_electrons[name] = image
        spectral_totals[name] = float(image.sum())
        scalar_totals[name] = float(_detected_electrons(scalar, opd, seed=1200).sum())

    camera = getframes.load_preset(PRESET)
    qe = camera.qe_curve

    figure = plt.figure(figsize=(14, 9), constrained_layout=True)
    grid = figure.add_gridspec(2, 3)

    # Panel: QE curve with the two stellar spectra overlaid.
    qe_axis = figure.add_subplot(grid[0, 0])
    qe_nm = qe.wavelength_nm
    qe_axis.plot(qe_nm, qe.value, color="black", label="detector QE")
    qe_axis.set_xlabel("wavelength (nm)")
    qe_axis.set_ylabel("quantum efficiency")
    qe_axis.set_xlim(380, 920)
    qe_axis.set_ylim(0, 1)
    spectrum_axis = qe_axis.twinx()
    band_nm = np.asarray(BAND_M) * 1e9
    spectrum_axis.plot(band_nm, blue_weights, "o-", color="tab:blue", label="blue star")
    spectrum_axis.plot(band_nm, red_weights, "o-", color="tab:red", label="red star")
    spectrum_axis.set_ylabel("relative photon weight")
    spectrum_axis.set_ylim(0, max(red_weights.max(), blue_weights.max()) * 1.3)
    qe_axis.set_title("Detector QE rolls off toward the red")
    lines = qe_axis.get_lines() + spectrum_axis.get_lines()
    qe_axis.legend(lines, [line.get_label() for line in lines], loc="upper right", fontsize=9)

    # Panel: detected-electron totals, spectral vs scalar QE.
    bar_axis = figure.add_subplot(grid[0, 1])
    labels = ["blue\nspectral", "red\nspectral", "blue\nscalar", "red\nscalar"]
    values = [
        spectral_totals["blue star"],
        spectral_totals["red star"],
        scalar_totals["blue star"],
        scalar_totals["red star"],
    ]
    colors = ["tab:blue", "tab:red", "tab:blue", "tab:red"]
    bars = bar_axis.bar(labels, values, color=colors, alpha=0.85)
    bar_axis.set_ylabel("total detected electrons")
    bar_axis.set_title("Spectral QE separates the colors; scalar QE cannot")
    bar_axis.bar_label(bars, fmt="%.2e", fontsize=8)

    ratio = spectral_totals["red star"] / spectral_totals["blue star"]
    text_axis = figure.add_subplot(grid[0, 2])
    text_axis.axis("off")
    text_axis.text(
        0.0,
        0.95,
        (
            "Same incident photon rate for both stars.\n\n"
            f"Wavelength-resolved QE:\n"
            f"  blue star: {spectral_totals['blue star']:.3e} e-\n"
            f"  red star:  {spectral_totals['red star']:.3e} e-\n"
            f"  red / blue: {ratio:.2f}\n\n"
            f"Scalar QE (single number):\n"
            f"  blue star: {scalar_totals['blue star']:.3e} e-\n"
            f"  red star:  {scalar_totals['red star']:.3e} e-\n"
            f"  red / blue: {scalar_totals['red star'] / scalar_totals['blue star']:.2f}\n\n"
            "Only the spectral path penalizes the red star."
        ),
        va="top",
        ha="left",
        family="monospace",
        fontsize=11,
        transform=text_axis.transAxes,
    )

    # Panels: detected-electron images for each star on a shared scale.
    vmax = float(max(image.max() for image in spectral_electrons.values()))
    blue_axis = figure.add_subplot(grid[1, 0])
    blue_artist = blue_axis.imshow(
        spectral_electrons["blue star"], origin="lower", cmap="cividis", vmin=0, vmax=vmax
    )
    blue_axis.set_title("blue star: detected electrons")
    figure.colorbar(blue_artist, ax=blue_axis, fraction=0.046, label="electrons")

    red_axis = figure.add_subplot(grid[1, 1])
    red_artist = red_axis.imshow(
        spectral_electrons["red star"], origin="lower", cmap="cividis", vmin=0, vmax=vmax
    )
    red_axis.set_title("red star: detected electrons (dimmer)")
    figure.colorbar(red_artist, ax=red_axis, fraction=0.046, label="electrons")

    diff_axis = figure.add_subplot(grid[1, 2])
    difference = spectral_electrons["blue star"] - spectral_electrons["red star"]
    diff_scale = float(np.abs(difference).max()) or 1.0
    diff_artist = diff_axis.imshow(
        difference, origin="lower", cmap="coolwarm", vmin=-diff_scale, vmax=diff_scale
    )
    diff_axis.set_title("blue - red (lost red signal)")
    figure.colorbar(diff_artist, ax=diff_axis, fraction=0.046, label="electrons")
    for axis in (blue_axis, red_axis, diff_axis):
        axis.set_xlabel("detector x (pixel)")
        axis.set_ylabel("detector y (pixel)")

    figure.suptitle(
        f"Wavelength-resolved detector QE distinguishes a blue guide star from a red one "
        f"({camera.name})",
        fontsize=14,
    )
    figure.savefig(args.output, dpi=140)
    print(f"wrote {args.output}; red/blue spectral ratio={ratio:.2f}")


if __name__ == "__main__":
    main()
