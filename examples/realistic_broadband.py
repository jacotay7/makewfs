"""Segmented/rotated pupil and monochromatic vs incoherent broadband sensing.

Two "realistic" features are shown together:

* The telescope pupil is a rotated segmented aperture with a central
  obscuration (like a segmented primary). The segmentation lives in the pupil
  *amplitude*, so the first panel shows that amplitude mask directly -- it is not
  visible in the injected OPD, which is a separate smooth test wavefront.
* A broadband source is an incoherent sum over wavelengths. Because each
  subaperture spot scales with wavelength, the broadband spot is radially
  smeared relative to the monochromatic one; the zoom panels make that visible.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

import makewfs
from makewfs.pupil import make_pupil


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
    # A wide, well-sampled band makes the chromatic spot smearing unambiguous
    # (each subaperture spot scales with wavelength).
    band_m = tuple(np.linspace(500e-9, 900e-9, 5))
    weights = tuple(np.full(len(band_m), 1.0 / len(band_m)))
    broadband_source = replace(
        base.source,
        angular_kernel_path=None,
        wavelengths_m=band_m,
        wavelength_weights=weights,
    )
    mono_source = replace(broadband_source, wavelengths_m=(700e-9,), wavelength_weights=(1.0,))

    yy, xx = np.mgrid[: base.input.shape[0], : base.input.shape[1]]
    radius = np.hypot(xx - 63.5, yy - 63.5)
    opd = 0.12e-6 * np.exp(-((radius / 32.0) ** 2))

    pupil = np.asarray(
        make_pupil(base.telescope, (256, 256), base.input.grid_extent_m, supersampling=2)
    )
    mono_image = makewfs.WavefrontSensor(replace(base, source=mono_source)).photon_rate(opd)
    broadband_image = makewfs.WavefrontSensor(replace(base, source=broadband_source)).photon_rate(
        opd
    )

    # Zoom on one off-axis subaperture so the chromatic spot smearing is visible.
    lenslets = base.shack_hartmann.lenslets_across_pupil
    ps = mono_image.shape[0] // lenslets
    ly, lx = lenslets // 2, lenslets - 2
    sy, sx = slice(ly * ps, (ly + 1) * ps), slice(lx * ps, (lx + 1) * ps)
    mono_spot = mono_image[sy, sx]
    broad_spot = broadband_image[sy, sx]
    spot_max = float(max(mono_spot.max(), broad_spot.max()))

    figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    pupil_artist = axes[0, 0].imshow(pupil, origin="lower", cmap="gray")
    axes[0, 0].set_title(
        f"segmented pupil amplitude\n(rotated {base.telescope.pupil_rotation_deg:g} deg, "
        f"{base.telescope.central_obscuration_ratio:g} obscuration)"
    )
    figure.colorbar(pupil_artist, ax=axes[0, 0], fraction=0.046, label="transmission")

    opd_artist = axes[0, 1].imshow(opd * 1e9, origin="lower", cmap="coolwarm")
    axes[0, 1].set_title("injected test wavefront (separate from pupil)")
    figure.colorbar(opd_artist, ax=axes[0, 1], fraction=0.046, label="OPD (nm)")

    mono_artist = axes[0, 2].imshow(mono_image, origin="lower", cmap="viridis")
    axes[0, 2].set_title("monochromatic 700 nm rate")
    figure.colorbar(mono_artist, ax=axes[0, 2], fraction=0.046, label="photons/s/pixel")

    broad_artist = axes[1, 0].imshow(broadband_image, origin="lower", cmap="viridis")
    axes[1, 0].set_title("broadband 500-900 nm rate")
    figure.colorbar(broad_artist, ax=axes[1, 0], fraction=0.046, label="photons/s/pixel")

    mono_zoom = axes[1, 1].imshow(mono_spot, origin="lower", cmap="viridis", vmin=0, vmax=spot_max)
    axes[1, 1].set_title("one subaperture: monochromatic")
    figure.colorbar(mono_zoom, ax=axes[1, 1], fraction=0.046, label="photons/s/pixel")

    broad_zoom = axes[1, 2].imshow(
        broad_spot, origin="lower", cmap="viridis", vmin=0, vmax=spot_max
    )
    axes[1, 2].set_title("one subaperture: broadband (radially smeared)")
    figure.colorbar(broad_zoom, ax=axes[1, 2], fraction=0.046, label="photons/s/pixel")

    for axis in axes.flat:
        axis.set_xlabel("detector x (pixel)")
        axis.set_ylabel("detector y (pixel)")
    figure.suptitle(
        "Realistic segmented pupil and incoherent broadband Shack-Hartmann sensing", fontsize=14
    )
    figure.savefig(args.output, dpi=140)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
