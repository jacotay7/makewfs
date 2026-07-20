"""Show a thin LGS return with cone-effect OPD supplied by pyturb.

This is intentionally a thin-beacon example. The current source contract does
not predict laser return flux or range-resolved sodium elongation; the configured
photon rate is supplied by the user and those extensions remain on the roadmap.

The plot contrasts the flat-wavefront thin beacon (sharp, point-like spots -- the
"thin" part of the name) with the same beacon seen through pyturb's cone-effect
LGS OPD, which displaces and blurs the subaperture spots. All panels carry units.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import makewfs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="lgs_thin_beacon.png")
    parser.add_argument("--seeing", type=float, default=0.5)
    args = parser.parse_args()
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pyturb
    except ImportError as exc:
        raise SystemExit("install makewfs[examples,interop] to run this example") from exc

    config = makewfs.load_config(Path(__file__).parent / "configs" / "lgs_thin_beacon.toml")
    sensor = makewfs.WavefrontSensor(config)
    flat = np.zeros(config.input.shape, dtype=np.float64)

    atmosphere = pyturb.Atmosphere.from_profile(
        "paranal-median",
        seeing=args.seeing,
        diameter=config.telescope.pupil_diameter_m,
        n=config.input.shape[0],
        lgs_altitude=90e3,
        seed=3,
    )
    _, opd = next(atmosphere.frames(dt=config.detector.exposure_s, steps=1))
    opd_m = np.asarray(pyturb.to_numpy(opd), dtype=np.float64)

    flat_rate = sensor.photon_rate(flat)
    turbulent_rate = sensor.photon_rate(opd_m)
    turbulent_frame = np.asarray(sensor.expose(opd_m, seed=3))

    figure, axes = plt.subplots(2, 2, figsize=(11, 9.5), constrained_layout=True)
    panels = (
        (axes[0, 0], flat_rate, "flat wavefront: point-like spots", "photons/s/pixel", "inferno"),
        (
            axes[0, 1],
            opd_m * 1e9,
            f"pyturb cone-effect LGS OPD\n({np.std(opd_m) * 1e9:.0f} nm RMS)",
            "OPD (nm)",
            "coolwarm",
        ),
        (
            axes[1, 0],
            turbulent_rate,
            "thin beacon through turbulence: ideal rate",
            "photons/s/pixel",
            "inferno",
        ),
        (
            axes[1, 1],
            turbulent_frame,
            "thin beacon through turbulence: detector frame",
            "signal (ADU)",
            "inferno",
        ),
    )
    for axis, image, title, label, cmap in panels:
        artist = axis.imshow(image, origin="lower", cmap=cmap)
        axis.set_title(title)
        axis.set_xlabel("detector x (pixel)")
        axis.set_ylabel("detector y (pixel)")
        figure.colorbar(artist, ax=axis, fraction=0.046, pad=0.04, label=label)
    figure.suptitle(
        "Thin sodium LGS: pyturb cone-effect OPD -> makewfs SH optics -> getframes frame\n"
        "(user-supplied return rate; spots stay point-like -- no range elongation)",
        fontsize=13,
    )
    figure.savefig(args.output, dpi=140)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
