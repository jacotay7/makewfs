"""Show a thin LGS return with cone-effect OPD supplied by pyturb.

This is intentionally a thin-beacon example. The current source contract does
not predict laser return flux or range-resolved sodium elongation; the configured
photon rate is supplied by the user and those extensions remain on the roadmap.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import makewfs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="lgs_thin_beacon.png")
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
    atmosphere = pyturb.Atmosphere.from_profile(
        "paranal-median",
        seeing=0.8,
        diameter=config.telescope.pupil_diameter_m,
        n=config.input.shape[0],
        lgs_altitude=90e3,
        seed=3,
    )
    _, opd = next(atmosphere.frames(dt=config.detector.exposure_s, steps=1))
    opd_m = np.asarray(pyturb.to_numpy(opd), dtype=np.float64)
    ideal = sensor.photon_rate(opd_m)
    frame = sensor.expose(opd_m, seed=3)

    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].imshow(ideal, origin="lower")
    axes[0].set_title("thin LGS ideal rate")
    axes[1].imshow(np.asarray(frame), origin="lower")
    axes[1].set_title("thin LGS detector ADU")
    for axis in axes:
        axis.set_axis_off()
    figure.suptitle("pyturb LGS cone-effect OPD; supplied return rate")
    figure.tight_layout()
    figure.savefig(args.output, dpi=140)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
