"""Feed moving frozen-flow OPD frames from pyturb through makewfs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import makewfs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="moving_atmosphere.png")
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.steps < 6:
        raise SystemExit("--steps must be at least 6 for the two-row diagnostic plot")
    try:
        import matplotlib.pyplot as plt
        import pyturb
    except ImportError as exc:
        raise SystemExit("install makewfs[examples,interop] to run this example") from exc

    config = makewfs.load_config(Path(__file__).parent / "configs" / "shack_hartmann_minimal.toml")
    sensor = makewfs.WavefrontSensor(config)
    atmosphere = pyturb.Atmosphere.from_profile(
        "paranal-median",
        seeing=0.8,
        diameter=config.telescope.pupil_diameter_m,
        n=config.input.shape[0],
        seed=args.seed,
    )
    rates: list[np.ndarray] = []
    frames: list[np.ndarray] = []
    for _, opd in atmosphere.frames(dt=config.detector.exposure_s, steps=args.steps):
        residual = np.asarray(pyturb.to_numpy(opd), dtype=np.float64)
        rates.append(sensor.photon_rate(residual))
        frames.append(np.asarray(sensor.expose(residual, seed=args.seed)))

    figure, axes = plt.subplots(2, 3, figsize=(12, 7))
    for index, axis in enumerate(axes.flat):
        image = rates[index] if index < 3 else frames[index - 3]
        axis.imshow(image, origin="lower")
        axis.set_title("ideal rate" if index < 3 else "detector ADU")
        axis.set_axis_off()
    figure.suptitle("pyturb frozen flow → makewfs optics → getframes detector")
    figure.tight_layout()
    figure.savefig(args.output, dpi=140)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
