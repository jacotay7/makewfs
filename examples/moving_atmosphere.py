"""Animate frozen-flow OPD frames from pyturb through makewfs to a GIF.

pyturb supplies a wind-advected (frozen-flow) wavefront; makewfs forms the
Shack-Hartmann image and getframes exposes each frame. The raw seeing-limited
wavefront is many waves and scrambles the spots, so a residual-scaling factor
emulates a partially corrected wavefront -- weak enough that the subaperture
spots stay identifiable and visibly drift with the wind. The output is a GIF.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import makewfs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="moving_atmosphere.gif")
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--dt", type=float, default=0.02, help="seconds between frozen-flow frames")
    parser.add_argument("--seeing", type=float, default=0.6)
    parser.add_argument(
        "--residual-scale",
        type=float,
        default=0.15,
        help="scale on the raw OPD emulating partial AO correction (1.0 = raw seeing)",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()
    if args.steps < 2:
        raise SystemExit("--steps must be >= 2")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pyturb
        from matplotlib.animation import FuncAnimation, PillowWriter
    except ImportError as exc:
        raise SystemExit("install makewfs[examples,interop] to run this example") from exc

    config = makewfs.load_config(Path(__file__).parent / "configs" / "shack_hartmann_minimal.toml")
    sensor = makewfs.WavefrontSensor(config)
    atmosphere = pyturb.Atmosphere.from_profile(
        "paranal-median",
        seeing=args.seeing,
        diameter=config.telescope.pupil_diameter_m,
        n=config.input.shape[0],
        seed=args.seed,
    )

    opds: list[np.ndarray] = []
    frames: list[np.ndarray] = []
    for _, opd in atmosphere.frames(dt=args.dt, steps=args.steps):
        residual = np.asarray(pyturb.to_numpy(opd), dtype=np.float64) * args.residual_scale
        opds.append(residual)
        frames.append(np.asarray(sensor.expose(residual, seed=args.seed)).astype(np.float64))

    # Fixed color scales across the whole sequence so motion, not rescaling, is
    # what the eye tracks.
    opd_scale = float(np.abs(np.stack(opds)).max()) * 1e9
    frame_max = float(np.percentile(np.stack(frames), 99.5))

    figure, axes = plt.subplots(1, 2, figsize=(11, 5.2), constrained_layout=True)
    opd_artist = axes[0].imshow(
        opds[0] * 1e9, origin="lower", cmap="coolwarm", vmin=-opd_scale, vmax=opd_scale
    )
    axes[0].set_title("residual OPD (nm)")
    figure.colorbar(opd_artist, ax=axes[0], fraction=0.046, label="OPD (nm)")
    frame_artist = axes[1].imshow(frames[0], origin="lower", vmin=0.0, vmax=frame_max)
    axes[1].set_title("Shack-Hartmann frame (ADU)")
    figure.colorbar(frame_artist, ax=axes[1], fraction=0.046, label="signal (ADU)")
    for axis in axes:
        axis.set_xlabel("detector x (pixel)")
        axis.set_ylabel("detector y (pixel)")
    suptitle = figure.suptitle("", fontsize=13)

    def update(index: int):
        opd_artist.set_data(opds[index] * 1e9)
        frame_artist.set_data(frames[index])
        suptitle.set_text(
            f"pyturb frozen flow -> makewfs -> getframes   "
            f"t = {index * args.dt * 1e3:.0f} ms   "
            f'(seeing {args.seeing:g}", residual x{args.residual_scale:g})'
        )
        return opd_artist, frame_artist, suptitle

    animation = FuncAnimation(figure, update, frames=len(frames), blit=False)
    animation.save(args.output, writer=PillowWriter(fps=args.fps))
    plt.close(figure)
    print(f"wrote {args.output} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
