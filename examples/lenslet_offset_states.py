"""Render one wavefront through four pre-declared lenslet-grid offset states.

``shack_hartmann.lenslet_grid_offset_states_fraction`` declares positions of
the lenslet grid relative to the pupil (here half-pitch dithers). makewfs
builds and caches each state's geometry once; ``offset_state`` selects one per
call, so a dithered or time-varying Shack-Hartmann loop needs no rebuilt
sensor. Each state has its own valid-subaperture map, and every frame records
its state in ``wfs_lenslet_offset_state``.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np

import makewfs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="lenslet_offset_states.png")
    args = parser.parse_args()
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("install makewfs[examples] to run this example") from exc

    sensor = makewfs.WavefrontSensor.from_toml(
        Path(__file__).parent / "configs" / "shack_hartmann_offset_states.toml"
    )
    shape = sensor.config.input.shape
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    radius = np.hypot(xx - (shape[1] - 1) / 2, yy - (shape[0] - 1) / 2) / (shape[0] / 2)
    opd = (1.5e-6 * (2.0 * radius**2 - 1.0)).astype(np.float32)  # defocus, metres

    offsets = sensor.lenslet_grid_offsets_fraction()
    frames = list(
        sensor.expose_many(
            [opd] * len(offsets),
            seeds=range(len(offsets)),
            offset_states=itertools.cycle(range(len(offsets))),
        )
    )
    figure, axes = plt.subplots(2, len(offsets), figsize=(4.2 * len(offsets), 8.2))
    for state, (offset, frame) in enumerate(zip(offsets, frames)):
        assert frame.metadata["wfs_lenslet_offset_state"] == state
        valid = np.asarray(sensor.valid_subapertures(offset_state=state))
        print(f"state {state}: offset {offset} pitch, {int(valid.sum())} valid subapertures")
        rate = np.asarray(sensor.photon_rate(opd, offset_state=state))
        artist = axes[0, state].imshow(rate, origin="lower", cmap="inferno")
        axes[0, state].set_title(f"state {state}: offset {offset}\nideal photon rate")
        figure.colorbar(artist, ax=axes[0, state], fraction=0.046, label="photons/s/pixel")
        artist = axes[1, state].imshow(valid, origin="lower", cmap="Greys", vmin=0, vmax=1)
        axes[1, state].set_title(f"valid subapertures ({int(valid.sum())})")
        figure.colorbar(artist, ax=axes[1, state], fraction=0.046, label="valid (1) / invalid (0)")
        axes[0, state].set_xlabel("detector x (pixel)")
        axes[0, state].set_ylabel("detector y (pixel)")
        axes[1, state].set_xlabel("lenslet x")
        axes[1, state].set_ylabel("lenslet y")
    figure.suptitle(
        "One defocused wavefront, four lenslet-grid offset states (1.5 um peak defocus OPD)"
    )
    figure.tight_layout()
    figure.savefig(args.output, dpi=110)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
