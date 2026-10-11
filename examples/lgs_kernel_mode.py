"""Compare the fast sodium-LGS kernel mode with the range/angle quadrature.

``source.lgs_model = "kernel"`` (``configs/lgs_kernel.toml``) folds the source
extent and the sodium range profile into one fixed convolution per subaperture,
so each frame needs one propagation per wavelength instead of one per source
angle and range. This script renders the same instrument both ways, prints the
warm frame time and captured flux of each, and saves both ideal photon-rate
images.

The quadrature model represents the 0.5 arcsec Gaussian source by a 3 x 3
Gauss-Hermite rule, which shows as a dotted spot; the kernel uses the
continuous Gaussian. Against a densely sampled Gaussian quadrature the kernel
agrees to about 0.1% of the image (see docs/validation.md).
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np

import makewfs


def _warm_rate(sensor: makewfs.WavefrontSensor, opd: np.ndarray) -> tuple[np.ndarray, float]:
    sensor.photon_rate(opd)
    start = perf_counter()
    rate = np.asarray(sensor.photon_rate(opd))
    return rate, perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="lgs_kernel_mode.png")
    args = parser.parse_args()
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("install makewfs[examples] to run this example") from exc

    config = makewfs.load_config(Path(__file__).parent / "configs" / "lgs_kernel.toml")
    opd = np.zeros(config.input.shape, dtype=np.float32)
    models = {
        "quadrature": replace(config, source=replace(config.source, lgs_model="quadrature")),
        "kernel": config,
    }
    rates: dict[str, np.ndarray] = {}
    for name, model_config in models.items():
        sensor = makewfs.WavefrontSensor(model_config)
        rate, elapsed = _warm_rate(sensor, opd)
        rates[name] = rate
        launched = config.source.detector_photon_rate_per_s
        assert launched is not None
        print(
            f"{name:>10}: {len(sensor.engine.source_states):3d} propagated states, "
            f"warm frame {1e3 * elapsed:7.1f} ms, captured {rate.sum() / launched:.4f} "
            "of the launched rate"
        )

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    scale = max(float(rates["quadrature"].max()), float(rates["kernel"].max()))
    titles = {
        "quadrature": "quadrature model\n(9 source angles x 9 ranges)",
        "kernel": "kernel model\n(1 propagation + per-subaperture convolution)",
    }
    for axis, (name, rate) in zip(axes[:2], rates.items()):
        artist = axis.imshow(rate, origin="lower", cmap="inferno", vmin=0.0, vmax=scale)
        axis.set_title(f"{titles[name]}\nideal photon rate")
        figure.colorbar(artist, ax=axis, fraction=0.046, label="photons/s/pixel")
    difference = rates["kernel"] - rates["quadrature"]
    limit = float(np.max(np.abs(difference)))
    artist = axes[2].imshow(difference, origin="lower", cmap="coolwarm", vmin=-limit, vmax=limit)
    axes[2].set_title("kernel - quadrature\n(ideal photon rate)")
    figure.colorbar(artist, ax=axes[2], fraction=0.046, label="photons/s/pixel")
    for axis in axes:
        axis.set_xlabel("detector x (pixel)")
        axis.set_ylabel("detector y (pixel)")
    figure.suptitle(
        "Side-launched sodium LGS: same geometry, two image-formation models "
        "(geometric range approximation, supplied return rate)"
    )
    figure.savefig(args.output, dpi=120)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
