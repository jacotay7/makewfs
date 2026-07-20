"""Apply one ideal SH map to several real getframes camera presets.

Detector choice matters most in the photon-starved regime, so this uses a faint
guide star (magnitude mode) where read noise and EM gain separate the cameras.
Each panel is one real getframes preset exposed from the identical optical
photon-rate map; only the detector model changes.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

import makewfs

# Real getframes presets spanning the archetypes visible-band AO cameras fall
# into. All have useful QE at the 700 nm sensing wavelength (a near-IR detector
# such as SAPHIRA would read pure noise here because its QE is zero at 700 nm).
PRESETS = (
    "scimeasure_little_joe_ccd39",  # classic Shack-Hartmann CCD
    "andor_ixon_ultra_888",  # EMCCD (sub-electron effective read noise)
    "hamamatsu_orca_fusion",  # modern sCMOS
    "zwo_asi2600mm",  # back-illuminated CMOS
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="detector_choices.png")
    parser.add_argument("--magnitude", type=float, default=11.0)
    args = parser.parse_args()
    try:
        import getframes
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("install makewfs[examples] to run this example") from exc

    base = makewfs.load_config(Path(__file__).parent / "configs" / "shack_hartmann_minimal.toml")
    faint_source = replace(
        base.source,
        normalization="magnitude",
        detector_photon_rate_per_s=None,
        magnitude_system="vega",
        magnitude=args.magnitude,
        band="R",
        throughput=0.35,
    )
    base = replace(base, source=faint_source)
    yy, xx = np.mgrid[: base.input.shape[0], : base.input.shape[1]]
    residual = 0.08e-6 * np.exp(-(((xx - 63.5) ** 2 + (yy - 63.5) ** 2) / 1500.0))

    figure, axes = plt.subplots(2, 2, figsize=(11, 9.5), constrained_layout=True)
    for axis, preset in zip(axes.flat, PRESETS):
        camera = getframes.load_preset(preset)
        config = replace(base, detector=replace(base.detector, preset=preset))
        frame = np.asarray(makewfs.WavefrontSensor(config).expose(residual, seed=21))
        # Percentile limits keep spots visible across presets whose bias pedestal
        # (e.g. the EM-amplified eAPD offset) would otherwise dominate autoscaling.
        vmin, vmax = np.percentile(frame, [2.0, 99.5])
        artist = axis.imshow(frame, origin="lower", vmin=vmin, vmax=vmax)
        gain = f", EM gain {camera.em_gain:g}" if camera.em_gain > 1 else ""
        axis.set_title(
            f"{camera.name}\n{camera.sensor_type.value}, "
            f"read noise {camera.read_noise_e:g} e-, QE {camera.quantum_efficiency:g}{gain}"
        )
        axis.set_xlabel("detector x (pixel)")
        axis.set_ylabel("detector y (pixel)")
        figure.colorbar(artist, ax=axis, fraction=0.046, pad=0.04, label="detector signal (ADU)")
    figure.suptitle(
        f"Same faint WFS input (R = {args.magnitude:g} mag), different getframes detectors",
        fontsize=14,
    )
    figure.savefig(args.output, dpi=140)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
