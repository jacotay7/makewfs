"""Compare integrating and correlated-double-sampling reads of a pyramid WFS.

Nondestructive-readout IR arrays are normally operated in CDS rather than as
simple integrators, so the readout mode is part of the instrument definition,
not a post-processing choice. This runs one identical ideal pyramid photon-rate
map through a C-RED One twice --- ``detector.readout_mode = "integrate"`` and
``"cds"`` --- and shows what differencing two reads of one global-reset ramp
does to the frame.

The signal is the same in both panels. What changes is everything around it:
CDS removes the reset (kTC) realization and the fixed bias structure, so the
frame becomes a signed difference about zero instead of an unsigned frame
sitting on a ~21,000 ADU pedestal. The price is amplifier read noise entering
twice, in quadrature.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

import makewfs

# The C-RED One reads up to 3500 full frames/s. CDS spends two of those reads on
# every delivered frame, so the fastest CDS frame rate is half that, and the
# integration between the pedestal and signal reads is one raw read period. The
# other half of the frame period is the reset and pedestal read, which collect no
# signal: CDS costs a factor of two in photons, and that is physical, not a
# modelling artifact.
RAW_READ_RATE_HZ = 3500.0
FRAME_RATE_HZ = RAW_READ_RATE_HZ / 2.0
INTEGRATION_S = 1.0 / RAW_READ_RATE_HZ

# The C-RED One is an IR array: its QE is zero below 800 nm and peaks at 80% from
# 1450 nm out. Sensing it in the visible would return pure noise, so this example
# moves the pyramid to H band, where such a sensor is actually used.
SENSING_WAVELENGTH_M = 1.65e-6


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="cds_readout.png")
    parser.add_argument(
        "--photon-rate-per-s",
        type=float,
        default=5.0e9,
        help=(
            "detector-surface photon rate. The default puts the pupils about "
            "25x above the CDS noise, so the readout comparison is made on "
            "clearly visible signal rather than at SNR ~ 1"
        ),
    )
    args = parser.parse_args()
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("install makewfs[examples] to run this example") from exc

    base = makewfs.load_config(Path(__file__).parent / "configs" / "pyramid_minimal.toml")
    base = replace(
        base,
        source=replace(base.source, detector_photon_rate_per_s=args.photon_rate_per_s),
        sensor=replace(base.sensor, wavelength_m=SENSING_WAVELENGTH_M),
        detector=replace(
            base.detector,
            preset="first_light_imaging_cred_one",
            exposure_s=INTEGRATION_S,
            temperature_c=-188.55,  # 84.6 K, the temperature the preset was fitted at
            binning=1,
        ),
    )

    # A weak defocus-like residual, so the four pupil images are not identical
    # and the readout comparison is made on a real signal rather than a flat.
    yy, xx = np.mgrid[: base.input.shape[0], : base.input.shape[1]]
    centre = (base.input.shape[0] - 1) / 2
    residual = 0.06e-6 * (((xx - centre) ** 2 + (yy - centre) ** 2) / centre**2 - 0.5)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)
    for axis, mode in zip(axes, ("integrate", "cds")):
        config = replace(base, detector=replace(base.detector, readout_mode=mode))
        # The same seed in both panels: the difference is the readout mode alone.
        frame = np.asarray(makewfs.WavefrontSensor(config).expose(residual, seed=21).data)
        vmin, vmax = np.percentile(frame, [1.0, 99.0])
        artist = axis.imshow(frame, origin="lower", vmin=vmin, vmax=vmax)
        axis.set_title(
            f"readout_mode = {mode!r}\n"
            f"{frame.dtype}, median {np.median(frame):,.0f} ADU, "
            f"spatial rms {frame.std():,.0f} ADU"
        )
        axis.set_xlabel("detector x (pixel)")
        axis.set_ylabel("detector y (pixel)")
        figure.colorbar(
            artist,
            ax=axis,
            fraction=0.046,
            pad=0.04,
            label="noisy detector signal (ADU)",
        )
    figure.suptitle(
        f"C-RED One pyramid frame, H band, {FRAME_RATE_HZ:g} Hz CDS "
        f"({INTEGRATION_S * 1e6:.0f} us integration)",
        fontsize=14,
    )
    figure.savefig(args.output, dpi=140)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
