"""Scintillated field to Shack-Hartmann image, subaperture fluxes, and slopes.

A single strong turbulent layer at 12 km (a ``pyturb`` phase screen) is
propagated to the pupil with ``aocore``'s Fresnel (angular-spectrum)
propagator. The pupil field ``U`` is then split into what makewfs takes per
frame: the amplitude ``|U|`` (relative to the unscintillated field) and an OPD
that carries the field's phase. A deformable mirror is emulated by removing the
OPD's spatial frequencies below the lenslet Nyquist frequency ``1 / (2 d)``.

The script renders the closed-loop residual with and without the amplitude,
and with a square spatial filter of width ``lambda / d`` in front of the
lenslets, and reports:

* the subaperture fluxes (``WavefrontSensor.subaperture_flux``) against the
  pupil intensity they sample: without the filter they measure the
  scintillation; behind the filter the residual phase also becomes intensity;
* the centre-of-gravity slopes, computed here only as an illustration
  (centroiding belongs to downstream AO software), with and without the
  amplitude, to show how much scintillation moves them.

Everything is ideal (noise-free) photon rate; ``expose`` would add the detector.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
from aocore import AngularSpectrumPropagator

import makewfs


def _cog_slopes(image: np.ndarray, lenslets: int, pixels: int) -> np.ndarray:
    """Plain centre-of-gravity slopes in pixels, ``(2, lenslets, lenslets)`` for (x, y)."""
    blocks = image.reshape(lenslets, pixels, lenslets, pixels).transpose(0, 2, 1, 3)
    coordinate = np.arange(pixels) - (pixels - 1) / 2
    total = np.maximum(blocks.sum(axis=(2, 3)), 1e-30)
    x = (blocks.sum(axis=2) * coordinate).sum(axis=2) / total
    y = (blocks.sum(axis=3) * coordinate).sum(axis=2) / total
    return np.stack([x, y])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="scintillation.png")
    parser.add_argument("--altitude-m", type=float, default=12000.0)
    parser.add_argument(
        "--r0-m", type=float, default=0.12, help="layer Fried parameter at 500 nm (metres)"
    )
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pyturb
    except ImportError as exc:
        raise SystemExit("install makewfs[examples,interop] to run this example") from exc

    config = makewfs.load_config(
        Path(__file__).parent / "configs" / "shack_hartmann_scintillation.toml"
    )
    assert config.shack_hartmann is not None
    sensor = makewfs.WavefrontSensor(config)
    size = config.input.shape[0]
    pitch_m = config.input.grid_extent_m / size
    wavelength = config.sensor.wavelength_m
    lenslets = config.shack_hartmann.lenslets_across_pupil
    pixels = config.shack_hartmann.pixels_per_subaperture
    subaperture_m = config.input.grid_extent_m / lenslets

    # Atmosphere (pyturb): one periodic layer, phase in radians at 500 nm.
    screen = pyturb.PhaseScreen(
        n=size, pixel_scale=pitch_m, r0=args.r0_m, seed=args.seed, dtype="float64"
    )
    layer_opd = np.asarray(pyturb.to_numpy(screen.generate())) * 500e-9 / (2 * math.pi)
    # Fresnel propagation to the pupil (aocore). The periodic screen matches
    # the FFT propagator's periodic grid.
    propagate = AngularSpectrumPropagator(
        (size, size), pitch_m, wavelength, args.altitude_m, paraxial=True
    )
    field = np.asarray(propagate(np.exp(2j * math.pi * layer_opd / wavelength)))
    amplitude = np.abs(field)
    # The field's phase as OPD: the layer OPD plus the small diffractive
    # departure, which needs no unwrapping while it stays within +-pi.
    departure = np.angle(field * np.exp(-2j * math.pi * layer_opd / wavelength))
    pupil_opd = layer_opd + departure * wavelength / (2 * math.pi)
    # Closed loop: a DM removes the OPD below the lenslet Nyquist frequency.
    frequency = np.fft.fftfreq(size, d=pitch_m)
    band = (np.abs(frequency)[None, :] < 1 / (2 * subaperture_m)) & (
        np.abs(frequency)[:, None] < 1 / (2 * subaperture_m)
    )
    residual = pupil_opd - np.real(np.fft.ifft2(np.fft.fft2(layer_opd) * band))

    pupil = np.asarray(sensor.pupil_illumination(), dtype=np.float64) > 0
    intensity = amplitude**2
    scintillation_index = float(np.var(intensity[pupil]) / np.mean(intensity[pupil]) ** 2)
    print(f"pupil scintillation index sigma_I^2 = {scintillation_index:.3f}")
    print(f"residual OPD RMS in the pupil = {np.std(residual[pupil]) * 1e9:.0f} nm")

    valid = np.asarray(sensor.valid_subapertures())
    flat = sensor.subaperture_flux(np.zeros_like(residual))
    plain = sensor.subaperture_flux(residual)
    scintillated = sensor.subaperture_flux(residual, amplitude=amplitude)
    filtered_config = replace(
        config,
        shack_hartmann=replace(config.shack_hartmann, spatial_filter_width_lambda_over_d=1.0),
    )
    filtered_sensor = makewfs.WavefrontSensor(filtered_config)
    filtered_flat = filtered_sensor.subaperture_flux(np.zeros_like(residual))
    filtered = filtered_sensor.subaperture_flux(residual, amplitude=amplitude)

    # The pupil intensity each subaperture samples, relative to its flat value.
    truth = intensity.reshape(lenslets, size // lenslets, lenslets, size // lenslets)
    truth = (truth * pupil.reshape(truth.shape)).sum(axis=(1, 3))
    truth = truth / np.maximum(
        pupil.reshape(lenslets, -1, lenslets, size // lenslets).sum(axis=(1, 3)), 1
    )

    def relative(flux: np.ndarray, reference: np.ndarray) -> np.ndarray:
        return np.asarray(flux)[valid] / np.asarray(reference)[valid]

    pupil_flux = relative(scintillated.pupil_photons_s, flat.pupil_photons_s)
    detector = relative(scintillated.detector_photons_s, flat.detector_photons_s)
    behind_filter = relative(filtered.detector_photons_s, filtered_flat.detector_photons_s)
    for name, values in (
        ("pupil plane", pupil_flux),
        ("detector, no filter", detector),
        ("detector, lambda/d spatial filter", behind_filter),
    ):
        correlation = float(np.corrcoef(values, truth[valid])[0, 1])
        print(
            f"subaperture flux ({name}): relative RMS {np.std(values / values.mean()):.3f}, "
            f"correlation with the pupil intensity {correlation:.2f}"
        )
    slopes_plain = _cog_slopes(np.asarray(plain.photon_rate), lenslets, pixels)
    slopes_scintillated = _cog_slopes(np.asarray(scintillated.photon_rate), lenslets, pixels)
    change = (slopes_scintillated - slopes_plain)[:, valid]
    spread = slopes_plain[:, valid]
    print(
        f"CoG slope RMS {np.std(spread) * 1e3:.1f} mpx; scintillation changes it by "
        f"{np.std(change) * 1e3:.2f} mpx RMS"
    )

    figure, axes = plt.subplots(2, 3, figsize=(15, 9.4))
    extent = [-config.input.grid_extent_m / 2, config.input.grid_extent_m / 2] * 2
    panels = [
        (
            axes[0, 0],
            np.where(pupil, intensity, np.nan),
            "pupil intensity |U|^2",
            "relative intensity",
            extent,
        ),
        (
            axes[0, 1],
            np.asarray(scintillated.photon_rate),
            "SH image with scintillation\n(ideal photon rate)",
            "photons/s/pixel",
            None,
        ),
        (
            axes[0, 2],
            np.asarray(filtered.photon_rate),
            "behind a lambda/d spatial filter\n(ideal photon rate)",
            "photons/s/pixel",
            None,
        ),
    ]
    for axis, image, title, label, image_extent in panels:
        artist = axis.imshow(image, origin="lower", extent=image_extent, cmap="inferno")
        axis.set_title(title)
        figure.colorbar(artist, ax=axis, fraction=0.046, label=label)
    axes[0, 0].set_xlabel("pupil x (m)")
    axes[0, 0].set_ylabel("pupil y (m)")
    for axis in axes[0, 1:]:
        axis.set_xlabel("detector x (pixel)")
        axis.set_ylabel("detector y (pixel)")
    flux_map = np.full((lenslets, lenslets), np.nan)
    flux_map[valid] = detector
    artist = axes[1, 0].imshow(flux_map, origin="lower", cmap="viridis")
    axes[1, 0].set_title("subaperture detector flux / flat")
    figure.colorbar(artist, ax=axes[1, 0], fraction=0.046, label="relative flux")
    axes[1, 0].set_xlabel("lenslet x")
    axes[1, 0].set_ylabel("lenslet y")
    axes[1, 1].scatter(truth[valid], detector, s=12, label="no filter")
    axes[1, 1].scatter(truth[valid], behind_filter, s=12, label="lambda/d spatial filter")
    axes[1, 1].set_xlabel("pupil intensity over the subaperture (relative)")
    axes[1, 1].set_ylabel("detector flux / flat")
    axes[1, 1].legend()
    axes[1, 1].set_title("flux as a scintillation measurement")
    axes[1, 2].scatter(spread[0] * 1e3, change[0] * 1e3, s=12, label="x")
    axes[1, 2].scatter(spread[1] * 1e3, change[1] * 1e3, s=12, label="y")
    axes[1, 2].set_xlabel("CoG slope without amplitude (mpx)")
    axes[1, 2].set_ylabel("change from scintillation (mpx)")
    axes[1, 2].legend()
    axes[1, 2].set_title("slopes (illustrative CoG)")
    figure.suptitle(
        f"Single layer at {args.altitude_m / 1e3:.0f} km, r0 = {args.r0_m} m at 500 nm, "
        f"sensing at {wavelength * 1e9:.0f} nm (sigma_I^2 = {scintillation_index:.2f})"
    )
    figure.tight_layout()
    figure.savefig(args.output, dpi=100)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
