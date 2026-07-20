"""Produce deterministic optical validation metrics and optional plots."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

import makewfs


def _spot_centroids(image: np.ndarray, lenslets: int, pixels: int) -> tuple[np.ndarray, np.ndarray]:
    """Return per-subaperture x/y centroids for a validation-only SH observable."""
    spots = image.reshape(lenslets, pixels, lenslets, pixels).transpose(0, 2, 1, 3)
    yy, xx = np.indices((pixels, pixels), dtype=np.float64)
    flux = np.sum(spots, axis=(2, 3))
    safe_flux = np.where(flux > 0, flux, 1.0)
    centroid_x = np.sum(spots * xx, axis=(2, 3)) / safe_flux
    centroid_y = np.sum(spots * yy, axis=(2, 3)) / safe_flux
    return centroid_x, centroid_y


def _metrics(config_path: Path) -> dict[str, Any]:
    config = makewfs.load_config(config_path)
    sensor = makewfs.WavefrontSensor(config)
    zero = np.zeros(config.input.shape, dtype=np.float64)
    start = perf_counter()
    reference = sensor.photon_rate(zero)
    render_s = perf_counter() - start
    piston = sensor.photon_rate(zero + 2.0e-6)
    result: dict[str, Any] = {
        "config": str(config_path),
        "config_digest": config.digest,
        "sensor": config.sensor.kind,
        "shape": list(reference.shape),
        "source_rate_per_s": sensor.engine.source_rate,
        "reference_rate_sum_per_s": float(reference.sum()),
        "piston_relative_error": float(
            np.max(np.abs(reference - piston)) / max(float(np.max(reference)), 1e-30)
        ),
        "warm_render_s": render_s,
    }
    if config.sensor.kind == "pyramid":
        half = reference.shape[0] // 2
        quadrants = [
            reference[:half, :half],
            reference[:half, half:],
            reference[half:, :half],
            reference[half:, half:],
        ]
        result["pyramid_quadrant_relative_spread"] = float(
            np.std([float(quadrant.sum()) for quadrant in quadrants])
            / max(float(reference.sum()), 1e-30)
        )
        yy, xx = np.indices(config.input.shape, dtype=np.float64)
        xx = (
            (xx + 0.5 - config.input.shape[1] / 2)
            * config.input.grid_extent_m
            / config.input.shape[1]
        )
        yy = (
            (yy + 0.5 - config.input.shape[0] / 2)
            * config.input.grid_extent_m
            / config.input.shape[0]
        )
        modes = {
            "tip": xx / config.telescope.pupil_diameter_m,
            "tilt": yy / config.telescope.pupil_diameter_m,
            "focus": (xx**2 + yy**2) / config.telescope.pupil_diameter_m**2,
        }
        amplitude_waves = 0.02
        for name, mode in modes.items():
            plus = sensor.photon_rate(amplitude_waves * config.sensor.wavelength_m * mode)
            minus = sensor.photon_rate(-amplitude_waves * config.sensor.wavelength_m * mode)
            plus_response = plus - reference
            minus_response = minus - reference
            antisymmetry = np.linalg.norm(plus_response + minus_response) / max(
                float(np.linalg.norm(plus_response - minus_response)), 1e-30
            )
            normalized_gain = np.linalg.norm((plus - minus) / 2) / max(
                float(np.linalg.norm(reference)) * amplitude_waves, 1e-30
            )
            result[f"pyramid_{name}_push_pull_antisymmetry"] = float(antisymmetry)
            result[f"pyramid_{name}_normalized_response_per_wave"] = float(normalized_gain)
    elif config.sensor.kind == "shack_hartmann":
        assert config.shack_hartmann is not None
        settings = config.shack_hartmann
        x = (
            (np.arange(config.input.shape[1], dtype=np.float64) + 0.5 - config.input.shape[1] / 2)
            * config.input.grid_extent_m
            / config.input.shape[1]
        )
        lenslet_pitch_m = config.telescope.pupil_diameter_m / settings.lenslets_across_pupil
        tilt_cycles_per_lenslet = 0.1
        opd_slope = tilt_cycles_per_lenslet * config.sensor.wavelength_m / lenslet_pitch_m
        tilted = sensor.photon_rate(np.broadcast_to(opd_slope * x[None, :], config.input.shape))
        margin = settings.detector_margin_pixels
        base_size = settings.lenslets_across_pupil * settings.pixels_per_subaperture
        reference_core = reference[margin : margin + base_size, margin : margin + base_size]
        tilted_core = tilted[margin : margin + base_size, margin : margin + base_size]
        reference_x, reference_y = _spot_centroids(
            reference_core, settings.lenslets_across_pupil, settings.pixels_per_subaperture
        )
        tilted_x, tilted_y = _spot_centroids(
            tilted_core, settings.lenslets_across_pupil, settings.pixels_per_subaperture
        )
        reference_flux = reference_core.reshape(
            settings.lenslets_across_pupil,
            settings.pixels_per_subaperture,
            settings.lenslets_across_pupil,
            settings.pixels_per_subaperture,
        ).sum(axis=(1, 3))
        valid = np.asarray(sensor.engine.lenslet_valid) & (reference_flux > 0)
        measured = tilted_x[valid] - reference_x[valid]
        cross_axis = tilted_y[valid] - reference_y[valid]
        sampling = settings.spot_sampling_pixels_per_lambda_over_d
        if sampling is None:
            assert settings.lenslet_focal_length_m is not None
            assert settings.detector_pixel_pitch_m is not None
            sampling = (
                settings.lenslet_focal_length_m
                * config.sensor.wavelength_m
                * settings.relay_magnification
                / (lenslet_pitch_m * settings.detector_pixel_pitch_m)
            )
        expected = tilt_cycles_per_lenslet * sampling
        result.update(
            {
                "sh_tilt_valid_lenslets": int(np.count_nonzero(valid)),
                "sh_tilt_expected_pixels": float(expected),
                "sh_tilt_mean_pixels": float(np.mean(measured)),
                "sh_tilt_std_pixels": float(np.std(measured)),
                "sh_tilt_relative_scale_error": float(abs(np.mean(measured) / expected - 1.0)),
                "sh_tilt_cross_axis_max_pixels": float(np.max(np.abs(cross_axis))),
            }
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        action="append",
        type=Path,
        help="configuration to validate (repeatable; defaults to shipped SH and pyramid files)",
    )
    parser.add_argument("--output", type=Path, default=Path("validation-metrics.json"))
    parser.add_argument("--plot", type=Path, default=None)
    args = parser.parse_args()
    root = Path(__file__).parents[1]
    configs = args.config or [
        root / "examples" / "configs" / "shack_hartmann_minimal.toml",
        root / "examples" / "configs" / "pyramid_minimal.toml",
    ]
    report: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "metrics": [_metrics(path) for path in configs],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.plot is not None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise SystemExit("--plot requires matplotlib") from exc
        figure, axes = plt.subplots(1, len(configs), figsize=(5 * len(configs), 4))
        axes = np.atleast_1d(axes)
        for axis, path in zip(axes, configs):
            sensor = makewfs.WavefrontSensor.from_toml(path)
            image = sensor.photon_rate(np.zeros(sensor.config.input.shape))
            axis.imshow(image, origin="lower")
            axis.set_title(path.stem)
            axis.set_axis_off()
        figure.tight_layout()
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.plot, dpi=140)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
