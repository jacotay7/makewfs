"""Fit the Keck secondary shadow to the supplied live HAKA pupil image.

The fit uses only relative subaperture illumination. It removes the measured
eight-output response, normalizes in an illuminated annulus, and fits the union
of a circle and a segment-aligned regular hexagon. No simulated detector flux is
used and no result from this fit scales the science images.
"""

from __future__ import annotations

import argparse
import json
import math
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from compare_real import DEFAULT_CUBE, _load_real, _subaperture_contrast
from numpy.typing import NDArray
from scipy.optimize import differential_evolution
from simulate import (
    HAKA_GRID_EXTENT_M,
    KECK_OCAM_AMPLIFIER_GAIN_FACTORS,
    KECK_OCAM_AMPLIFIER_LAYOUT,
    KECK_SECONDARY_CIRCLE_RADIUS_M,
    KECK_SECONDARY_HEX_CIRCUMRADIUS_M,
    KECK_SECONDARY_OFFSET_X_M,
    KECK_SECONDARY_OFFSET_Y_M,
    _inside_hexagon,
    make_keck_pupil,
)

HERE = Path(__file__).resolve().parent
FIT_SAMPLES_PER_SUBAPERTURE = 8


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-cube", default=str(DEFAULT_CUBE))
    parser.add_argument("--output", default=str(HERE / "secondary_fit.png"))
    parser.add_argument("--manifest", help="default: output path with .json suffix")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--maxiter", type=int, default=80)
    parser.add_argument("--population-size", type=int, default=12)
    return parser.parse_args()


def _apply_amplifier_response(cube: NDArray[np.float32]) -> NDArray[np.float64]:
    """Convert each output to a common relative electron/count response."""
    corrected = cube.astype(np.float64, copy=True)
    y_edges = (0, 54, 114, 174, 228)
    x_edges = (0, 114, 228)
    factors = np.asarray(KECK_OCAM_AMPLIFIER_GAIN_FACTORS).reshape(KECK_OCAM_AMPLIFIER_LAYOUT)
    for row, (y0, y1) in enumerate(pairwise(y_edges)):
        for column, (x0, x1) in enumerate(pairwise(x_edges)):
            corrected[:, y0:y1, x0:x1] *= factors[row, column]
    return corrected


def _measured_illumination(cube: NDArray[np.float32]) -> NDArray[np.float64]:
    corrected = _apply_amplifier_response(cube)
    contrast = _subaperture_contrast(np.mean(corrected, axis=0, dtype=np.float64))
    yy, xx = np.indices((57, 57), dtype=np.float64)
    radius = np.hypot(xx - 28.0, yy - 28.0)
    reference = (radius >= 9.0) & (radius <= 13.0)
    scale = float(np.median(contrast[reference]))
    if scale <= 0:
        raise ValueError("live cube has no positive illuminated-annulus signal")
    return np.clip(contrast / scale, 0.0, 1.25)


def _model_illumination(parameters: NDArray[np.float64]) -> NDArray[np.float64]:
    circle_radius_m, hex_circumradius_m, offset_x_m, offset_y_m = parameters
    samples = FIT_SAMPLES_PER_SUBAPERTURE
    size = 57 * samples
    # Primary edge, segment gaps, and support arms are fixed. Only the secondary
    # shadow is fitted.
    clear = make_keck_pupil(
        shape=(size, size),
        central_obscuration_diameter_m=0.0,
        central_hex_circumradius_m=0.0,
        supersampling=1,
    ).astype(np.float64)
    coordinate = ((np.arange(size, dtype=np.float64) + 0.5) / size - 0.5) * HAKA_GRID_EXTENT_M
    xx, yy = np.meshgrid(coordinate, coordinate)
    xx -= offset_x_m
    yy -= offset_y_m
    shadow = (np.hypot(xx, yy) <= circle_radius_m) | _inside_hexagon(xx, yy, hex_circumradius_m)
    clear[shadow] = 0.0
    return clear.reshape(57, samples, 57, samples).mean(axis=(1, 3))


def _loss(parameters: NDArray[np.float64], measured: NDArray[np.float64]) -> float:
    model = _model_illumination(parameters)
    yy, xx = np.indices((57, 57), dtype=np.float64)
    radius = np.hypot(xx - 28.0, yy - 28.0)
    # The central 20-lenslet diameter contains the secondary boundary while
    # excluding the primary rim. A clipped quadratic reduces sensitivity to
    # individual hot/weak lenslets without changing the fitted geometry.
    selection = radius <= 10.0
    residual = model[selection] - measured[selection]
    clipped = np.clip(residual, -0.35, 0.35)
    return float(np.mean(clipped * clipped))


def _plot(
    output: Path,
    measured: NDArray[np.float64],
    model: NDArray[np.float64],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    residual = measured - model
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 4.2), constrained_layout=True)
    first = axes[0].imshow(measured, origin="lower", vmin=0.0, vmax=1.15)
    axes[0].set_title("Live relative illumination")
    axes[1].imshow(model, origin="lower", vmin=0.0, vmax=1.15)
    axes[1].set_title("Fitted circle plus hexagon")
    third = axes[2].imshow(residual, origin="lower", cmap="coolwarm", vmin=-0.5, vmax=0.5)
    axes[2].set_title("Live - model")
    figure.colorbar(first, ax=axes[:2], label="relative subaperture illumination")
    figure.colorbar(third, ax=axes[2], label="relative residual")
    for axis in axes:
        axis.set_xlabel("x subaperture")
        axis.set_ylabel("y subaperture")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> None:
    args = _arguments()
    if args.maxiter < 1 or args.population_size < 4:
        raise SystemExit("--maxiter must be positive and --population-size must be >= 4")
    real, real_metadata = _load_real(Path(args.real_cube))
    measured = _measured_illumination(real)
    result = differential_evolution(
        _loss,
        bounds=((1.15, 1.38), (1.30, 1.58), (-0.12, 0.12), (-0.12, 0.12)),
        args=(measured,),
        seed=args.seed,
        maxiter=args.maxiter,
        popsize=args.population_size,
        polish=True,
        workers=1,
        updating="immediate",
        tol=2e-4,
    )
    parameters = np.asarray(result.x, dtype=np.float64)
    model = _model_illumination(parameters)
    output = Path(args.output).resolve()
    _plot(output, measured, model)
    manifest = Path(args.manifest).resolve() if args.manifest else output.with_suffix(".json")
    fitted = {
        "circle_radius_m": float(parameters[0]),
        "circle_diameter_m": float(2.0 * parameters[0]),
        "hex_circumradius_m": float(parameters[1]),
        "hex_flat_to_flat_m": float(math.sqrt(3.0) * parameters[1]),
        "hex_corner_to_corner_m": float(2.0 * parameters[1]),
        "offset_x_m": float(parameters[2]),
        "offset_y_m": float(parameters[3]),
        "rotation_deg": 0.0,
    }
    configured = {
        "circle_radius_m": KECK_SECONDARY_CIRCLE_RADIUS_M,
        "hex_circumradius_m": KECK_SECONDARY_HEX_CIRCUMRADIUS_M,
        "offset_x_m": KECK_SECONDARY_OFFSET_X_M,
        "offset_y_m": KECK_SECONDARY_OFFSET_Y_M,
    }
    payload: dict[str, Any] = {
        "fit_image": output.name,
        "reference_cube": {
            "path": real_metadata["path"],
            "sha256": real_metadata["sha256"],
        },
        "method": (
            "amplifier-response-corrected 4x4 core-minus-border illumination; "
            "robust least-squares fit within 10 lenslets of pupil center"
        ),
        "geometry": "union of circle and pointy-top segment-aligned regular hexagon",
        "samples_per_subaperture": FIT_SAMPLES_PER_SUBAPERTURE,
        "objective": float(result.fun),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "fitted": fitted,
        "configured_model": configured,
        "configured_parameter_max_abs_difference_m": float(
            np.max(
                np.abs(
                    parameters
                    - np.asarray(
                        [
                            KECK_SECONDARY_CIRCLE_RADIUS_M,
                            KECK_SECONDARY_HEX_CIRCUMRADIUS_M,
                            KECK_SECONDARY_OFFSET_X_M,
                            KECK_SECONDARY_OFFSET_Y_M,
                        ]
                    )
                )
            )
        ),
        "fit_does_not_scale_simulated_flux": True,
    }
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output} and {manifest}; objective={result.fun:.8f}")


if __name__ == "__main__":
    main()
