"""Compare the fast LGS kernel mode with the range/angle quadrature model.

For a fixed beacon geometry, ``source.lgs_model = "kernel"`` replaces the
incoherent sum over source-angle and sodium-range states by one convolution per
subaperture. This script renders the same instrument both ways and reports how
far the images, centroids and captured flux differ, and how long a warm frame
takes. Two quadrature references are used:

``quadrature``
    The configured model as shipped: three-point Gauss-Hermite source extent
    times the configured sodium ranges. Differences include that quadrature's
    own discretization of the Gaussian source.
``dense``
    The same sodium ranges with the Gaussian source extent sampled densely
    through ``angular_kernel_path`` (``--dense-step`` sigma spacing out to four
    sigma). This isolates the kernel approximation itself.

Centroids are a validation-only observable here; makewfs exposes no centroider.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

import makewfs
from makewfs.config import WFSConfig

ROOT = Path(__file__).parents[1]
DEFAULT_QUADRATURE = ROOT / "benchmarks" / "configs" / "shack_hartmann_20x20_lgs_quadrature.toml"


def _centroids(image: np.ndarray, lenslets: int, pixels: int) -> tuple[np.ndarray, np.ndarray]:
    spots = image.reshape(lenslets, pixels, lenslets, pixels).transpose(0, 2, 1, 3)
    coordinate = np.arange(pixels) - (pixels - 1) / 2
    flux = spots.sum(axis=(2, 3))
    safe = np.where(flux > 0, flux, 1.0)
    x = (spots * coordinate[None, None, None, :]).sum(axis=(2, 3)) / safe
    y = (spots * coordinate[None, None, :, None]).sum(axis=(2, 3)) / safe
    return x, y


def _warm_render(
    sensor: makewfs.WavefrontSensor, opd: np.ndarray, frames: int
) -> tuple[Any, float]:
    image = sensor.engine.render(opd)
    start = perf_counter()
    for _ in range(frames):
        image = sensor.engine.render(opd)
    return image, (perf_counter() - start) / frames


def _dense_reference(config: WFSConfig, directory: Path, step_sigma: float) -> WFSConfig:
    fwhm = config.source.angular_fwhm_arcsec
    if fwhm <= 0.0:
        return config
    sigma = fwhm / 2.3548200450309493
    nodes = np.arange(-4.0, 4.0 + 1e-9, step_sigma) * sigma
    x, y = np.meshgrid(nodes, nodes)
    weight = np.exp(-0.5 * (x**2 + y**2) / sigma**2)
    path = directory / "dense_gaussian_source.txt"
    np.savetxt(path, np.c_[x.ravel(), y.ravel(), weight.ravel()])
    return replace(
        config,
        source=replace(config.source, angular_fwhm_arcsec=0.0, angular_kernel_path=str(path)),
    )


def _compare(
    name: str,
    reference: makewfs.WavefrontSensor,
    kernel: makewfs.WavefrontSensor,
    opd: np.ndarray,
    frames: int,
) -> dict[str, Any]:
    settings = reference.config.shack_hartmann
    assert settings is not None
    expected, reference_s = _warm_render(reference, opd, frames)
    actual, kernel_s = _warm_render(kernel, opd, frames)
    expected_rate = np.asarray(expected.photon_rate)
    actual_rate = np.asarray(actual.photon_rate)
    valid = np.asarray(reference.valid_subapertures())
    expected_x, expected_y = _centroids(
        expected_rate, settings.lenslets_across_pupil, settings.pixels_per_subaperture
    )
    actual_x, actual_y = _centroids(
        actual_rate, settings.lenslets_across_pupil, settings.pixels_per_subaperture
    )
    centroid_error = np.hypot(actual_x - expected_x, actual_y - expected_y)[valid]
    return {
        "reference": name,
        "reference_states": len(reference.engine.source_states),
        "kernel_states": len(kernel.engine.source_states),
        "image_l1_relative": float(np.abs(actual_rate - expected_rate).sum() / expected_rate.sum()),
        "image_peak_relative": float(
            np.abs(actual_rate - expected_rate).max() / expected_rate.max()
        ),
        "centroid_error_max_px": float(centroid_error.max()),
        "centroid_error_rms_px": float(np.sqrt(np.mean(centroid_error**2))),
        "captured_fraction_reference": float(
            expected.captured_rate_per_s / expected.launched_rate_per_s
        ),
        "captured_fraction_kernel": float(actual.captured_rate_per_s / actual.launched_rate_per_s),
        "warm_render_reference_s": reference_s,
        "warm_render_kernel_s": kernel_s,
        "speedup": reference_s / kernel_s,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_QUADRATURE)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--dense-step", type=float, default=0.5)
    parser.add_argument(
        "--skip-dense",
        action="store_true",
        help="compare only against the configured quadrature (fast)",
    )
    parser.add_argument("--output", type=Path, default=Path("lgs-kernel-validation.json"))
    args = parser.parse_args()
    config = makewfs.load_config(args.config)
    config = replace(
        config,
        numerics=replace(config.numerics, dtype=args.dtype),
        source=replace(config.source, lgs_model="quadrature"),
    )
    kernel_config = replace(config, source=replace(config.source, lgs_model="kernel"))
    rng = np.random.default_rng(19)
    # A modest residual: 30 nm RMS white OPD on the input grid.
    opd = rng.normal(0.0, 30e-9, config.input.shape)
    kernel = makewfs.WavefrontSensor(kernel_config)
    comparisons = [
        _compare("quadrature", makewfs.WavefrontSensor(config), kernel, opd, args.frames)
    ]
    if not args.skip_dense:
        with tempfile.TemporaryDirectory() as directory:
            dense = _dense_reference(config, Path(directory), args.dense_step)
            comparisons.append(_compare("dense", makewfs.WavefrontSensor(dense), kernel, opd, 1))
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "makewfs_version": makewfs.__version__,
        "config": str(args.config),
        "dtype": args.dtype,
        "lgs_kernel_radius_px": kernel.engine.lgs_kernel_radius_samples
        / config.numerics.fft_oversampling,
        "lgs_kernel_margin_px": kernel.engine.lgs_kernel_margin_samples
        / config.numerics.fft_oversampling,
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
