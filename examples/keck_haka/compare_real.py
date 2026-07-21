"""Compare the real Keck II HAKA OCAM2K cube with an unscaled simulation.

The real cube is raw RTC telemetry without a matched dark. Its bias is estimated
outside the pupil for each of the eight outputs and each position of the repeated
4x4 lenslet cell. Relative amplifier response is inferred from fully illuminated
non-seam subapertures. The simulation uses source throughput 1.0 and is never
globally rescaled to the observation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from simulate import (
    KECK_OCAM_AMPLIFIER_BOUNDARIES_X_PX,
    KECK_OCAM_AMPLIFIER_BOUNDARIES_Y_PX,
    KECK_OCAM_AMPLIFIER_GAIN_FACTORS,
    KECK_OCAM_AMPLIFIER_LAYOUT,
    KECK_OCAM_AMPLIFIER_OFFSETS_ADU,
    CameraMode,
    configured_sensor,
    make_keck_pupil,
)

import makewfs

HERE = Path(__file__).resolve().parent
DEFAULT_CUBE = HERE / "ocam_20260720" / "ocam_k2_20260720T113551_raw.npz"
TARGET_NAME = "eng519"
TARGET_RA_ICRS = "21 27 41.910"
TARGET_DEC_ICRS = "+15 18 23.00"
TARGET_EPOCH = 2000.0
TARGET_V_MAG = 10.16
TARGET_B_MINUS_V = 0.46


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-cube", default=str(DEFAULT_CUBE))
    parser.add_argument("--output", default=str(HERE / "real_vs_simulation.png"))
    parser.add_argument("--comparison-gif", default=str(HERE / "real_vs_simulation.gif"))
    parser.add_argument(
        "--skip-animation", action="store_true", help="write only the diagnostic PNG"
    )
    parser.add_argument("--manifest", help="default: output path with .json suffix")
    parser.add_argument("--magnitude", type=float, default=TARGET_V_MAG)
    parser.add_argument("--em-gain", type=float, default=600.0)
    parser.add_argument("--frame-rate", type=float, default=750.0)
    parser.add_argument("--telemetry-decimation", type=int, default=10)
    parser.add_argument(
        "--simulated-frames",
        type=int,
        default=750,
        help="decimated simulated frames (default matches all 750 reference frames)",
    )
    parser.add_argument("--samples-per-exposure", type=int, default=1)
    parser.add_argument("--master-dark-frames", type=int, default=32)
    parser.add_argument("--seeing", type=float, default=0.65)
    parser.add_argument("--seed", type=int, default=57)
    parser.add_argument("--atmosphere-engine", choices=("extrude", "spectral"), default="extrude")
    parser.add_argument("--playback-fps", type=int, default=25)
    parser.add_argument(
        "--animation-frames",
        type=int,
        default=150,
        help=(
            "paired decimated GIF frames (default: 150 = 2 s of 750 Hz data; "
            "metrics still use every simulated/reference frame)"
        ),
    )
    return parser.parse_args()


def _amplifier_edges() -> tuple[tuple[int, ...], tuple[int, ...]]:
    return (
        (0, *KECK_OCAM_AMPLIFIER_BOUNDARIES_Y_PX, 228),
        (0, *KECK_OCAM_AMPLIFIER_BOUNDARIES_X_PX, 228),
    )


def _amplifier_background(
    images: NDArray[np.uint16], pupil: NDArray[np.float32]
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Estimate each output's pedestal without using illuminated pixels.

    A separate histogram mode is measured for every amplifier and global 4x4
    lenslet phase. This captures the visible output offsets without letting the
    guide-star spots pull the pedestal estimate upward.
    """
    y_edges, x_edges = _amplifier_edges()
    outside_pupil = pupil < 0.01
    background = np.empty((228, 228), dtype=np.float64)
    output_modes: list[list[int]] = []
    phase_modes: list[list[list[list[int]]]] = []
    temporal_rms: list[list[float]] = []
    for row in range(KECK_OCAM_AMPLIFIER_LAYOUT[0]):
        row_modes: list[int] = []
        row_phase_modes: list[list[list[int]]] = []
        row_rms: list[float] = []
        for column in range(KECK_OCAM_AMPLIFIER_LAYOUT[1]):
            y0, y1 = y_edges[row : row + 2]
            x0, x1 = x_edges[column : column + 2]
            block = images[:, y0:y1, x0:x1]
            dark_pixels = outside_pupil[y0:y1, x0:x1]
            raw_dark = block[:, dark_pixels]
            row_modes.append(int(np.argmax(np.bincount(raw_dark.ravel()))))
            row_rms.append(float(np.median(np.std(raw_dark, axis=0, dtype=np.float64))))
            output_phase_modes: list[list[int]] = []
            for global_y_phase in range(4):
                phase_row: list[int] = []
                local_y_phase = (global_y_phase - y0) % 4
                for global_x_phase in range(4):
                    local_x_phase = (global_x_phase - x0) % 4
                    values = block[:, local_y_phase::4, local_x_phase::4][
                        :, dark_pixels[local_y_phase::4, local_x_phase::4]
                    ].ravel()
                    mode = int(np.argmax(np.bincount(values)))
                    phase_row.append(mode)
                    background[
                        y0 + local_y_phase : y1 : 4,
                        x0 + local_x_phase : x1 : 4,
                    ] = mode
                output_phase_modes.append(phase_row)
            row_phase_modes.append(output_phase_modes)
        output_modes.append(row_modes)
        phase_modes.append(row_phase_modes)
        temporal_rms.append(row_rms)
    return background, {
        "layout_rows_columns": list(KECK_OCAM_AMPLIFIER_LAYOUT),
        "roi_boundaries_y_px": list(KECK_OCAM_AMPLIFIER_BOUNDARIES_Y_PX),
        "roi_boundaries_x_px": list(KECK_OCAM_AMPLIFIER_BOUNDARIES_X_PX),
        "raw_pedestal_modes_adu": output_modes,
        "outside_pupil_temporal_rms_adu": temporal_rms,
        "phase_pedestal_modes_adu": phase_modes,
    }


def _amplifier_response(
    mean_image: NDArray[np.float64], pupil: NDArray[np.float32]
) -> dict[str, Any]:
    """Infer relative output response from fully illuminated, non-seam cells."""
    cells = mean_image.reshape(57, 4, 57, 4).transpose(0, 2, 1, 3)
    signal = np.sum(cells, axis=(-1, -2))
    pupil_cells = pupil.reshape(57, 4, 57, 4).transpose(0, 2, 1, 3)
    illuminated_fraction = np.mean(pupil_cells, axis=(-1, -2))
    y_edges, x_edges = _amplifier_edges()
    medians: list[float] = []
    cell_counts: list[int] = []
    for row in range(KECK_OCAM_AMPLIFIER_LAYOUT[0]):
        for column in range(KECK_OCAM_AMPLIFIER_LAYOUT[1]):
            values: list[float] = []
            for subap_y in range(57):
                pixel_y0 = 4 * subap_y
                pixel_y1 = pixel_y0 + 4
                if pixel_y0 < y_edges[row] or pixel_y1 > y_edges[row + 1]:
                    continue
                for subap_x in range(57):
                    pixel_x0 = 4 * subap_x
                    pixel_x1 = pixel_x0 + 4
                    if pixel_x0 < x_edges[column] or pixel_x1 > x_edges[column + 1]:
                        continue
                    if illuminated_fraction[subap_y, subap_x] >= 0.98:
                        values.append(float(signal[subap_y, subap_x]))
            if not values:
                raise ValueError("no fully illuminated cells available for amplifier fit")
            medians.append(float(np.median(values)))
            cell_counts.append(len(values))
    # Normalize ADU/e- response to arithmetic mean one. This fits only relative
    # output response and cannot change the global source photon normalization.
    response = np.asarray(medians) / np.mean(medians)
    conversion_gain_factors = 1.0 / response
    return {
        "method": (
            "median dark-subtracted signal in >=98%-illuminated 4x4 cells wholly "
            "inside one amplifier; ADU/e- responses normalized to arithmetic mean one"
        ),
        "row_major_cell_counts": cell_counts,
        "row_major_median_signal_adu": medians,
        "row_major_relative_adu_response": response.tolist(),
        "row_major_relative_e_per_adu": conversion_gain_factors.tolist(),
        "configured_relative_e_per_adu": list(KECK_OCAM_AMPLIFIER_GAIN_FACTORS),
    }


def _subaperture_contrast(mean_image: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return a pedestal-insensitive 2x2-core-minus-border signal per lenslet."""
    cells = mean_image.reshape(57, 4, 57, 4).transpose(0, 2, 1, 3)
    border = np.ones((4, 4), dtype=np.bool_)
    border[1:3, 1:3] = False
    core = np.sum(cells[..., 1:3, 1:3], axis=(-1, -2))
    return np.asarray(core - 4.0 * np.mean(cells[..., border], axis=-1))


def _mean_spot(mean_image: NDArray[np.float64]) -> NDArray[np.float64]:
    """Aggregate all lenslet cells and remove their mean border pedestal."""
    cells = mean_image.reshape(57, 4, 57, 4).transpose(0, 2, 1, 3)
    spot = np.mean(cells, axis=(0, 1))
    border = np.ones((4, 4), dtype=np.bool_)
    border[1:3, 1:3] = False
    spot -= np.mean(spot[border])
    return np.clip(spot, 0.0, None)


def _centroid(image: NDArray[np.float64]) -> tuple[float, float]:
    total = float(np.sum(image))
    if total <= 0:
        return (float("nan"), float("nan"))
    yy, xx = np.indices(image.shape, dtype=np.float64)
    return (float(np.sum(xx * image) / total), float(np.sum(yy * image) / total))


def _cube_metrics(cube: NDArray[np.float32]) -> dict[str, Any]:
    mean_image = np.mean(cube, axis=0, dtype=np.float64)
    sums = np.sum(cube, axis=(1, 2), dtype=np.float64)
    peaks = np.max(cube, axis=(1, 2))
    temporal_rms = np.std(cube, axis=0, dtype=np.float64)
    low_signal = mean_image <= np.percentile(mean_image, 25.0)
    spot = _mean_spot(mean_image)
    centroid_x, centroid_y = _centroid(spot)
    normalized_spot = spot / np.sum(spot)
    return {
        "frames": int(cube.shape[0]),
        "mean_signal_counts_per_frame": float(np.mean(sums)),
        "std_signal_counts_per_frame": float(np.std(sums)),
        "median_peak_counts": float(np.median(peaks)),
        "peak_count_percentiles_10_50_90": [
            float(value) for value in np.percentile(peaks, [10.0, 50.0, 90.0])
        ],
        "lower_quartile_pixel_temporal_rms_counts": float(np.median(temporal_rms[low_signal])),
        "mean_spot_centroid_xy_pixels": [centroid_x, centroid_y],
        "mean_spot_central_2x2_fraction": float(np.sum(normalized_spot[1:3, 1:3])),
        "mean_spot_normalized": normalized_spot.tolist(),
    }


def _load_real(path: Path) -> tuple[NDArray[np.float32], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        images = np.asarray(archive["images"])
        timestamps = np.asarray(archive["timestamps"], dtype=np.int64)
        counters = np.asarray(archive["frame_counters"], dtype=np.int64)
    if images.ndim != 3 or images.shape[1:] != (228, 228) or images.dtype != np.uint16:
        raise ValueError("real OCAM cube must be uint16 with shape (frames, 228, 228)")
    if len(images) < 2 or not np.all(np.diff(counters) == 10):
        raise ValueError("reference cube must contain every tenth OCAM frame")
    pupil = make_keck_pupil()
    background, amplifier_diagnostics = _amplifier_background(images, pupil)
    reduced = images.astype(np.float32) - background.astype(np.float32)
    amplifier_diagnostics["configured_offsets_adu"] = list(KECK_OCAM_AMPLIFIER_OFFSETS_ADU)
    amplifier_diagnostics["relative_response_fit"] = _amplifier_response(
        np.mean(reduced, axis=0, dtype=np.float64), pupil
    )
    resolved_path = path.resolve()
    try:
        reported_path = str(resolved_path.relative_to(HERE))
    except ValueError:
        reported_path = str(resolved_path)
    metadata = {
        "path": reported_path,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "raw_frames": len(images),
        "raw_dtype": str(images.dtype),
        "raw_min_counts": int(images.min()),
        "raw_max_counts": int(images.max()),
        "telemetry_cadence_hz": float(1e9 / np.median(np.diff(timestamps))),
        "frame_counter_step": 10,
        "background_method": (
            "integer histogram mode outside the pupil, independently for each "
            "amplifier and global position in the repeated 4x4 cell"
        ),
        "amplifier_diagnostics": amplifier_diagnostics,
        "matched_dark_available": False,
    }
    return reduced, metadata


def _simulate(args: argparse.Namespace) -> tuple[NDArray[np.float32], dict[str, Any]]:
    try:
        import getframes
        import pyturb
    except ImportError as exc:
        raise SystemExit("install makewfs[examples,interop] to run this comparison") from exc

    base = makewfs.load_config(HERE / "keck_haka.toml")
    base = replace(
        base,
        source=replace(
            base.source,
            magnitude=args.magnitude,
            magnitude_system="vega",
            band="V",
        ),
    )
    if base.source.throughput != 1.0:
        raise ValueError("comparison requires source.throughput = 1.0; no flux scaling is allowed")
    mode = CameraMode(
        watao=10,
        min_magnitude=args.magnitude,
        max_magnitude=np.nextafter(args.magnitude, np.inf),
        em_gain=args.em_gain,
        frame_rate_1_hz=args.frame_rate,
        frame_rate_2_hz=args.frame_rate,
        filter_name="open",
        background_mode=1,
    )
    pupil = make_keck_pupil()
    exposure_s = 1.0 / args.frame_rate
    atmosphere = pyturb.Atmosphere.from_profile(
        "mauna-kea",
        seeing=args.seeing,
        diameter=base.input.grid_extent_m,
        n=base.input.shape[0],
        seed=args.seed,
        engine=args.atmosphere_engine,
    )
    frames: list[NDArray[np.float32]] = []
    with tempfile.TemporaryDirectory(prefix="makewfs-haka-comparison-") as temporary:
        pupil_path = Path(temporary) / "keck_primary.npy"
        np.save(pupil_path, pupil)
        sensor = configured_sensor(
            base,
            magnitude=args.magnitude,
            mode=mode,
            frame_rate_column="WSFRRT1",
            pupil_path=pupil_path,
        )
        master_dark = sensor.detector.camera.master_dark(
            exposure_s,
            args.master_dark_frames,
            sensor.config.detector.temperature_c,
            seed=args.seed * 1000,
            method="median",
        )
        last_frame: Any = None
        for index in range(args.simulated_frames):
            samples = [
                atmosphere.evolve(exposure_s / args.samples_per_exposure)
                for _ in range(args.samples_per_exposure)
            ]
            last_frame = sensor.expose_integrated(samples, seed=args.seed * 10_000 + index)
            reduced = getframes.calibrate(last_frame, dark=master_dark)
            frames.append(np.asarray(getframes.to_numpy(reduced.data), dtype=np.float32))
            skipped_time_s = (args.telemetry_decimation - 1) * exposure_s
            if skipped_time_s > 0:
                atmosphere.evolve(skipped_time_s)
        if last_frame is None or last_frame.truth is None:
            raise AssertionError("simulation did not produce detector truth")
        camera = sensor.detector.camera.config
        simulation_metadata = {
            "throughput": base.source.throughput,
            "atmosphere_profile": "mauna-kea",
            "pyturb_engine": args.atmosphere_engine,
            "wavefront_source": ("seeded generated pyturb phase screens evolved by frozen flow"),
            "seeing_arcsec_at_500_nm": args.seeing,
            "sensing_wavelength_nm": 673.0,
            "magnitude_system": "Vega V",
            "magnitude": args.magnitude,
            "target": {
                "name": TARGET_NAME,
                "ra_icrs": TARGET_RA_ICRS,
                "dec_icrs": TARGET_DEC_ICRS,
                "epoch": TARGET_EPOCH,
                "v_magnitude": TARGET_V_MAG,
                "b_minus_v": TARGET_B_MINUS_V,
                "lgs_flag": 1,
            },
            "em_gain": args.em_gain,
            "frame_rate_hz": args.frame_rate,
            "exposure_s": exposure_s,
            "telemetry_decimation": args.telemetry_decimation,
            "master_dark_frames": args.master_dark_frames,
            "image_full_well_e": camera.full_well_e,
            "output_full_well_e": camera.output_full_well_e,
            "gain_e_per_adu": camera.gain_e_per_adu,
            "amplifier_layout_rows_columns": list(camera.amplifier_layout),
            "amplifier_boundaries_y_px": list(camera.amplifier_boundaries_y_px),
            "amplifier_boundaries_x_px": list(camera.amplifier_boundaries_x_px),
            "amplifier_gain_factors_relative_e_per_adu": list(camera.amplifier_gain_factors or ()),
            "amplifier_offsets_adu": list(camera.amplifier_offsets_adu or ()),
            "launched_photons_per_s": float(last_frame.metadata["wfs_launched_photons_s"]),
            "captured_photons_per_s": float(last_frame.metadata["wfs_captured_photons_s"]),
            "expected_photoelectrons_last_frame": float(
                np.sum(getframes.to_numpy(last_frame.truth.mean_photoelectrons))
            ),
        }
    return np.stack(frames), simulation_metadata


def _plot(
    output: Path,
    real: NDArray[np.float32],
    simulated: NDArray[np.float32],
    real_metrics: dict[str, Any],
    simulated_metrics: dict[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm

    real_mean = np.mean(real, axis=0, dtype=np.float64)
    simulated_mean = np.mean(simulated, axis=0, dtype=np.float64)
    real_contrast = _subaperture_contrast(real_mean)
    simulated_contrast = _subaperture_contrast(simulated_mean)
    vmax = max(float(np.percentile(real_mean, 99.9)), float(np.percentile(simulated_mean, 99.9)))
    norm = SymLogNorm(linthresh=10.0, vmin=-10.0, vmax=vmax)

    figure, axes = plt.subplots(2, 3, figsize=(15.5, 9.2), constrained_layout=True)
    for axis, image, title in (
        (axes[0, 0], real_mean, "Real RTC mean: per-output phase-mode bias subtracted"),
        (axes[0, 1], simulated_mean, "Simulation mean: master-dark subtracted"),
    ):
        artist = axis.imshow(image, origin="lower", norm=norm, interpolation="nearest")
        figure.colorbar(artist, ax=axis, label="dark/bias-subtracted count (shared scale)")
        axis.set_title(title)

    real_sums = np.sum(real, axis=(1, 2), dtype=np.float64) / 1e6
    simulated_sums = np.sum(simulated, axis=(1, 2), dtype=np.float64) / 1e6
    axes[0, 2].hist(real_sums, bins=30, alpha=0.7, label="real")
    axes[0, 2].hist(simulated_sums, bins=30, alpha=0.7, label="simulation")
    axes[0, 2].set_xlabel("signed signal sum (million count / frame)")
    axes[0, 2].set_ylabel("frames")
    axes[0, 2].legend()
    axes[0, 2].set_title("No flux rescaling")

    for axis, contrast, title in (
        (axes[1, 0], real_contrast, "Real 57x57 pupil contrast"),
        (axes[1, 1], simulated_contrast, "Simulated 57x57 pupil contrast"),
    ):
        scale = max(float(np.percentile(contrast, 95.0)), 1.0)
        artist = axis.imshow(
            contrast / scale,
            origin="lower",
            vmin=0.0,
            vmax=1.5,
            interpolation="nearest",
        )
        figure.colorbar(artist, ax=axis, label="contrast / own 95th percentile")
        axis.set_title(title)

    real_spot = _mean_spot(real_mean)
    simulated_spot = _mean_spot(simulated_mean)
    combined = np.concatenate(
        [
            real_spot / np.sum(real_spot),
            np.full((4, 1), np.nan),
            simulated_spot / np.sum(simulated_spot),
        ],
        axis=1,
    )
    artist = axes[1, 2].imshow(combined, origin="lower", interpolation="nearest")
    figure.colorbar(artist, ax=axes[1, 2], label="fraction of aggregate 4x4 signal")
    axes[1, 2].set_xticks([1.5, 6.5], ["real", "simulation"])
    axes[1, 2].set_title("Mean lenslet spot (no registration fit)")

    ratio = (
        real_metrics["mean_signal_counts_per_frame"]
        / simulated_metrics["mean_signal_counts_per_frame"]
    )
    figure.suptitle(
        "Keck II HAKA on eng519: V=10.16, B-V=0.46, EM x600, 750 Hz, open loop | "
        f"real/simulation total signal = {ratio:.3f} (reported, not applied)",
        fontsize=13,
    )
    for axis in axes.flat[:2]:
        axis.set_xlabel("x (native OCAM2K pixel)")
        axis.set_ylabel("y (native OCAM2K pixel)")
    for axis in axes[1, :2]:
        axis.set_xlabel("x subaperture")
        axis.set_ylabel("y subaperture")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def _animate_comparison(
    output: Path,
    real: NDArray[np.float32],
    simulated: NDArray[np.float32],
    *,
    frame_rate_hz: float,
    telemetry_decimation: int,
    playback_fps: int,
    animation_frames: int,
) -> None:
    """Write an unscaled, shared-color-scale real/simulation comparison GIF."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.colors import SymLogNorm

    frame_count = min(len(real), len(simulated), animation_frames)
    real = real[:frame_count]
    simulated = simulated[:frame_count]
    vmax = max(
        float(np.percentile(real, 99.95)),
        float(np.percentile(simulated, 99.95)),
        1.0,
    )
    norm = SymLogNorm(linthresh=10.0, vmin=-20.0, vmax=vmax)
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), constrained_layout=True)
    artists = [
        axes[0].imshow(real[0], origin="lower", norm=norm, interpolation="nearest"),
        axes[1].imshow(simulated[0], origin="lower", norm=norm, interpolation="nearest"),
    ]
    axes[0].set_title("Real RTC (bias subtracted)", fontsize=10)
    axes[1].set_title("Simulation (dark subtracted)", fontsize=10)
    for axis in axes:
        axis.set_xlabel("x (native OCAM2K pixel)")
        axis.set_ylabel("y (native OCAM2K pixel)")
    figure.colorbar(artists[1], ax=axes, label="dark/bias-subtracted count (shared, unscaled)")
    title = figure.suptitle("")

    def update(index: int) -> tuple[Any, ...]:
        artists[0].set_data(real[index])
        artists[1].set_data(simulated[index])
        elapsed_s = index * telemetry_decimation / frame_rate_hz
        title.set_text(
            f"eng519 | V={TARGET_V_MAG:.2f}, B-V={TARGET_B_MINUS_V:.2f} | "
            f"750 Hz, every 10th frame | t={elapsed_s:.3f} s"
        )
        return (*artists, title)

    output.parent.mkdir(parents=True, exist_ok=True)
    animation = FuncAnimation(
        figure,
        update,
        frames=frame_count,
        interval=1000.0 / playback_fps,
        blit=False,
    )
    animation.save(output, writer=PillowWriter(fps=playback_fps))
    plt.close(figure)


def main() -> None:
    args = _arguments()
    if (
        args.simulated_frames < 1
        or args.samples_per_exposure < 1
        or args.master_dark_frames < 1
        or args.telemetry_decimation < 1
        or args.frame_rate <= 0
        or args.playback_fps < 1
        or args.animation_frames < 1
        or not 1 <= args.em_gain <= 600
    ):
        raise SystemExit("invalid positive frame/sample counts, frame rate, or EM gain")
    real_path = Path(args.real_cube)
    real, real_metadata = _load_real(real_path)
    simulated, simulation_metadata = _simulate(args)
    real_metrics = _cube_metrics(real)
    simulated_metrics = _cube_metrics(simulated)
    flux_ratio = (
        real_metrics["mean_signal_counts_per_frame"]
        / simulated_metrics["mean_signal_counts_per_frame"]
    )

    output = Path(args.output).resolve()
    _plot(output, real, simulated, real_metrics, simulated_metrics)
    animation_output: Path | None = None
    if not args.skip_animation:
        animation_output = Path(args.comparison_gif).resolve()
        _animate_comparison(
            animation_output,
            real,
            simulated,
            frame_rate_hz=args.frame_rate,
            telemetry_decimation=args.telemetry_decimation,
            playback_fps=args.playback_fps,
            animation_frames=args.animation_frames,
        )
    manifest = Path(args.manifest).resolve() if args.manifest else output.with_suffix(".json")
    payload = {
        "comparison_image": output.name,
        "comparison_animation": (animation_output.name if animation_output is not None else None),
        "comparison_animation_frames": (
            min(len(real), len(simulated), args.animation_frames)
            if animation_output is not None
            else 0
        ),
        "reference": real_metadata,
        "simulation": simulation_metadata,
        "real_metrics": real_metrics,
        "simulated_metrics": simulated_metrics,
        "diagnostics_not_applied_to_simulation": {
            "real_to_simulated_total_signal_ratio": flux_ratio,
            "interpretation": (
                "This ratio combines unmodeled instrument transmission, photometric-band and "
                "detector-calibration uncertainty; it is reported but never used as a scale factor."
            ),
        },
        "known_limitations": [
            (
                "No matched real OCAM dark was supplied; an outside-pupil, per-output, "
                "repeated-4x4 phase-mode pedestal is used."
            ),
            (
                "The real pupil has static illumination/vignetting structure absent "
                "from phase-only pyturb."
            ),
            "No image registration, spot shift, pupil rotation, or flux normalization is fitted.",
            (
                "The catalog V=10.16 sets the photon budget while the morphology is "
                "monochromatic at 673 nm; B-V=0.46 is recorded but is not used to "
                "invent an unmeasured open-filter throughput curve."
            ),
        ],
    }
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {output}, {manifest}"
        + (f", and {animation_output}" if animation_output is not None else "")
        + f"; real/simulation signal ratio={flux_ratio:.3f} "
        "(diagnostic only)"
    )


if __name__ == "__main__":
    main()
