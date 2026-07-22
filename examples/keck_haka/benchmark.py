"""Benchmark warm open-loop Keck II HAKA frames on CPU and GPU.

The timed path advances a non-periodic ``pyturb`` Mauna Kea atmosphere by one
physical detector exposure, forms the eight-wavelength 57x57 Shack--Hartmann
image, and exposes the 228x228 OCAM2K detector. Static source radiometry, pupil
sampling, sensor construction, atmosphere construction, and warm-up are outside
the timed interval.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shlex
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray
from simulate import (
    HAKA_DOWNSTREAM_THROUGHPUT,
    HAKA_REFERENCE_AIRMASS,
    HAKA_SOURCE_TEMPERATURE_K,
    OCAM2K_MAX_FRAME_RATE_HZ,
    configured_sensor,
    load_camera_modes,
    make_keck_pupil,
    pupil_collecting_area_m2,
    select_camera_mode,
)

import makewfs

HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ThroughputSample:
    """One cumulative warm-throughput observation."""

    elapsed_s: float
    frames: int
    frames_per_s: float


@dataclass(frozen=True)
class DeviceResult:
    """Warm end-to-end result for one execution device."""

    device: str
    frames: int
    elapsed_s: float
    frames_per_s: float
    physical_frame_rate_hz: float
    simulated_open_loop_time_s: float
    real_time_factor: float
    samples: tuple[ThroughputSample, ...]


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=5.0, help="timed seconds per device")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--warmup-frames", type=int, default=3)
    parser.add_argument("--magnitude", type=float, default=10.16)
    parser.add_argument("--frame-rate", type=float, default=750.0)
    parser.add_argument("--seeing", type=float, default=0.65)
    parser.add_argument("--seed", type=int, default=57)
    parser.add_argument("--device", choices=("cpu", "gpu", "both"), default="both")
    parser.add_argument("--output", type=Path, default=HERE / "haka_cpu_gpu_benchmark.gif")
    parser.add_argument("--manifest", type=Path, help="default: output path with .json suffix")
    parser.add_argument("--animation-frames", type=int, default=24)
    parser.add_argument("--playback-fps", type=int, default=6)
    args = parser.parse_args()
    if args.seconds <= 0 or not 0 < args.frame_rate <= OCAM2K_MAX_FRAME_RATE_HZ or args.seeing <= 0:
        parser.error(
            f"--seconds and --seeing must be positive; --frame-rate must be in "
            f"(0, {OCAM2K_MAX_FRAME_RATE_HZ:g}]"
        )
    if (
        args.batch_size < 1
        or args.warmup_frames < 1
        or args.animation_frames < 2
        or args.playback_fps < 1
    ):
        parser.error("batch/warm-up/playback counts must be positive and animation frames >= 2")
    if args.output.suffix.lower() != ".gif":
        parser.error("--output must end in .gif")
    return args


def _synchronize(sensor: makewfs.WavefrontSensor) -> None:
    """Wait for all queued CUDA work before reading the wall clock."""
    if not sensor.backend.is_cpu:
        sensor.backend.xp.cuda.Stream.null.synchronize()


def _gpu_available() -> bool:
    """Return whether a usable CuPy CUDA device is present."""
    try:
        import cupy

        return bool(cupy.cuda.runtime.getDeviceCount())
    except (ImportError, RuntimeError):
        return False


def _devices(selection: str) -> tuple[str, ...]:
    gpu = _gpu_available()
    if selection == "gpu":
        if not gpu:
            raise SystemExit("--device gpu requested, but no usable CuPy CUDA device was found")
        return ("gpu",)
    if selection == "both":
        if not gpu:
            print("warning: no usable GPU; rendering the CPU result only", file=sys.stderr)
            return ("cpu",)
        return ("cpu", "gpu")
    return ("cpu",)


def _build_runtime(
    *,
    device: str,
    pupil_path: Path,
    collecting_area_m2: float,
    magnitude: float,
    frame_rate_hz: float,
    seeing: float,
    seed: int,
) -> tuple[makewfs.WavefrontSensor, Any, float]:
    """Construct the persistent HAKA sensor and atmosphere outside timed work."""
    try:
        import pyturb
    except ImportError as exc:  # pragma: no cover - example dependency
        raise SystemExit("install makewfs[examples,interop] to run this benchmark") from exc

    base = makewfs.load_config(HERE / "keck_haka.toml")
    base = replace(base, numerics=replace(base.numerics, device=device))
    mode = select_camera_mode(magnitude, load_camera_modes())
    mode = replace(mode, frame_rate_1_hz=frame_rate_hz, frame_rate_2_hz=frame_rate_hz)
    sensor = configured_sensor(
        base,
        magnitude=magnitude,
        mode=mode,
        frame_rate_column="WSFRRT1",
        pupil_path=pupil_path,
        collecting_area_m2=collecting_area_m2,
        source_temperature_k=HAKA_SOURCE_TEMPERATURE_K,
        airmass=HAKA_REFERENCE_AIRMASS,
    )
    atmosphere = pyturb.Atmosphere.from_profile(
        "mauna-kea",
        seeing=seeing,
        diameter=base.input.grid_extent_m,
        n=base.input.shape[0],
        seed=seed,
        engine="extrude",
        dtype=base.numerics.dtype,
        device=device,
    )
    return sensor, atmosphere, 1.0 / frame_rate_hz


def _measure_device(
    *,
    device: str,
    pupil_path: Path,
    collecting_area_m2: float,
    seconds: float,
    batch_size: int,
    warmup_frames: int,
    magnitude: float,
    frame_rate_hz: float,
    seeing: float,
    seed: int,
) -> DeviceResult:
    """Construct once, then measure completed warm HAKA frames for ``seconds``."""
    sensor, atmosphere, exposure_s = _build_runtime(
        device=device,
        pupil_path=pupil_path,
        collecting_area_m2=collecting_area_m2,
        magnitude=magnitude,
        frame_rate_hz=frame_rate_hz,
        seeing=seeing,
        seed=seed,
    )
    for frame_index in range(warmup_frames):
        opd_m = atmosphere.evolve(exposure_s)
        sensor.expose(opd_m, seed=seed * 100_000 + frame_index)
    _synchronize(sensor)

    samples: list[ThroughputSample] = []
    measured_frames = 0
    start = perf_counter()
    while True:
        for _ in range(batch_size):
            opd_m = atmosphere.evolve(exposure_s)
            sensor.expose(opd_m, seed=seed * 100_000 + warmup_frames + measured_frames)
            measured_frames += 1
        _synchronize(sensor)
        elapsed_s = perf_counter() - start
        samples.append(
            ThroughputSample(
                elapsed_s=elapsed_s,
                frames=measured_frames,
                frames_per_s=measured_frames / elapsed_s,
            )
        )
        if elapsed_s >= seconds:
            break

    elapsed_s = samples[-1].elapsed_s
    frames_per_s = measured_frames / elapsed_s
    return DeviceResult(
        device=device,
        frames=measured_frames,
        elapsed_s=elapsed_s,
        frames_per_s=frames_per_s,
        physical_frame_rate_hz=frame_rate_hz,
        simulated_open_loop_time_s=measured_frames * exposure_s,
        real_time_factor=frames_per_s / frame_rate_hz,
        samples=tuple(samples),
    )


def _render_device_sequence(
    result: DeviceResult,
    *,
    pupil_path: Path,
    collecting_area_m2: float,
    magnitude: float,
    seeing: float,
    seed: int,
    animation_frames: int,
    playback_fps: int,
) -> NDArray[np.float32]:
    """Render untimed images at atmosphere times implied by measured throughput."""
    try:
        import getframes
    except ImportError as exc:  # pragma: no cover - required dependency
        raise SystemExit("makewfs requires getframes to render HAKA frames") from exc

    sensor, atmosphere, exposure_s = _build_runtime(
        device=result.device,
        pupil_path=pupil_path,
        collecting_area_m2=collecting_area_m2,
        magnitude=magnitude,
        frame_rate_hz=result.physical_frame_rate_hz,
        seeing=seeing,
        seed=seed,
    )
    atmosphere_step_s = result.frames_per_s * exposure_s / playback_fps
    images: list[NDArray[np.float32]] = []
    for frame_index in range(animation_frames):
        opd_m = atmosphere.opd() if frame_index == 0 else atmosphere.evolve(atmosphere_step_s)
        frame = sensor.expose(opd_m, seed=seed * 1_000_000 + frame_index)
        _synchronize(sensor)
        images.append(np.asarray(getframes.to_numpy(frame.data), dtype=np.float32))
    return np.stack(images)


def _display_state(result: DeviceResult, frame_index: int, playback_fps: int) -> tuple[int, float]:
    """Return generated-frame counter and atmosphere time for one GIF frame."""
    playback_time_s = frame_index / playback_fps
    simulated_frame = round(playback_time_s * result.frames_per_s)
    return simulated_frame, simulated_frame / result.physical_frame_rate_hz


def _write_gif(
    output: Path,
    results: tuple[DeviceResult, ...],
    sequences: dict[str, NDArray[np.float32]],
    *,
    animation_frames: int,
    playback_fps: int,
) -> None:
    """Show equal-wall-time CPU/GPU detector streams advancing at measured rates."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
    except ImportError as exc:  # pragma: no cover - example dependency
        raise SystemExit("install makewfs[examples] to render the benchmark GIF") from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    stack = np.concatenate([sequences[result.device] for result in results], axis=0)
    display_min = float(np.percentile(stack, 0.5))
    display_max = float(np.percentile(stack, 99.8))
    if display_max <= display_min:
        display_max = display_min + 1.0
    figure, axis_grid = plt.subplots(
        1,
        len(results),
        figsize=(5.5 * len(results), 5.7),
        constrained_layout=True,
        squeeze=False,
    )
    axes = axis_grid[0]
    image_artists = []
    overlays = []
    for axis, result in zip(axes, results):
        artist = axis.imshow(
            sequences[result.device][0],
            origin="lower",
            vmin=display_min,
            vmax=display_max,
            interpolation="nearest",
        )
        image_artists.append(artist)
        axis.set_title(result.device.upper())
        axis.set_xlabel("OCAM2K x pixel")
        axis.set_ylabel("OCAM2K y pixel")
        overlay = axis.text(
            0.025,
            0.975,
            "",
            transform=axis.transAxes,
            ha="left",
            va="top",
            color="white",
            fontsize=10,
            bbox={"facecolor": "black", "alpha": 0.72, "edgecolor": "none", "pad": 5},
        )
        overlays.append(overlay)
    figure.colorbar(
        image_artists[0],
        ax=list(axes),
        fraction=0.035,
        pad=0.02,
        label="raw OCAM2K signal (ADU; shared scale)",
    )
    title = figure.suptitle(
        "Keck II HAKA open loop: equal wall-clock playback\n"
        "pyturb + 8-wavelength Shack--Hartmann optics + OCAM2K\n"
        "Atmosphere clocks advance at measured throughput; intermediate frames are skipped"
    )

    def update(frame_index: int) -> tuple[Any, ...]:
        artists: list[Any] = [title]
        for result, sequence, image_artist, overlay in zip(
            results,
            (sequences[result.device] for result in results),
            image_artists,
            overlays,
        ):
            simulated_frame, atmosphere_time_s = _display_state(result, frame_index, playback_fps)
            image_artist.set_data(sequence[frame_index])
            overlay.set_text(
                f"{result.frames_per_s:,.1f} generated frames/s\n"
                f"{result.real_time_factor:.3f}x real time\n"
                f"frame {simulated_frame:,} | atmosphere t={atmosphere_time_s:.3f} s"
            )
            artists.extend((image_artist, overlay))
        return tuple(artists)

    animation = FuncAnimation(
        figure,
        update,
        frames=animation_frames,
        interval=1000.0 / playback_fps,
        blit=False,
    )
    animation.save(output, writer=PillowWriter(fps=playback_fps))
    plt.close(figure)


def _installed_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def _gpu_model() -> str | None:
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return output.splitlines()[0].strip() if output.splitlines() else None


def _git_state(root: Path) -> tuple[str | None, bool | None]:
    """Return source revision and dirty state for benchmark provenance."""
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return revision, dirty


def main() -> int:
    args = _arguments()
    devices = _devices(args.device)
    pupil = make_keck_pupil()
    collecting_area_m2 = pupil_collecting_area_m2(pupil)
    with tempfile.TemporaryDirectory(prefix="makewfs-haka-benchmark-") as temporary:
        pupil_path = Path(temporary) / "keck_primary.npy"
        np.save(pupil_path, pupil)
        results = tuple(
            _measure_device(
                device=device,
                pupil_path=pupil_path,
                collecting_area_m2=collecting_area_m2,
                seconds=args.seconds,
                batch_size=args.batch_size,
                warmup_frames=args.warmup_frames,
                magnitude=args.magnitude,
                frame_rate_hz=args.frame_rate,
                seeing=args.seeing,
                seed=args.seed,
            )
            for device in devices
        )
        sequences = {
            result.device: _render_device_sequence(
                result,
                pupil_path=pupil_path,
                collecting_area_m2=collecting_area_m2,
                magnitude=args.magnitude,
                seeing=args.seeing,
                seed=args.seed,
                animation_frames=args.animation_frames,
                playback_fps=args.playback_fps,
            )
            for result in results
        }

    _write_gif(
        args.output,
        results,
        sequences,
        animation_frames=args.animation_frames,
        playback_fps=args.playback_fps,
    )
    manifest = args.manifest or args.output.with_suffix(".json")
    revision, source_dirty = _git_state(HERE.parents[1])
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "source_revision": revision,
        "source_dirty": source_dirty,
        "output_gif": args.output.name,
        "machine": {
            "platform": platform.platform(),
            "cpu": _cpu_model(),
            "gpu": _gpu_model() if "gpu" in devices else None,
        },
        "dependencies": {
            name: _installed_version(name)
            for name in ("makewfs", "numpy", "scipy", "cupy", "getframes", "pyturb")
        },
        "haka": {
            "magnitude_v": args.magnitude,
            "physical_frame_rate_hz": args.frame_rate,
            "atmosphere_profile": "mauna-kea",
            "atmosphere_engine": "extrude",
            "seeing_arcsec_at_500_nm": args.seeing,
            "input_shape": list(pupil.shape),
            "detector_shape": [228, 228],
            "lenslets": [57, 57],
            "pixels_per_subaperture": [4, 4],
            "spectral_samples": 8,
            "numerics_dtype": "float32",
            "downstream_haka_throughput": HAKA_DOWNSTREAM_THROUGHPUT,
        },
        "methodology": {
            "target_wall_time_per_device_s": args.seconds,
            "batch_size": args.batch_size,
            "warmup_frames": args.warmup_frames,
            "included": [
                "pyturb non-periodic atmosphere evolution by one physical exposure",
                "eight-wavelength Shack-Hartmann optical propagation",
                "OCAM2K EMCCD exposure, detector noise, and truth arrays",
            ],
            "excluded": [
                "source SED and atmospheric-extinction quadrature",
                "pupil generation and file loading",
                "atmosphere and WavefrontSensor construction",
                "warm-up and CUDA kernel/FFT plan initialization",
                "host transfer, calibration, plotting, and file output",
            ],
            "persistent_sensor": True,
            "device_resident_gpu_path": True,
            "cuda_synchronization": "after warm-up and every timed batch",
            "distinct_detector_seed_per_frame": True,
            "gif_visualization": (
                "untimed detector renders at atmosphere times implied by measured throughput; "
                "intermediate generated frames are skipped"
            ),
        },
        "results": [
            {
                **asdict(result),
                "samples": [asdict(sample) for sample in result.samples],
            }
            for result in results
        ],
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    summary = ", ".join(
        f"{result.device.upper()} {result.frames_per_s:,.1f} fps "
        f"({result.real_time_factor:.2f}x real time)"
        for result in results
    )
    print(f"wrote {args.output} and {manifest}; {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
