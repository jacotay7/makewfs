"""Showcase: four wavefront sensors on one sky, with live throughput.

Renders an animated clip of four `makewfs` sensor configurations watching the
same wind-blown atmosphere, so the sensor images evolve together under one
frozen-flow wavefront:

  1. Shack-Hartmann 20x20   -- float32, 160x160 detector; the classic
                               subaperture spot grid, spots drifting with the
                               local wavefront slope
  2. Shack-Hartmann 60x60   -- float64, 360x360 detector; a high-order
                               geometry where each subaperture is near r0
  3. Pyramid, 8-pt modulation -- float32, 80x80 detector; four pupil images
                               whose intensity difference encodes the slope
  4. Broadband LGS SH       -- three wavelengths x three sodium ranges, with a
                               side-launched beacon, so the spots show
                               range-dependent elongation

The wavefront comes from `pyturb` (frozen flow), the optics from `makewfs`, and
the ADU from `getframes` -- the full atmosphere -> optics -> detector path. Each
panel is overlaid with the end-to-end throughput that configuration sustained on
this machine (warm sensor, device-resident, CUDA synchronized), so the clip
doubles as a speed sheet across the sensor set.

The raw seeing-limited wavefront is many waves and would scramble the spots
entirely, so `--residual-scale` emulates a partially corrected wavefront: weak
enough that the structure stays identifiable and visibly moves with the wind.

Run:  ``python examples/showcase.py``                  (auto GPU if available)
      ``python examples/showcase.py --device cpu --frames 40``
      ``python examples/showcase.py --out docs/assets/showcase.webp``

Needs matplotlib + pillow + pyturb (``pip install "makewfs[examples,interop]"``);
a CUDA GPU (``makewfs[gpu]``) is what turns the overlays into the headline.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np

import makewfs

# --- the shared sky -----------------------------------------------------------
SITE = "paranal-median"
SEEING = 0.6  # arcsec at 500 nm
SEED = 7
RESIDUAL_SCALE = 0.12  # partial-AO residual factor on the raw OPD (1.0 = raw seeing)

# The benchmark configurations are deliberately photon-starved (a 1 ms exposure
# of a 2e6 photons/s source puts about one photon in the brightest pixel), which
# times the optics honestly but renders as pure noise. Each panel's source rate
# is scaled so its brightest pixel lands near this many photons, which exposes
# every geometry comparably without touching the optics or the detector model.
TARGET_PEAK_PHOTONS = 3000.0

# --- the movie ---------------------------------------------------------------
DT = 5e-3  # s per frame of frozen flow
N_FRAMES = 120
PLAYBACK_FPS = 25
BENCH_WARMUP = 5  # untimed frames before the throughput measurement
BENCH_RUNS = 60  # frames timed for the per-panel frames/s number

_ROOT = Path(__file__).resolve().parents[1]
BENCH_CONFIGS = _ROOT / "benchmarks" / "configs"
EXAMPLE_CONFIGS = _ROOT / "examples" / "configs"


def _modulated_pyramid(config):
    """8-point modulation on the example geometry, whose pupils stay separated.

    The `benchmarks/configs` pyramids deliberately set `pupil_separation_pixels`
    below `pixels_across_pupil`, which keeps the timed detector small but merges
    the four pupil images into one blob. The example geometry separates them, so
    the clip shows the four-pupil signal a pyramid actually produces.
    """
    return replace(
        config,
        pyramid=replace(
            config.pyramid,
            modulation_radius_lambda_over_d=2.0,
            modulation_samples=8,
        ),
    )


PANELS = (
    ("Shack-Hartmann 20x20", BENCH_CONFIGS / "shack_hartmann_20x20_float32.toml", None),
    ("Shack-Hartmann 60x60", BENCH_CONFIGS / "shack_hartmann_60x60_float64.toml", None),
    ("Pyramid, 8-pt modulation", EXAMPLE_CONFIGS / "pyramid_minimal.toml", _modulated_pyramid),
    ("Broadband LGS SH", BENCH_CONFIGS / "shack_hartmann_broadband_lgs.toml", None),
)


def _resolve_device(requested: str) -> str:
    """``"auto"`` -> ``"gpu"`` when CuPy imports, else ``"cpu"``."""
    if requested != "auto":
        return requested
    try:
        import cupy  # noqa: F401

        return "gpu"
    except Exception:
        return "cpu"


def _device_label(device: str) -> str:
    if device == "gpu":
        try:
            import cupy as cp

            name = cp.cuda.runtime.getDeviceProperties(0)["name"]
            name = name.decode() if isinstance(name, bytes) else str(name)
            return name.replace("NVIDIA GeForce ", "").strip()
        except Exception:
            return "GPU"
    import platform

    return f"CPU ({platform.machine()})"


def _make_sync(device: str) -> Callable[[], None]:
    """A no-op on CPU; a full device barrier on GPU (so timings are honest)."""
    if device != "gpu":
        return lambda: None
    import cupy as cp

    return cp.cuda.Stream.null.synchronize


class Panel:
    """One sensor configuration: its sensor, its matching atmosphere, its timing.

    The sensor and the atmosphere are rebuilt for each pass over the panel
    (throughput, then frame collection), so the timed loop never pays for the
    frame-collection bookkeeping and vice versa.
    """

    def __init__(self, title: str, config_path: Path, tweak, device: str, residual_scale: float):
        self.title = title
        self.config_name = config_path.name
        self.device = device
        self.residual_scale = residual_scale
        self.fps: float | None = None
        self.badge_extra: str | None = None

        config = makewfs.load_config(config_path)
        if tweak is not None:
            config = tweak(config)
        self.config = replace(config, numerics=replace(config.numerics, device=device))
        self.dtype = self.config.numerics.dtype
        self.output_shape = makewfs.WavefrontSensor(self.config).engine.output_shape
        self.bias_adu = self._bias_offset_adu()
        self._scale_source_to_target_peak()

    def _bias_offset_adu(self) -> float:
        """The detector's bias pedestal, so the clip can show signal above it."""
        import getframes

        preset = self.config.detector.preset
        return float(getframes.load_preset(preset).bias_offset_adu) if preset else 0.0

    def _scale_source_to_target_peak(self) -> None:
        """Rescale the source so the brightest pixel lands near the display target.

        The ideal photon-rate map is linear in the source rate, so one probe
        render fixes the factor. Only the source normalization changes; the
        optics, the sampling and the detector model are untouched.
        """
        import pyturb

        sensor = self.build_sensor()
        atmosphere = self.build_atmosphere()
        rate = sensor.photon_rate(self.residual(atmosphere, 0))
        peak = float(np.max(pyturb.to_numpy(rate))) * self.config.detector.exposure_s
        if peak <= 0.0:
            return
        source = self.config.source
        scaled = source.detector_photon_rate_per_s * (TARGET_PEAK_PHOTONS / peak)
        self.config = replace(
            self.config, source=replace(source, detector_photon_rate_per_s=scaled)
        )

    def build_sensor(self) -> makewfs.WavefrontSensor:
        return makewfs.WavefrontSensor(self.config)

    def build_atmosphere(self):
        """A frozen-flow atmosphere matching this panel's input grid and precision."""
        import pyturb

        return pyturb.Atmosphere.from_profile(
            SITE,
            seeing=SEEING,
            diameter=self.config.telescope.pupil_diameter_m,
            n=self.config.input.shape[0],
            seed=SEED,
            device=self.device,
            dtype=self.dtype,
            # Row extrusion, not the periodic spectral engine: at these wind
            # speeds a spectral screen wraps partway through the clip and starts
            # replaying turbulence it has already shown.
            engine="extrude",
        )

    def residual(self, atmosphere, index: int):
        """The partially corrected OPD [m] for frame ``index``, on the device."""
        opd = atmosphere.opd(index * DT)
        return (opd * self.residual_scale).astype(self.dtype)


def benchmark_panel(panel: Panel, sync: Callable[[], None]) -> float:
    """End-to-end frames/s for this configuration, warm and device-synchronized.

    Timed on a fixed residual wavefront, matching `benchmarks/run.py`: the cost
    being measured is the optics + detector path, not the atmosphere driving it.
    """
    sensor = panel.build_sensor()
    atmosphere = panel.build_atmosphere()
    opd = panel.residual(atmosphere, 0)
    for index in range(BENCH_WARMUP):
        sensor.expose(opd, seed=index)
    sync()
    start = time.perf_counter()
    for index in range(BENCH_RUNS):
        sensor.expose(opd, seed=index)
    sync()
    return BENCH_RUNS / (time.perf_counter() - start)


def collect_frames(panel: Panel) -> np.ndarray:
    """``(N_FRAMES, *output_shape)`` detector ADU on the host, plus the OPD rms."""
    import getframes
    import pyturb

    sensor = panel.build_sensor()
    atmosphere = panel.build_atmosphere()
    out = np.empty((N_FRAMES, *panel.output_shape), dtype=np.float32)
    rms = []
    for index in range(N_FRAMES):
        residual = panel.residual(atmosphere, index)
        rms.append(float(np.std(pyturb.to_numpy(residual))) * 1e9)
        frame = sensor.expose(residual, seed=index)
        out[index] = getframes.to_numpy(frame.data).astype(np.float32)
    panel.badge_extra = f"residual {np.mean(rms):.0f} nm rms"
    return np.clip(out - panel.bias_adu, 0.0, None)


def render_animation(panels, frames, device_label, residual_scale, out_path):
    """Assemble the four panels into one animated clip via matplotlib + pillow.

    Saved as animated WebP rather than GIF: a quantized GIF palette shared
    across frames bands and flickers on the smooth detector gradients as the
    photon noise drifts across quantization boundaries frame to frame.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    ink, sub, bg = "#e6edf3", "#9aa7b4", "#0b0f14"
    accent = "#ffd166"
    highlight = "#7fd1c1"

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 8.8), dpi=68)
    fig.patch.set_facecolor(bg)
    fig.subplots_adjust(left=0.015, right=0.985, top=0.80, bottom=0.115, wspace=0.03, hspace=0.22)

    images = []
    for ax, panel, stack in zip(axes.flat, panels, frames):
        ax.set_facecolor(bg)
        # Per-panel scale on signal above the bias pedestal: the configurations
        # differ in geometry and flux spreading, and the pedestal is several
        # times the per-pixel signal, so a shared 0-based ADU range would show
        # four flat rectangles.
        vmax = float(np.percentile(stack, 99.8)) or 1.0
        im = ax.imshow(
            stack[0],
            cmap="inferno",
            vmin=0.0,
            vmax=vmax,
            origin="lower",
            interpolation="nearest",
            animated=True,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#233040")
        ax.set_title(panel.title, color=ink, fontsize=12.5, fontweight="bold", pad=16)
        rate = f"{panel.fps:,.0f} frames/s" if panel.fps else ""
        ax.text(
            0.045,
            0.955,
            rate,
            transform=ax.transAxes,
            color=accent,
            fontsize=12,
            fontweight="bold",
            va="top",
            ha="left",
            bbox={"boxstyle": "round,pad=0.32", "fc": "#11161d", "ec": "#2c3846"},
        )
        shape = "x".join(str(value) for value in panel.output_shape)
        ax.text(
            0.955,
            0.045,
            f"{shape} · {panel.dtype}",
            transform=ax.transAxes,
            color=highlight,
            fontsize=10,
            fontweight="bold",
            va="bottom",
            ha="right",
            bbox={"boxstyle": "round,pad=0.3", "fc": "#11161d", "ec": "#2c3846"},
        )
        images.append(im)

    fig.text(
        0.5,
        0.965,
        "makewfs — wavefront sensor showcase",
        color=ink,
        fontsize=17,
        fontweight="bold",
        ha="center",
        va="top",
    )
    fig.text(
        0.5,
        0.925,
        f"Paranal median · seeing {SEEING:g}″ · 8 m pupil · residual x{residual_scale:g} · "
        f"{device_label}",
        color=sub,
        fontsize=10.5,
        ha="center",
        va="top",
    )
    fig.text(
        0.5,
        0.895,
        "pyturb frozen flow → makewfs optics → getframes detector (ADU)",
        color=sub,
        fontsize=10,
        ha="center",
        va="top",
    )

    tstamp = fig.text(
        0.985, 0.098, "", color=sub, fontsize=10, ha="right", va="bottom", family="monospace"
    )
    fig.text(
        0.015,
        0.098,
        "overlay = live end-to-end throughput on this machine",
        color=sub,
        fontsize=9,
        ha="left",
        va="bottom",
    )

    cax = fig.add_axes([0.30, 0.048, 0.40, 0.016])
    cb = fig.colorbar(images[0], cax=cax, orientation="horizontal")
    cb.set_label("signal above bias [ADU] (per-panel scale)", color=sub, fontsize=9)
    cb.outline.set_edgecolor("#233040")
    cax.tick_params(colors=sub, labelsize=8)

    def rgba_frame() -> Image.Image:
        fig.canvas.draw()
        return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB")

    pil_frames = []
    for index in range(N_FRAMES):
        for im, stack in zip(images, frames):
            im.set_data(stack[index])
        tstamp.set_text(f"t = {index * DT * 1e3:5.0f} ms")
        pil_frames.append(rgba_frame())
    plt.close(fig)

    pil_frames[0].save(
        out_path,
        format="WEBP",
        save_all=True,
        append_images=pil_frames[1:],
        duration=int(1000 / PLAYBACK_FPS),
        loop=0,
        quality=90,
        method=6,
    )
    return out_path


def main() -> None:
    global N_FRAMES

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"])
    parser.add_argument(
        "--frames", type=int, default=None, help=f"number of animation frames (default {N_FRAMES})"
    )
    parser.add_argument(
        "--residual-scale",
        type=float,
        default=RESIDUAL_SCALE,
        help="scale on the raw OPD emulating partial AO correction",
    )
    parser.add_argument("--out", default="makewfs_showcase.webp")
    args = parser.parse_args()

    if args.frames is not None:
        if args.frames < 2:
            raise SystemExit("--frames must be >= 2")
        N_FRAMES = args.frames

    device = _resolve_device(args.device)
    label = _device_label(device)
    sync = _make_sync(device)
    print(f"device: {device}  ({label})")

    panels = [
        Panel(title, path, tweak, device, args.residual_scale) for title, path, tweak in PANELS
    ]

    print("benchmarking throughput per panel ...")
    for panel in panels:
        panel.fps = benchmark_panel(panel, sync)
        print(f"  {panel.config_name:42s} {panel.fps:9,.0f} frames/s")

    print(f"rendering {N_FRAMES} frames x {len(panels)} panels ...")
    frames = [collect_frames(panel) for panel in panels]

    out = render_animation(panels, frames, label, args.residual_scale, args.out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
