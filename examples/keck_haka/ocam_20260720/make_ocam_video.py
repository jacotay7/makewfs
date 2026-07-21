#!/usr/bin/env python3
"""Render the OCAM image cube saved by extract_ocam_images.py as a video.

Intensity scaling is computed once over the whole cube and held fixed for every
frame, so brightness changes you see are real and not autoscaling artifacts.

Writes MP4 when an ffmpeg encoder is importable, otherwise GIF.

Example
-------
    python make_ocam_video.py ocam_raw_images.npz -o ocam.gif
    python make_ocam_video.py ocam_raw_images.npz --fps 25 --scale 2 --step 2
"""

import argparse
import os

import imageio.v2 as imageio
import numpy as np
from matplotlib import colormaps
from PIL import Image, ImageDraw


def scale_to_uint8(images, lo_pct, hi_pct):
    """Map the cube to 0-255 using fixed percentile limits from the whole cube."""
    lo = np.percentile(images, lo_pct)
    hi = np.percentile(images, hi_pct)
    if hi <= lo:
        hi = lo + 1.0
    normalized = (images.astype(np.float32) - lo) / (hi - lo)
    return np.clip(normalized, 0.0, 1.0), float(lo), float(hi)


def render_frames(images, timestamps, cmap_name, scale, annotate):
    """Colormap, upscale, and annotate each frame; yields uint8 RGB arrays."""
    cmap = colormaps[cmap_name]
    t0 = timestamps[0] if timestamps is not None else None

    for index, frame in enumerate(images):
        rgb = (cmap(frame)[..., :3] * 255).astype(np.uint8)
        image = Image.fromarray(rgb)

        if scale != 1:
            image = image.resize((image.width * scale, image.height * scale), Image.NEAREST)

        if annotate:
            draw = ImageDraw.Draw(image)
            label = f"frame {index:4d}"
            if t0 is not None:
                label += f"   t+{(timestamps[index] - t0) / 1e9:6.3f} s"
            draw.text((6, 6), label, fill=(255, 255, 255))

        yield np.asarray(image)


def pick_writer(output):
    """Return a usable output path, downgrading MP4 to GIF if no encoder exists."""
    if not output.lower().endswith(".mp4"):
        return output
    try:
        import imageio_ffmpeg  # noqa: F401

        return output
    except ImportError:
        fallback = output[:-4] + ".gif"
        print(
            "No ffmpeg encoder available (pip install imageio-ffmpeg for MP4); "
            f"writing {fallback} instead"
        )
        return fallback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        default="ocam_raw_images.npz",
        help=".npz written by extract_ocam_images.py (default: %(default)s)",
    )
    parser.add_argument(
        "-o", "--output", default="ocam_images.mp4", help="Output video path (default: %(default)s)"
    )
    parser.add_argument(
        "--fps", type=float, default=25.0, help="Playback frames per second (default: %(default)s)"
    )
    parser.add_argument(
        "--step", type=int, default=1, help="Use every Nth image (default: %(default)s)"
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=2,
        help="Integer upscale factor, nearest-neighbour (default: %(default)s)",
    )
    parser.add_argument(
        "--cmap", default="inferno", help="Matplotlib colormap (default: %(default)s)"
    )
    parser.add_argument(
        "--lo-pct",
        type=float,
        default=1.0,
        help="Lower intensity percentile (default: %(default)s)",
    )
    parser.add_argument(
        "--hi-pct",
        type=float,
        default=99.9,
        help="Upper intensity percentile (default: %(default)s)",
    )
    parser.add_argument("--no-annotate", action="store_true", help="Omit the frame/time overlay")
    args = parser.parse_args()

    data = np.load(args.input)
    images = data["images"][:: args.step]
    timestamps = data["timestamps"][:: args.step] if "timestamps" in data else None

    print(f"Loaded {len(images)} images of {images.shape[1]}x{images.shape[2]} from {args.input}")

    normalized, lo, hi = scale_to_uint8(images, args.lo_pct, args.hi_pct)
    print(
        f"Intensity scaling fixed at [{lo:.0f}, {hi:.0f}] ADU "
        f"(percentiles {args.lo_pct} - {args.hi_pct})"
    )

    output = pick_writer(args.output)
    frames = render_frames(normalized, timestamps, args.cmap, args.scale, not args.no_annotate)

    if output.lower().endswith(".gif"):
        imageio.mimsave(output, list(frames), format="GIF", duration=1000.0 / args.fps, loop=0)
    else:
        imageio.mimsave(output, list(frames), format="FFMPEG", fps=args.fps, quality=8)

    real_seconds = 0.0
    if timestamps is not None and len(timestamps) > 1:
        real_seconds = (timestamps[-1] - timestamps[0]) / 1e9
    print(
        f"Wrote {output} ({os.path.getsize(output) / 1e6:.1f} MB) - "
        f"{len(images)} frames at {args.fps:g} fps = "
        f"{len(images) / args.fps:.1f} s playback for {real_seconds:.1f} s of real time"
    )


if __name__ == "__main__":
    main()
