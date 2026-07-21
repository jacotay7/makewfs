#!/usr/bin/env python3
"""Extract raw OCAM2K images from TRS over a 10 s window centered on a UTC time.

Raw images live in the `ocam2k` table as `SHWFS1Image`. Two details matter and
are not obvious from the schema (both verified against K2 data, 2026-07-20):

  * The field is declared `repeated fixed32` but holds uint16 pixels: each
    32-bit word packs TWO pixels, low half first. A record of 25992 words is
    therefore 51984 pixels = 228 x 228.
  * Only every 10th telemetry row carries an image (~75 Hz out of a 750 Hz
    frame rate). Rows without one come back as an empty array and are skipped.

Example
-------
    python extract_ocam_images.py --host k2
    python extract_ocam_images.py --center 2026-07-20T11:35:51.583000Z --duration 10
"""

import argparse
import os

import numpy as np
from astropy.time import Time
from kaotools.trs_tools.trs_tools import TRSSession

IMAGE_FIELD = "SHWFS1Image"
DEFAULT_CENTER = "2026-07-20T11:35:51.583000Z"


def iso_to_utc_ns(iso_string):
    """Convert an ISO-8601 UTC string to integer UTC nanoseconds."""
    unix_seconds = Time(iso_string.rstrip("Z"), format="isot", scale="utc").to_value(
        "unix", subfmt="long"
    )
    return int(np.rint(unix_seconds * np.longdouble(1_000_000_000)))


def unpack_image(raw_field):
    """Unpack a packed SHWFS1Image record into a 2-D uint16 frame.

    Each fixed32 word holds two pixels; the low 16 bits are the first (lower
    column index) of the pair. The selected-field query path can hand back the
    words as strings, so coerce through int64 before masking.
    """
    words = np.asarray(raw_field)
    if words.size == 0:
        return None
    words = words.astype(np.int64)

    pixels = np.empty(words.size * 2, dtype=np.uint16)
    pixels[0::2] = words & 0xFFFF
    pixels[1::2] = (words >> 16) & 0xFFFF

    side = round(np.sqrt(pixels.size))
    if side * side != pixels.size:
        raise ValueError(f"unpacked {pixels.size} pixels, which is not a square image")
    return pixels.reshape(side, side)


def fetch_ocam_images(host, start_ns, end_ns, chunk_seconds=1.0):
    """Fetch raw OCAM frames between two UTC timestamps, one chunk at a time."""
    chunk_ns = int(chunk_seconds * 1_000_000_000)
    timestamps = []
    frame_counters = []
    images = []

    with TRSSession(host) as client:
        chunk_start = start_ns
        while chunk_start < end_ns:
            chunk_end = min(chunk_start + chunk_ns, end_ns)
            frames = client.get_frames_between(
                "ocam2k",
                chunk_start,
                chunk_end,
                fields=["timestamp", "frameCounter", IMAGE_FIELD],
            )
            kept = 0
            for frame in frames:
                image = unpack_image(frame.get(IMAGE_FIELD, []))
                if image is None:
                    continue
                timestamps.append(frame["timestamp"])
                frame_counters.append(frame.get("frameCounter"))
                images.append(image)
                kept += 1
            print(f"  {len(frames)} rows, {kept} images")
            chunk_start = chunk_end

    return (
        np.array(timestamps, dtype=np.int64),
        np.array(frame_counters, dtype=np.int64),
        np.array(images),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--center",
        default=DEFAULT_CENTER,
        help="Center of the window, ISO-8601 UTC (default: %(default)s)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Total window length in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("TRS_SERVER"),
        help="TRS host: 'k1', 'k2', 'k3', or an IP (default: $TRS_SERVER)",
    )
    parser.add_argument(
        "--output", default="ocam_raw_images.npz", help="Output .npz path (default: %(default)s)"
    )
    args = parser.parse_args()

    center_ns = iso_to_utc_ns(args.center)
    half_window_ns = int(args.duration / 2.0 * 1_000_000_000)
    start_ns = center_ns - half_window_ns
    end_ns = center_ns + half_window_ns

    print(f"Center : {args.center} ({center_ns} ns UTC)")
    print(f"Window : {args.duration} s -> [{start_ns}, {end_ns}]")
    print(f"Host   : {args.host}")

    timestamps, frame_counters, images = fetch_ocam_images(args.host, start_ns, end_ns)

    if images.size == 0:
        print("No OCAM images found in that window.")
        return

    print(f"\nRetrieved {len(images)} images, shape {images.shape}, dtype {images.dtype}")
    print(f"Pixel range: {images.min()} - {images.max()}")
    print(f"Span: {(timestamps[-1] - timestamps[0]) / 1e9:.3f} s")

    np.savez_compressed(
        args.output,
        images=images,
        timestamps=timestamps,
        frame_counters=frame_counters,
        timestamps_iso=np.array([utc_ns_to_iso(t) for t in timestamps]),
        **dataset_metadata(args, start_ns, end_ns, images, timestamps),
    )
    print(f"Wrote {args.output} ({os.path.getsize(args.output) / 1e6:.1f} MB)")


def utc_ns_to_iso(timestamp_ns):
    """Format UTC nanoseconds as an ISO-8601 string."""
    t = Time(
        int(timestamp_ns) // 1_000_000_000,
        (int(timestamp_ns) % 1_000_000_000) / 1e9,
        format="unix",
        scale="utc",
    )
    return t.utc.isot + "Z"


def dataset_metadata(args, start_ns, end_ns, images, timestamps):
    """Provenance so the file stands on its own once copied off this machine."""
    cadence_ns = np.diff(timestamps) if len(timestamps) > 1 else np.array([0])
    return {
        "source": f"Keck RTC TRS database, table ocam2k, field {IMAGE_FIELD}",
        "trs_host": str(args.host),
        "window_center_utc": args.center,
        "window_start_utc": utc_ns_to_iso(start_ns),
        "window_end_utc": utc_ns_to_iso(end_ns),
        "window_duration_s": float(args.duration),
        "num_images": len(images),
        "image_shape": np.array(images.shape[1:]),
        "cadence_hz": float(1e9 / np.median(cadence_ns)) if cadence_ns.max() > 0 else 0.0,
        "units": "ADU (raw detector counts, uncalibrated)",
        "timestamp_units": "UTC nanoseconds since the Unix epoch",
        "decoding": (
            'SHWFS1Image is declared "repeated fixed32" but packs two uint16 '
            "pixels per word, low half = lower column index. 25992 words -> "
            "51984 pixels -> 228x228. Only every 10th ocam2k row carries an "
            "image (~75 Hz of a 750 Hz frame rate)."
        ),
        "caveats": (
            "(1) Pixel-pair order within each 32-bit word is inferred from "
            "little-endian packing convention, not confirmed against a "
            "reference image; if wrong, adjacent column pairs are swapped. "
            "(2) Raw frames carry an odd/even column offset (~16 ADU) that "
            "looks like per-amplifier structure and is not calibrated out."
        ),
        "created_utc": Time.now().utc.isot + "Z",
        "created_by": "extract_ocam_images.py",
    }


if __name__ == "__main__":
    main()
