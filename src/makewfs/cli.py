"""Command-line interface for config validation and phase rendering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .api import WavefrontSensor
from .config import load_config


def _load_array(path: str | Path) -> NDArray[Any]:
    source = Path(path)
    if source.suffix.lower() == ".npy":
        return np.asarray(np.load(source))
    if source.suffix.lower() == ".npz":
        archive = np.load(source)
        if not archive.files:
            raise ValueError(f"phase archive {source} contains no arrays")
        return np.asarray(archive[archive.files[0]])
    if source.suffix.lower() in {".fits", ".fit"}:
        try:
            from astropy.io import fits  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover
            raise ImportError("FITS input requires astropy") from exc
        return np.asarray(fits.getdata(source))
    raise ValueError(f"unsupported array format {source.suffix!r}")


def _write_array(path: str | Path, array: NDArray[Any]) -> None:
    destination = Path(path)
    if destination.suffix.lower() == ".npy":
        np.save(destination, array)
    elif destination.suffix.lower() == ".npz":
        np.savez_compressed(destination, data=array)
    elif destination.suffix.lower() in {".fits", ".fit"}:
        try:
            from astropy.io import fits
        except ImportError as exc:  # pragma: no cover
            raise ImportError("FITS output requires astropy") from exc
        fits.writeto(destination, array, overwrite=True)
    else:
        raise ValueError(f"unsupported output format {destination.suffix!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="makewfs", description="Render configured AO WFS images from phase/OPD arrays."
    )
    parser.add_argument("--version", action="version", version="makewfs 0.1.0.dev0")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-config", help="validate a TOML configuration")
    validate.add_argument("config")
    render = commands.add_parser("render", help="render one detector frame")
    render.add_argument("config")
    render.add_argument("wavefront")
    render.add_argument("--output", required=True)
    render.add_argument("--seed", type=int, default=None)
    ideal = commands.add_parser("ideal", help="write an ideal photon-rate map")
    ideal.add_argument("config")
    ideal.add_argument("wavefront")
    ideal.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "validate-config":
            config = load_config(args.config)
            print(
                json.dumps({"valid": True, "digest": config.digest, "sensor": config.sensor.kind})
            )
            return 0
        sensor = WavefrontSensor.from_toml(args.config)
        wavefront = _load_array(args.wavefront)
        if args.command == "ideal":
            _write_array(args.output, sensor.photon_rate(wavefront))
        else:
            frame = sensor.expose(wavefront, seed=args.seed)
            frame.to_fits(args.output, overwrite=True) if Path(args.output).suffix.lower() in {
                ".fits",
                ".fit",
            } else _write_array(args.output, np.asarray(frame))
        return 0
    except (ValueError, KeyError, FileNotFoundError, ImportError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
