"""Build a deterministic, labelled documentation gallery of core workflows."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

import makewfs


def _opd(shape: tuple[int, int], *, scale_m: float = 0.12e-6) -> np.ndarray:
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    radius = np.hypot(xx - (shape[1] - 1) / 2, yy - (shape[0] - 1) / 2)
    return scale_m * np.exp(-((radius / (shape[0] / 4)) ** 2))


def _image(axis: Any, image: np.ndarray, *, title: str, label: str, cmap: str = "viridis") -> None:
    artist = axis.imshow(image, origin="lower", cmap=cmap)
    axis.set_title(title)
    axis.set_xlabel("detector x (pixel)")
    axis.set_ylabel("detector y (pixel)")
    axis.figure.colorbar(artist, ax=axis, fraction=0.046, pad=0.04, label=label)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("docs/gallery/makewfs-gallery.svg"))
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("install makewfs[examples] to build the gallery") from exc

    root = Path(__file__).parent
    repository_root = root.parent.resolve()
    sh_config_path = root / "configs" / "shack_hartmann_minimal.toml"
    pyramid_config_path = root / "configs" / "pyramid_minimal.toml"
    lgs_config_path = root / "configs" / "lgs_thin_beacon.toml"
    sh_config = makewfs.load_config(sh_config_path)
    pyramid_config = makewfs.load_config(pyramid_config_path)
    lgs_config = makewfs.load_config(lgs_config_path)
    sh_sensor = makewfs.WavefrontSensor(sh_config)
    pyramid_sensor = makewfs.WavefrontSensor(pyramid_config)
    lgs_source = replace(
        lgs_config.source,
        lgs_ranges_m=(89e3, 91e3),
        lgs_range_weights=(0.5, 0.5),
        lgs_launch_position_m=(2.0, 0.0),
    )
    lgs_sensor = makewfs.WavefrontSensor(replace(lgs_config, source=lgs_source))

    sh_opd = _opd(sh_config.input.shape)
    pyramid_opd = _opd(pyramid_config.input.shape)
    lgs_opd = _opd(lgs_config.input.shape, scale_m=0.04e-6)
    sh_rate = sh_sensor.photon_rate(sh_opd)
    pyramid_rate = pyramid_sensor.photon_rate(pyramid_opd)
    lgs_rate = lgs_sensor.photon_rate(lgs_opd)
    modulated = makewfs.WavefrontSensor(
        replace(
            pyramid_config,
            pyramid=replace(
                pyramid_config.pyramid,
                modulation_radius_lambda_over_d=2.0,
                modulation_samples=8,
            ),
        )
    )
    modulation_reference = modulated.photon_rate(np.zeros(pyramid_config.input.shape))
    modulation_response = modulated.photon_rate(pyramid_opd) - modulation_reference
    magnitude_source = replace(
        sh_config.source,
        normalization="magnitude",
        detector_photon_rate_per_s=None,
        magnitude_system="vega",
        magnitude=12.0,
        band="R",
        throughput=0.35,
    )
    magnitude_sensor = makewfs.WavefrontSensor(replace(sh_config, source=magnitude_source))
    magnitude_frame = np.asarray(magnitude_sensor.expose(sh_opd, seed=1200))

    figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    _image(axes[0, 0], sh_opd * 1e9, title="Injected OPD", label="OPD (nm)", cmap="coolwarm")
    _image(
        axes[0, 1],
        sh_rate,
        title="Shack-Hartmann ideal image",
        label="photon rate (photons/s/pixel)",
    )
    _image(
        axes[0, 2],
        pyramid_rate,
        title="Pyramid ideal image",
        label="photon rate (photons/s/pixel)",
    )
    _image(
        axes[1, 0],
        lgs_rate,
        title="Side-launched finite sodium layer",
        label="photon rate (photons/s/pixel)",
    )
    _image(
        axes[1, 1],
        modulation_response,
        title="Modulated pyramid push response",
        label="rate difference (photons/s/pixel)",
        cmap="coolwarm",
    )
    _image(
        axes[1, 2],
        magnitude_frame,
        title="NGS magnitude 12 detector frame",
        label="detector signal (ADU)",
    )
    figure.suptitle(
        "makewfs capability gallery — CPU optics, seeded detector, explicit units",
        fontsize=15,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=140)
    manifest_path = args.manifest or args.output.with_suffix(".json")
    manifest: dict[str, Any] = {
        "seed": 1200,
        "units": {
            "input": "OPD metres converted to nm for display",
            "ideal_images": "photons/s/native detector pixel",
            "detector_frame": "ADU",
            "modulation_response": "photon-rate difference",
        },
        "configs": {
            "shack_hartmann": str(sh_config_path.resolve().relative_to(repository_root)),
            "pyramid": str(pyramid_config_path.resolve().relative_to(repository_root)),
            "lgs": str(lgs_config_path.resolve().relative_to(repository_root)),
        },
        "digests": {
            "shack_hartmann": sh_config.digest,
            "pyramid": pyramid_config.digest,
            "lgs": lgs_config.digest,
        },
        "notes": [
            "LGS return rate is supplied by configuration; only geometric elongation is modeled.",
            "The pyramid panel uses an ideal fixed four-face mask.",
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output} and {manifest_path}")


if __name__ == "__main__":
    main()
