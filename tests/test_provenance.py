"""Provenance and referenced-file hashing tests."""

from dataclasses import replace
from pathlib import Path

import numpy as np

from makewfs import WavefrontSensor, load_config


def test_source_curve_digest_is_recorded(tmp_path: Path) -> None:
    path = tmp_path / "sed.txt"
    np.savetxt(path, [[600.0, 1.0], [700.0, 1.0]])
    config = load_config(
        Path(__file__).parents[1] / "examples" / "configs" / "shack_hartmann_minimal.toml"
    )
    source = replace(config.source, wavelengths_m=(), wavelength_weights=(), sed_path=str(path))
    sensor = WavefrontSensor(replace(config, source=source))
    frame = sensor.expose(np.zeros(config.input.shape), seed=1)
    assert len(frame.metadata["wfs_source_sed_sha256"]) == 16
