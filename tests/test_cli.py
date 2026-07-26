"""CLI smoke tests for array loading and writing."""

from pathlib import Path

import numpy as np
import pytest

from makewfs.cli import build_parser, main

CONFIG = Path(__file__).parents[1] / "examples" / "configs" / "shack_hartmann_minimal.toml"


def test_version_matches_package(capsys) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(SystemExit, match="0"):
        build_parser().parse_args(["--version"])
    assert capsys.readouterr().out.strip() == "makewfs 1.0.0"


def test_validate_and_ideal_commands(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    assert main(["validate-config", str(CONFIG)]) == 0
    assert '"valid": true' in capsys.readouterr().out
    wavefront_path = tmp_path / "phase.npy"
    output_path = tmp_path / "ideal.npy"
    np.save(wavefront_path, np.zeros((128, 128)))
    assert main(["ideal", str(CONFIG), str(wavefront_path), "--output", str(output_path)]) == 0
    result = np.load(output_path)
    assert result.shape == (64, 64)
    assert np.all(result >= 0)


def test_render_command_writes_detector_array(tmp_path: Path) -> None:
    wavefront_path = tmp_path / "phase.npy"
    output_path = tmp_path / "frame.npy"
    np.save(wavefront_path, np.zeros((128, 128)))
    assert (
        main(
            [
                "render",
                str(CONFIG),
                str(wavefront_path),
                "--output",
                str(output_path),
                "--seed",
                "1",
            ]
        )
        == 0
    )
    assert np.load(output_path).shape == (64, 64)


def test_npz_input_and_output_paths(tmp_path: Path) -> None:
    wavefront_path = tmp_path / "phase.npz"
    ideal_path = tmp_path / "ideal.npz"
    render_path = tmp_path / "frame.npz"
    np.savez(wavefront_path, phase=np.zeros((128, 128)))
    assert main(["ideal", str(CONFIG), str(wavefront_path), "--output", str(ideal_path)]) == 0
    assert np.load(ideal_path)["data"].shape == (64, 64)
    assert main(["render", str(CONFIG), str(wavefront_path), "--output", str(render_path)]) == 0
    assert np.load(render_path)["data"].shape == (64, 64)


def test_cli_reports_unsupported_file_format(tmp_path: Path) -> None:
    wavefront_path = tmp_path / "phase.npy"
    np.save(wavefront_path, np.zeros((128, 128)))
    with pytest.raises(SystemExit):
        main(["ideal", str(CONFIG), str(wavefront_path), "--output", str(tmp_path / "out.dat")])
