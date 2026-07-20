"""Tests for portable benchmark-report formatting and regression inputs."""

from benchmarks.render_device_table import render as render_devices
from benchmarks.render_table import render


def test_render_table_includes_environment_and_kernel_metadata() -> None:
    report = {
        "python": "3.13",
        "platform": "test-platform",
        "dependencies": {"numpy": "1.0"},
        "results": [
            {
                "config": "/tmp/sh.toml",
                "sensor": "shack_hartmann",
                "device": "gpu",
                "shape": [16, 16],
                "dtype": "float32",
                "source_states": 3,
                "wavelength_samples": 3,
                "range_samples": 1,
                "modulation_samples": 1,
                "modulation_radius_lambda_over_d": 0.0,
                "construction_s": 0.001,
                "warm_optical_frame_s": 0.002,
                "warm_detector_frame_s": 0.004,
                "warm_optical_frames_per_s": 500.0,
                "warm_detector_frames_per_s": 250.0,
                "python_peak_memory_mib": 2.5,
            }
        ],
    }
    table = render(report, source="report.json")
    assert "test-platform" in table
    assert "| sh.toml | shack_hartmann | gpu |" in table
    assert "| 3 | 1 | 1 @ 0.0 λ/D |" in table
    assert "500.0" in table
    assert "2.5" in table


def test_render_device_table_pairs_devices_and_reports_speedup() -> None:
    common = {
        "config": "/tmp/sh.toml",
        "sensor": "shack_hartmann",
        "shape": [16, 16],
        "source_states": 1,
        "modulation_samples": 1,
    }
    report = {
        "platform": "test-platform",
        "gpu": "test-gpu",
        "dependencies": {"numpy": "1.0"},
        "results": [
            {**common, "device": "cpu", "warm_detector_frames_per_s": 100.0},
            {**common, "device": "gpu", "warm_detector_frames_per_s": 250.0},
        ],
    }
    table = render_devices(report, source="report.json")
    assert "test-gpu" in table
    assert "| sh.toml | shack_hartmann | 16x16 | 1 | 100.0 | 250.0 | 2.50x |" in table
