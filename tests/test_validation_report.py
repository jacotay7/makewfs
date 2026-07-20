"""Assertions for the deterministic, dependency-free validation report."""

from pathlib import Path

import numpy as np
from validation.run import _metrics

ROOT = Path(__file__).parents[1]


def test_shack_hartmann_report_quantifies_analytic_tilt_response() -> None:
    metrics = _metrics(ROOT / "examples" / "configs" / "shack_hartmann_minimal.toml")
    assert metrics["sh_tilt_valid_lenslets"] > 0
    assert metrics["sh_tilt_relative_scale_error"] < 0.05
    assert metrics["sh_tilt_cross_axis_max_pixels"] < 0.01
    assert metrics["sh_tilt_std_pixels"] < 0.01


def test_pyramid_report_quantifies_low_order_push_pull_response() -> None:
    metrics = _metrics(ROOT / "examples" / "configs" / "pyramid_minimal.toml")
    for mode in ("tip", "tilt", "focus"):
        assert metrics[f"pyramid_{mode}_push_pull_antisymmetry"] < 0.03
        assert metrics[f"pyramid_{mode}_normalized_response_per_wave"] > 0
    assert np.isclose(
        metrics["pyramid_tip_normalized_response_per_wave"],
        metrics["pyramid_tilt_normalized_response_per_wave"],
        rtol=1e-3,
    )
