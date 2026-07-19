"""Tests for the private portable optical-array backend boundary."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import numpy as np

from makewfs.backend import ArrayBackend
from makewfs.config import load_config
from makewfs.sensors.pyramid import PyramidEngine
from makewfs.sensors.shack_hartmann import ShackHartmannEngine


class _NamespaceSpy:
    """Delegate an Array API namespace while recording called operations."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __getattr__(self, name: str) -> Any:
        value = getattr(np, name)
        if not callable(value):
            return value

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            self.calls.append(name)
            return value(*args, **kwargs)

        return wrapped


def test_sensor_modules_do_not_allocate_through_numpy_directly() -> None:
    """Keep portable sensor kernels dependent on ``ArrayBackend`` methods."""
    forbidden = {
        "abs",
        "arange",
        "array",
        "asarray",
        "empty",
        "exp",
        "fft",
        "full",
        "hypot",
        "indices",
        "meshgrid",
        "mean",
        "ones",
        "ptp",
        "repeat",
        "sum",
        "tile",
        "where",
        "zeros",
    }
    root = Path(__file__).parents[1] / "src" / "makewfs" / "sensors"
    violations: list[str] = []
    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "np"
                and node.attr in forbidden
            ):
                violations.append(f"{path.name}:{node.lineno}: np.{node.attr}")
    assert not violations, "portable sensor operations leaked NumPy: " + ", ".join(violations)


def test_shack_hartmann_uses_injected_backend_and_preserves_cpu_result() -> None:
    config = load_config(
        Path(__file__).parents[1] / "examples" / "configs" / "shack_hartmann_minimal.toml"
    )
    spy = _NamespaceSpy()
    backend = ArrayBackend(spy, name="cpu")
    reference = ShackHartmannEngine(config).render(np.zeros(config.input.shape)).photon_rate
    result = (
        ShackHartmannEngine(config, backend=backend)
        .render(np.zeros(config.input.shape))
        .photon_rate
    )
    np.testing.assert_allclose(result, reference, rtol=0.0, atol=0.0)
    assert {"asarray", "zeros", "exp", "sum"} <= set(spy.calls)


def test_pyramid_uses_injected_backend_and_preserves_cpu_result() -> None:
    config = load_config(
        Path(__file__).parents[1] / "examples" / "configs" / "pyramid_minimal.toml"
    )
    spy = _NamespaceSpy()
    backend = ArrayBackend(spy, name="cpu")
    reference = PyramidEngine(config).render(np.zeros(config.input.shape)).photon_rate
    result = PyramidEngine(config, backend=backend).render(np.zeros(config.input.shape)).photon_rate
    np.testing.assert_allclose(result, reference, rtol=0.0, atol=0.0)
    assert {"asarray", "zeros", "exp", "sum"} <= set(spy.calls)
