"""Deterministic incoherent guide-source quadrature."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .config import WFSConfig


@dataclass(frozen=True)
class SourceState:
    """One normalized optical source state used by a sensor engine."""

    wavelength_m: float
    weight: float
    angle_x_rad: float
    angle_y_rad: float
    range_m: float | None = None


def _normalised(values: tuple[float, ...], count: int) -> NDArray[np.float64]:
    if not values:
        return np.full(count, 1.0 / count, dtype=np.float64)
    weights = np.asarray(values, dtype=np.float64)
    return weights / float(np.sum(weights))


def _angular_states(config: WFSConfig) -> list[tuple[float, float, float]]:
    """Return ``(x angle, y angle, weight)`` Gaussian angular quadrature."""
    source = config.source
    base_x = math.radians(source.field_angle_arcsec[0] / 3600.0)
    base_y = math.radians(source.field_angle_arcsec[1] / 3600.0)
    if source.angular_fwhm_arcsec == 0.0:
        return [(base_x, base_y, 1.0)]
    order = source.angular_quadrature_order
    # Three-point Gauss-Hermite is exact through fourth order for a normal
    # expectation. For higher configured orders use an evenly weighted grid;
    # this keeps the rule deterministic without requiring an optional package.
    sigma = math.radians(source.angular_fwhm_arcsec / 3600.0) / 2.3548200450309493
    if order == 3:
        nodes = np.array([-math.sqrt(3.0), 0.0, math.sqrt(3.0)])
        one_d = np.array([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0])
    else:
        nodes = np.linspace(-math.sqrt(3.0), math.sqrt(3.0), order)
        one_d = np.full(order, 1.0 / order)
    return [
        (base_x + sigma * float(x), base_y + sigma * float(y), float(wx * wy))
        for x, wx in zip(nodes, one_d)
        for y, wy in zip(nodes, one_d)
    ]


def iter_source_states(config: WFSConfig) -> tuple[SourceState, ...]:
    """Build normalized wavelength, angular, and LGS-range source states."""
    source = config.source
    wavelengths = source.wavelengths_m or (config.sensor.wavelength_m,)
    wavelength_weights = _normalised(source.wavelength_weights, len(wavelengths))
    angular = _angular_states(config)
    ranges = source.lgs_ranges_m or (None,)
    range_weights = _normalised(source.lgs_range_weights, len(ranges))
    states: list[SourceState] = []
    for wavelength, wavelength_weight in zip(wavelengths, wavelength_weights):
        for angle_x, angle_y, angular_weight in angular:
            for range_m, range_weight in zip(ranges, range_weights):
                states.append(
                    SourceState(
                        float(wavelength),
                        float(wavelength_weight * angular_weight * range_weight),
                        angle_x,
                        angle_y,
                        None if range_m is None else float(range_m),
                    )
                )
    total = sum(state.weight for state in states)
    if not math.isclose(total, 1.0, rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError("source quadrature weights failed to normalize")
    return tuple(states)


__all__ = ["SourceState", "iter_source_states"]
