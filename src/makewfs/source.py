"""Deterministic incoherent guide-source quadrature."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

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


def _load_curve(path: str) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    data = np.asarray(np.loadtxt(Path(path)), dtype=np.float64)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"source curve {path} must contain two columns: wavelength_nm value")
    wavelength = data[:, 0]
    value = data[:, 1]
    if len(wavelength) < 2 or np.any(~np.isfinite(data)) or np.any(np.diff(wavelength) <= 0):
        raise ValueError(f"source curve {path} must have finite increasing wavelengths")
    if np.any(value < 0):
        raise ValueError(f"source curve {path} contains negative values")
    return wavelength, value


def _spectral_samples(config: WFSConfig) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Resolve configured wavelengths and optional relative SED/transmission curves."""
    source = config.source
    sed_grid: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None
    transmission_grid: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None
    if source.sed_path is not None:
        sed_grid = _load_curve(source.sed_path)
    if source.transmission_path is not None:
        transmission_grid = _load_curve(source.transmission_path)
        if np.any(transmission_grid[1] > 1.0):
            raise ValueError("source transmission curve must be in [0, 1]")

    if source.wavelengths_m and sed_grid is None and transmission_grid is None:
        return source.wavelengths_m, source.wavelength_weights

    if source.wavelengths_m:
        wavelengths = np.asarray(source.wavelengths_m, dtype=np.float64) * 1e9
    elif sed_grid is not None:
        wavelengths = sed_grid[0]
    elif transmission_grid is not None:
        wavelengths = transmission_grid[0]
    else:
        return (config.sensor.wavelength_m,), source.wavelength_weights

    weights = (
        np.asarray(source.wavelength_weights, dtype=np.float64)
        if source.wavelength_weights
        else np.ones(len(wavelengths), dtype=np.float64)
    )
    if sed_grid is not None:
        weights *= np.interp(wavelengths, sed_grid[0], sed_grid[1], left=0.0, right=0.0)
    if transmission_grid is not None:
        weights *= np.interp(
            wavelengths,
            transmission_grid[0],
            transmission_grid[1],
            left=0.0,
            right=0.0,
        )
    if sed_grid is not None or transmission_grid is not None:
        quadrature = np.empty(len(wavelengths), dtype=np.float64)
        quadrature[0] = (wavelengths[1] - wavelengths[0]) / 2.0
        quadrature[-1] = (wavelengths[-1] - wavelengths[-2]) / 2.0
        if len(wavelengths) > 2:
            quadrature[1:-1] = (wavelengths[2:] - wavelengths[:-2]) / 2.0
        weights *= quadrature
    if np.sum(weights) <= 0:
        raise ValueError("source spectral curves have no supported wavelength overlap")
    # Convert to metres after evaluating text curves, whose documented axis is nm.
    return tuple((wavelengths * 1e-9).tolist()), tuple(weights.tolist())


def iter_source_states(config: WFSConfig) -> tuple[SourceState, ...]:
    """Build normalized wavelength, angular, and LGS-range source states."""
    source = config.source
    wavelengths, spectral_weights = _spectral_samples(config)
    wavelength_weights = _normalised(spectral_weights, len(wavelengths))
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
