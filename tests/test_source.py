"""Guide-source quadrature and finite-range elongation tests."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from makewfs import WavefrontSensor, load_config
from makewfs.source import iter_source_states

SH_CONFIG = Path(__file__).parents[1] / "examples" / "configs" / "shack_hartmann_minimal.toml"


def test_wavelength_and_angular_quadrature_weights_normalize() -> None:
    config = load_config(SH_CONFIG)
    source = replace(
        config.source,
        wavelengths_m=(600e-9, 700e-9),
        wavelength_weights=(1.0, 3.0),
        angular_fwhm_arcsec=0.4,
        angular_quadrature_order=3,
    )
    states = iter_source_states(replace(config, source=source))
    assert len(states) == 18
    assert np.isclose(sum(state.weight for state in states), 1.0)
    assert np.isclose(
        sum(state.weight for state in states if np.isclose(state.wavelength_m, 600e-9)),
        0.25,
    )


def test_custom_sed_and_transmission_curves_drive_wavelength_weights(tmp_path: Path) -> None:
    config = load_config(SH_CONFIG)
    sed_path = tmp_path / "sed.txt"
    transmission_path = tmp_path / "transmission.txt"
    np.savetxt(sed_path, [[600.0, 1.0], [700.0, 2.0], [800.0, 1.0]])
    np.savetxt(transmission_path, [[600.0, 0.5], [700.0, 1.0], [800.0, 0.5]])
    source = replace(
        config.source,
        wavelengths_m=(),
        wavelength_weights=(),
        sed_path=str(sed_path),
        transmission_path=str(transmission_path),
    )
    states = iter_source_states(replace(config, source=source))
    assert np.allclose([state.wavelength_m for state in states], [600e-9, 700e-9, 800e-9])
    assert np.allclose([state.weight for state in states], [0.1, 0.8, 0.1])


def test_broadband_and_finite_source_maps_are_flux_preserving() -> None:
    config = load_config(SH_CONFIG)
    source = replace(
        config.source,
        wavelengths_m=(600e-9, 700e-9),
        wavelength_weights=(1.0, 1.0),
        angular_fwhm_arcsec=0.25,
    )
    sensor = WavefrontSensor(replace(config, source=source))
    rate = sensor.photon_rate(np.zeros(config.input.shape))
    assert np.all(rate >= 0)
    assert rate.sum() <= source.detector_photon_rate_per_s * (1.0 + 1e-12)
    assert rate.sum() > source.detector_photon_rate_per_s * 0.8


def test_thin_lgs_profile_matches_a_single_mean_range() -> None:
    config = load_config(SH_CONFIG)
    telescope = replace(config.telescope, central_obscuration_ratio=0.0)
    thin_source = replace(
        config.source,
        kind="lgs",
        lgs_ranges_m=(90e3,),
        lgs_range_weights=(1.0,),
    )
    profile_source = replace(
        config.source,
        kind="lgs",
        lgs_ranges_m=(89e3, 91e3),
        lgs_range_weights=(0.5, 0.5),
    )
    phase = np.zeros(config.input.shape)
    thin = WavefrontSensor(replace(config, telescope=telescope, source=thin_source)).photon_rate(
        phase
    )
    profile = WavefrontSensor(
        replace(config, telescope=telescope, source=profile_source)
    ).photon_rate(phase)
    assert not np.allclose(thin, profile, rtol=1e-6, atol=1e-10)
    assert np.isclose(thin.sum(), profile.sum(), rtol=1e-2)


def test_centre_launched_lgs_elongation_grows_toward_edge_subapertures() -> None:
    config = load_config(SH_CONFIG)
    source = replace(
        config.source,
        kind="lgs",
        lgs_ranges_m=(89e3, 91e3),
        lgs_range_weights=(0.5, 0.5),
    )
    telescope = replace(config.telescope, central_obscuration_ratio=0.0)
    sensor = WavefrontSensor(replace(config, telescope=telescope, source=source))
    mosaic = sensor.photon_rate(np.zeros(config.input.shape)).reshape(8, 8, 8, 8)
    _yy, xx = np.indices((8, 8), dtype=np.float64)

    def x_variance(spot: np.ndarray) -> float:
        weights = spot / spot.sum()
        mean = float(np.sum(weights * xx))
        return float(np.sum(weights * (xx - mean) ** 2))

    centre = x_variance(mosaic[4, 4])
    edge = x_variance(mosaic[4, 0])
    assert edge > centre * 1.01


def test_side_launched_lgs_elongation_tracks_launch_baseline() -> None:
    config = load_config(SH_CONFIG)
    source = replace(
        config.source,
        kind="lgs",
        lgs_ranges_m=(89e3, 91e3),
        lgs_range_weights=(0.5, 0.5),
        lgs_launch_position_m=(0.8, 0.0),
    )
    telescope = replace(config.telescope, central_obscuration_ratio=0.0)
    image = WavefrontSensor(replace(config, telescope=telescope, source=source)).photon_rate(
        np.zeros(config.input.shape)
    )
    mosaic = image.reshape(8, 8, 8, 8)
    _yy, xx = np.indices((8, 8), dtype=np.float64)

    def x_variance(spot: np.ndarray) -> float:
        weights = spot / spot.sum()
        mean = float(np.sum(weights * xx))
        return float(np.sum(weights * (xx - mean) ** 2))

    # A positive launch coordinate is closer to the right-hand subapertures,
    # so the left-hand rays have the larger range-dependent angular baseline.
    assert x_variance(mosaic[4, 0]) > x_variance(mosaic[4, 7])


def test_source_curve_failures_are_actionable(tmp_path: Path) -> None:
    config = load_config(SH_CONFIG)
    malformed = tmp_path / "malformed.txt"
    np.savetxt(malformed, [[700.0]])
    source = replace(config.source, sed_path=str(malformed))
    with pytest.raises(ValueError, match="two columns"):
        iter_source_states(replace(config, source=source))
    transmission = tmp_path / "transmission.txt"
    np.savetxt(transmission, [[600.0, 2.0], [700.0, 2.0]])
    source = replace(config.source, transmission_path=str(transmission))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        iter_source_states(replace(config, source=source))
