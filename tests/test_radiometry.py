"""Magnitude photon budgets account for light blocked inside the annulus."""

from dataclasses import replace

import numpy as np

from makewfs import WavefrontSensor, load_config
from makewfs.backend import cpu_backend
from makewfs.config import SourceConfig, SpiderConfig, TelescopeConfig
from makewfs.pupil import make_pupil
from makewfs.radiometry import clear_aperture_fraction

SH_CONFIG = "examples/configs/shack_hartmann_minimal.toml"
PYRAMID_CONFIG = "examples/configs/pyramid_minimal.toml"
MAGNITUDE = SourceConfig.from_dict({"normalization": "magnitude", "magnitude": 8, "band": "R"})


def _fraction(telescope: TelescopeConfig, n: int = 256) -> float:
    pupil = make_pupil(telescope, (n, n), telescope.pupil_diameter_m * 1.05, supersampling=4)
    return clear_aperture_fraction(
        pupil,
        telescope,
        (n, n),
        telescope.pupil_diameter_m * 1.05,
        supersampling=4,
        backend=cpu_backend(),
    )


def test_a_plain_annulus_keeps_every_photon() -> None:
    assert _fraction(TelescopeConfig(8.0, central_obscuration_ratio=0.14)) == 1.0


def test_wedge_spiders_remove_their_angular_share() -> None:
    # Each wedge spans width_fraction * pi of the 2 pi azimuth.
    # Pixel sampling biases the edges by O(1/n); the error must shrink with n.
    spiders = tuple(SpiderConfig(angle_deg=a, width_fraction=0.02) for a in (0, 90, 180, 270))
    telescope = TelescopeConfig(8.0, central_obscuration_ratio=0.14, spiders=spiders)
    expected = 1.0 - 4 * 0.02 / 2.0
    coarse = abs(_fraction(telescope, 256) - expected)
    fine = abs(_fraction(telescope, 512) - expected)
    assert fine < 2e-3
    assert fine < 0.7 * coarse


def test_segment_gaps_reduce_the_clear_fraction() -> None:
    gapped = TelescopeConfig(8.0, segments_across_pupil=4, segment_gap_fraction=0.05)
    assert 0.85 < _fraction(gapped) < 0.97


def _magnitude_engine(path: str, telescope_changes: dict) -> object:
    config = load_config(path)
    config = replace(
        config,
        source=MAGNITUDE,
        telescope=replace(config.telescope, **telescope_changes),
    )
    return WavefrontSensor(config).engine


def test_magnitude_source_rate_scales_with_the_clear_aperture() -> None:
    spiders = (
        SpiderConfig(angle_deg=0, width_fraction=0.05),
        SpiderConfig(angle_deg=180, width_fraction=0.05),
    )
    for path in (SH_CONFIG, PYRAMID_CONFIG):
        plain = _magnitude_engine(path, {})
        vaned = _magnitude_engine(path, {"spiders": spiders})
        assert plain.clear_aperture_fraction == 1.0
        assert 0.9 < vaned.clear_aperture_fraction < 0.98
        assert np.isclose(
            vaned.source_rate / plain.source_rate, vaned.clear_aperture_fraction, rtol=1e-12
        )


def test_a_direct_detector_rate_is_never_rescaled() -> None:
    config = load_config(SH_CONFIG)
    spiders = (SpiderConfig(angle_deg=0, width_fraction=0.05),)
    vaned = replace(config, telescope=replace(config.telescope, spiders=spiders))
    engine = WavefrontSensor(vaned).engine
    assert engine.clear_aperture_fraction == 1.0
    assert engine.source_rate == config.source.detector_photon_rate_per_s
