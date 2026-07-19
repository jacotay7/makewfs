"""Shared wavefront and pupil invariants."""

import numpy as np
import pytest

from makewfs.config import WFSConfig
from makewfs.pupil import make_pupil
from makewfs.wavefront import WavefrontInput, resample_opd


def _config(quantity: str = "opd") -> WFSConfig:
    data = {
        "schema_version": 1,
        "input": {
            "quantity": quantity,
            "unit": "m" if quantity == "opd" else "rad",
            "shape": [8, 8],
            "grid_extent_m": 2.0,
            "reference_wavelength_m": 500e-9,
        },
        "telescope": {"pupil_diameter_m": 2.0},
        "source": {"normalization": "detector_photon_rate", "detector_photon_rate_per_s": 1.0},
        "sensor": {"kind": "shack_hartmann", "wavelength_m": 500e-9},
        "shack_hartmann": {
            "lenslets_across_pupil": 2,
            "pixels_per_subaperture": 4,
            "spot_sampling_pixels_per_lambda_over_d": 2.0,
            "minimum_illuminated_fraction": 0.2,
        },
        "detector": {"preset": "generic_cmos", "exposure_s": 0.0},
    }
    return WFSConfig.from_dict(data)


def test_phase_to_opd_conversion() -> None:
    config = _config("phase")
    wavefront = WavefrontInput(config)
    phase = np.full(config.input.shape, np.pi)
    assert np.allclose(wavefront.opd(phase), 250e-9)


def test_opd_resampling_preserves_plane() -> None:
    source = np.tile(np.arange(5, dtype=float), (5, 1))
    result = resample_opd(source, (10, 10), 1.0)
    assert np.isfinite(result).all()
    assert np.isclose(result[5, 5] - result[5, 4], 0.5, atol=1e-12)


def test_circular_pupil_area_is_reasonable() -> None:
    config = _config()
    pupil = make_pupil(config.telescope, (256, 256), 2.0)
    expected = np.pi * (1.0**2) / (2.0**2)
    assert pupil.mean() == pytest.approx(expected, rel=0.02)


def test_pupil_supersampling_improves_boundary_area() -> None:
    config = _config()
    expected = np.pi / 4.0
    coarse = make_pupil(config.telescope, (32, 32), 2.0)
    fine = make_pupil(config.telescope, (32, 32), 2.0, supersampling=8)
    assert abs(float(fine.mean()) - expected) <= abs(float(coarse.mean()) - expected)
