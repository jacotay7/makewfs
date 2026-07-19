"""Shared wavefront and pupil invariants."""

import numpy as np
import pytest

from makewfs.config import WFSConfig
from makewfs.pupil import make_pupil
from makewfs.wavefront import WavefrontInput, iter_phase_samples, load_static_opd, resample_opd


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


def test_wavefront_shape_type_finiteness_and_static_map(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = _config()
    with pytest.raises(ValueError, match="shape"):
        WavefrontInput(config).opd(np.zeros((4, 4)))
    with pytest.raises(TypeError, match="numeric"):
        WavefrontInput(config).opd(np.full(config.input.shape, "bad", dtype=object))
    with pytest.raises(ValueError, match="non-finite"):
        WavefrontInput(config).opd(np.full(config.input.shape, np.nan))
    static_path = tmp_path / "static.npy"
    np.save(static_path, np.ones(config.input.shape) * 2e-9)
    static_config = WFSConfig.from_dict(
        {
            **config.to_dict(),
            "input": {**config.to_dict()["input"], "static_opd_path": str(static_path)},
        }
    )
    static = load_static_opd(static_config)
    assert static is not None
    assert np.allclose(
        WavefrontInput(static_config, static).opd(np.zeros(config.input.shape)), 2e-9
    )


def test_static_map_archive_and_invalid_shape(tmp_path) -> None:  # type: ignore[no-untyped-def]
    config = _config()
    archive = tmp_path / "static.npz"
    np.savez(archive, data=np.zeros(config.input.shape))
    archive_config = WFSConfig.from_dict(
        {
            **config.to_dict(),
            "input": {**config.to_dict()["input"], "static_opd_path": str(archive)},
        }
    )
    assert load_static_opd(archive_config) is not None
    bad = tmp_path / "bad.npy"
    np.save(bad, np.zeros((3, 3)))
    bad_config = WFSConfig.from_dict(
        {**config.to_dict(), "input": {**config.to_dict()["input"], "static_opd_path": str(bad)}}
    )
    with pytest.raises(ValueError, match="does not match"):
        load_static_opd(bad_config)


def test_phase_sample_iterator_supports_snapshot_stack_and_iterable() -> None:
    config = _config()
    snapshot = np.zeros(config.input.shape)
    assert len(list(iter_phase_samples(snapshot, config.input.shape))) == 1
    stack = np.zeros((3, *config.input.shape))
    assert len(list(iter_phase_samples(stack, config.input.shape))) == 3
    assert len(list(iter_phase_samples([snapshot, snapshot], config.input.shape))) == 2
    with pytest.raises(ValueError, match="end in shape"):
        list(iter_phase_samples(np.zeros((2, 4, 4)), config.input.shape))
    with pytest.raises(ValueError, match="sample must have shape"):
        list(iter_phase_samples([np.zeros((4, 4))], config.input.shape))


def test_opd_resampling_preserves_plane() -> None:
    source = np.tile(np.arange(5, dtype=float), (5, 1))
    result = resample_opd(source, (10, 10), 1.0)
    assert np.isfinite(result).all()
    assert np.isclose(result[5, 5] - result[5, 4], 0.5, atol=1e-12)


def test_opd_resampling_preserves_y_plane_and_constant() -> None:
    source_y = np.tile(np.arange(5, dtype=float)[:, None], (1, 5))
    result_y = resample_opd(source_y, (10, 10), 1.0)
    assert np.isclose(result_y[5, 5] - result_y[4, 5], 0.5, atol=1e-12)
    constant = resample_opd(np.full((5, 7), 3.2), (9, 11), 1.0)
    assert np.allclose(constant, 3.2)


def test_opd_resampling_preserves_low_order_defocus() -> None:
    yy, xx = np.mgrid[:7, :7]
    x = (xx - 3.0) / 7.0
    y = (yy - 3.0) / 7.0
    source = x**2 + y**2
    result = resample_opd(source, (14, 14), 1.0)
    target_y, target_x = np.mgrid[:14, :14]
    expected = ((target_x - 6.5) / 14.0) ** 2 + ((target_y - 6.5) / 14.0) ** 2
    assert np.allclose(result[2:-2, 2:-2], expected[2:-2, 2:-2], atol=0.03)
    assert resample_opd(source.astype(np.float32), (14, 14), 1.0).dtype == np.float32


def test_phase_is_converted_before_resampling_without_wrapping() -> None:
    config = _config("phase")
    wavefront = WavefrontInput(config)
    phase = np.linspace(-12.0 * np.pi, 12.0 * np.pi, config.input.shape[1])[None, :]
    phase = np.broadcast_to(phase, config.input.shape)
    result = wavefront.opd(phase, target_shape=(16, 16))
    source_index = np.clip((np.arange(16, dtype=float) + 0.5) * 8.0 / 16.0 - 0.5, 0, 7)
    expected = (-12.0 * np.pi + source_index / 7.0 * 24.0 * np.pi)[None, :]
    expected = np.broadcast_to(expected, (16, 16)) * 500e-9 / (2.0 * np.pi)
    assert np.allclose(result, expected, atol=2e-8)


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
