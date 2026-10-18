"""Per-frame pupil amplitude, subaperture flux, spatial filter and cross-talk (ADR 0006).

The amplitude tests fix the physics against closed-form subaperture integrals
and against the same light delivered through a static pupil mask; the spatial
filter against Gaussian-windowed plane waves and an independent zero-padded
FFT; cross-talk against an overlap-add of wider, independently propagated
windows.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from makewfs import ConfigError, WavefrontSensor, load_config, simulate
from makewfs.config import WFSConfig
from makewfs.sampling import spatial_filter_half_width, square_spatial_filter

ROOT = Path(__file__).parents[1]
SH_CONFIG = ROOT / "examples" / "configs" / "shack_hartmann_minimal.toml"
PYRAMID_CONFIG = ROOT / "examples" / "configs" / "pyramid_minimal.toml"
LENSLETS = 8
SAMPLES = 16
EXTENT_M = 2.0
WAVELENGTH_M = 5.0e-7


def _square_config(
    *,
    pixels: int = 8,
    sampling: float = 2.0,
    dtype: str = "float64",
    shack_hartmann: dict[str, Any] | None = None,
    source: dict[str, Any] | None = None,
    telescope: dict[str, Any] | None = None,
    margin: int = 0,
) -> WFSConfig:
    """Fully illuminated square lenslets: the pupil is wider than the grid diagonal.

    The input grid equals the propagation grid, so neither OPD nor amplitude
    is resampled.
    """
    size = LENSLETS * SAMPLES
    return WFSConfig.from_dict(
        {
            "schema_version": 1,
            "input": {"shape": [size, size], "grid_extent_m": EXTENT_M},
            "telescope": {"pupil_diameter_m": 3.0, **(telescope or {})},
            "source": {
                "kind": "ngs",
                "normalization": "detector_photon_rate",
                "detector_photon_rate_per_s": 1.0e6,
                **(source or {}),
            },
            "sensor": {"kind": "shack_hartmann", "wavelength_m": WAVELENGTH_M},
            "shack_hartmann": {
                "lenslets_across_pupil": LENSLETS,
                "pixels_per_subaperture": pixels,
                "spot_sampling_pixels_per_lambda_over_d": sampling,
                "detector_margin_pixels": margin,
                **(shack_hartmann or {}),
            },
            "detector": {"preset": "generic_cmos", "exposure_s": 0.001},
            "numerics": {"dtype": dtype, "pupil_samples_per_lenslet": SAMPLES},
        }
    )


def _coordinates(config: WFSConfig) -> tuple[np.ndarray, np.ndarray]:
    """Pixel-centre ``(x, y)`` coordinates of the input grid in metres."""
    height, width = config.input.shape
    pitch = config.input.grid_extent_m / width
    x = (np.arange(width) - (width - 1) / 2) * pitch
    y = (np.arange(height) - (height - 1) / 2) * pitch
    return np.meshgrid(x, y)


def _opd(shape: tuple[int, int], seed: int = 11, rms_m: float = 40e-9) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=shape) * rms_m


# --------------------------------------------------------------------------- config


def test_new_shack_hartmann_keys_validate_and_serialize() -> None:
    data = load_config(SH_CONFIG).to_dict()
    # Omitted at their defaults, so existing digests do not move.
    assert "spatial_filter_width_lambda_over_d" not in data["shack_hartmann"]
    assert "crosstalk_reach_subapertures" not in data["shack_hartmann"]
    data["shack_hartmann"]["spatial_filter_width_lambda_over_d"] = 1.5
    data["shack_hartmann"]["crosstalk_reach_subapertures"] = 2
    config = WFSConfig.from_dict(data)
    assert config.shack_hartmann is not None
    assert config.shack_hartmann.spatial_filter_width_lambda_over_d == 1.5
    assert config.shack_hartmann.crosstalk_reach_subapertures == 2
    assert WFSConfig.from_dict(config.to_dict()).digest == config.digest
    assert config.digest != load_config(SH_CONFIG).digest
    for key, value, match in (
        ("spatial_filter_width_lambda_over_d", 0.0, "must be > 0"),
        ("spatial_filter_width_lambda_over_d", -1.0, "must be > 0"),
        ("spatial_filter_width_lambda_over_d", float("nan"), "finite"),
        ("spatial_filter_width_lambda_over_d", "wide", "finite"),
        ("crosstalk_reach_subapertures", -1, "integer >= 0"),
        ("crosstalk_reach_subapertures", 1.5, "integer >= 0"),
        ("crosstalk_reach_subapertures", True, "integer >= 0"),
    ):
        broken = load_config(SH_CONFIG).to_dict()
        broken["shack_hartmann"][key] = value
        with pytest.raises(ConfigError, match=f"shack_hartmann.{key}: .*{match}"):
            WFSConfig.from_dict(broken)


def test_new_keys_reject_incompatible_lgs_models() -> None:
    lgs = {
        "kind": "lgs",
        "lgs_ranges_m": [88e3, 92e3],
        "lgs_launch_position_m": [1.0, 0.0],
    }
    data = _square_config(source=lgs).to_dict()
    data["shack_hartmann"]["spatial_filter_width_lambda_over_d"] = 2.0
    with pytest.raises(ConfigError, match="range-resolved LGS"):
        WFSConfig.from_dict(data)
    data = _square_config(source={**lgs, "lgs_model": "kernel"}).to_dict()
    data["shack_hartmann"]["crosstalk_reach_subapertures"] = 1
    with pytest.raises(ConfigError, match="lgs_model = 'kernel'"):
        WFSConfig.from_dict(data)
    # Engines refuse the same combinations from a programmatically built config.
    config = _square_config(source=lgs)
    assert config.shack_hartmann is not None
    with pytest.raises(ValueError, match="range-resolved LGS"):
        WavefrontSensor(
            replace(
                config,
                shack_hartmann=replace(
                    config.shack_hartmann, spatial_filter_width_lambda_over_d=2.0
                ),
            )
        )
    config = _square_config(source={**lgs, "lgs_model": "kernel"})
    assert config.shack_hartmann is not None
    with pytest.raises(ValueError, match="lgs_model = 'kernel'"):
        WavefrontSensor(
            replace(
                config,
                shack_hartmann=replace(config.shack_hartmann, crosstalk_reach_subapertures=1),
            )
        )


# ------------------------------------------------------------------------ amplitude


def _rotated_offset(config: WFSConfig) -> WFSConfig:
    assert config.shack_hartmann is not None
    return replace(
        config,
        shack_hartmann=replace(
            config.shack_hartmann,
            lenslet_grid_rotation_deg=4.0,
            lenslet_grid_offset_fraction=(0.2, -0.1),
        ),
    )


def _with_dtype(config: WFSConfig, dtype: str) -> WFSConfig:
    return replace(config, numerics=replace(config.numerics, dtype=dtype))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize(
    "make",
    [
        lambda: load_config(SH_CONFIG),
        lambda: _rotated_offset(load_config(SH_CONFIG)),
        lambda: load_config(
            ROOT / "benchmarks" / "configs" / "shack_hartmann_quadrature_9sample.toml"
        ),
        lambda: load_config(PYRAMID_CONFIG),
        lambda: load_config(ROOT / "benchmarks" / "configs" / "pyramid_60_mod8_float32.toml"),
    ],
    ids=["sh", "sh-rotated-offset", "sh-broadband-lgs", "pyramid", "pyramid-modulated"],
)
def test_unit_amplitude_reproduces_opd_only_results_bit_for_bit(make: Any, dtype: str) -> None:
    config = _with_dtype(make(), dtype)
    sensor = WavefrontSensor(config)
    opd = _opd(config.input.shape)
    ones = np.ones(config.input.shape)
    np.testing.assert_array_equal(sensor.photon_rate(opd, amplitude=ones), sensor.photon_rate(opd))
    stack = np.stack([opd, 0.5 * opd])
    plain = sensor.expose_integrated(stack, seed=3)
    scintillated = sensor.expose_integrated(stack, amplitude=[ones, ones], seed=3)
    np.testing.assert_array_equal(scintillated.truth.photon_rate, plain.truth.photon_rate)
    np.testing.assert_array_equal(scintillated.data, plain.data)
    assert scintillated.metadata["wfs_input_amplitude"] is True
    assert plain.metadata["wfs_input_amplitude"] is False


def test_unit_amplitude_on_a_resampled_grid_is_still_exact() -> None:
    """A 100-sample input feeds a 104-sample lenslet grid (non-integer ratio)."""
    config = load_config(SH_CONFIG)
    config = replace(
        config,
        input=replace(config.input, shape=(100, 100)),
        numerics=replace(config.numerics, dtype="float64"),
    )
    sensor = WavefrontSensor(config)
    assert sensor.engine.internal_shape == (104, 104)
    opd = _opd(config.input.shape)
    np.testing.assert_array_equal(
        sensor.photon_rate(opd, amplitude=np.ones(config.input.shape)), sensor.photon_rate(opd)
    )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_constant_amplitude_scales_the_photon_rate_by_its_square(dtype: str) -> None:
    for config in (_with_dtype(load_config(SH_CONFIG), dtype), load_config(PYRAMID_CONFIG)):
        sensor = WavefrontSensor(config)
        opd = _opd(config.input.shape)
        reference = sensor.expose(opd, seed=1)
        frame = sensor.expose(opd, amplitude=np.full(config.input.shape, 0.5), seed=1)
        np.testing.assert_allclose(
            frame.truth.photon_rate, 0.25 * reference.truth.photon_rate, rtol=1e-6, atol=0.0
        )
        assert frame.metadata["wfs_captured_photons_s"] == pytest.approx(
            0.25 * reference.metadata["wfs_captured_photons_s"], rel=1e-6
        )
        assert (
            frame.metadata["wfs_launched_photons_s"] == reference.metadata["wfs_launched_photons_s"]
        )


def test_sinusoidal_intensity_gives_the_expected_subaperture_flux() -> None:
    """``I = 1 + m sin(2 pi x / Lambda)`` across fully illuminated lenslets.

    The pupil-plane flux of lenslet ``j`` is the source rate over ``n^2``
    times the lenslet mean of ``I``: ``1 + m D sin(2 pi x_j / Lambda)``, where
    ``D = sin(pi s dx / Lambda) / (s sin(pi dx / Lambda))`` is the exact
    ``s``-sample average (the continuous ``sinc(d / Lambda)`` as ``s`` grows).
    A y modulation must move only the rows, an x modulation only the columns.
    """
    config = _square_config()
    sensor = WavefrontSensor(config)
    xx, yy = _coordinates(config)
    pitch = EXTENT_M / LENSLETS
    sample = EXTENT_M / (LENSLETS * SAMPLES)
    period = 4.0 * pitch
    depth = 0.3
    average = np.sin(np.pi * SAMPLES * sample / period) / (
        SAMPLES * np.sin(np.pi * sample / period)
    )
    centres = (np.arange(LENSLETS) - (LENSLETS - 1) / 2) * pitch
    expected = (
        sensor.engine.source_rate
        / LENSLETS**2
        * (1.0 + depth * average * np.sin(2.0 * np.pi * centres / period))
    )
    zero = np.zeros(config.input.shape)
    flat = sensor.subaperture_flux(zero)
    for axis, coordinate in (("x", xx), ("y", yy)):
        intensity = 1.0 + depth * np.sin(2.0 * np.pi * coordinate / period)
        flux = sensor.subaperture_flux(zero, amplitude=np.sqrt(intensity))
        target = expected[None, :] if axis == "x" else expected[:, None]
        np.testing.assert_allclose(
            flux.pupil_photons_s, np.broadcast_to(target, (LENSLETS, LENSLETS)), rtol=1e-12
        )
        # The detector window captures nearly the same fraction of every
        # spot, so its flux follows the pupil flux once divided by the flat
        # (amplitude-one) calibration. The residual is the window loss changed
        # by the intensity gradient across a lenslet: 0.26% here.
        np.testing.assert_allclose(
            flux.detector_photons_s / flat.detector_photons_s,
            flux.pupil_photons_s / flat.pupil_photons_s,
            rtol=5e-3,
        )
        np.testing.assert_allclose(
            np.sum(flux.detector_photons_s), np.sum(flux.photon_rate), rtol=1e-12
        )
    assert np.all(flat.valid)
    np.testing.assert_allclose(np.sum(flat.pupil_photons_s), sensor.engine.source_rate, rtol=1e-12)


def test_amplitude_matches_the_same_pattern_in_a_static_pupil_mask(tmp_path: Path) -> None:
    """The amplitude multiplies the pupil: a mask ``M A`` gives the same image.

    Only the source normalization differs, because the configured rate is
    defined for the unscintillated pupil ``M``.
    """
    rng = np.random.default_rng(4)
    for config in (_square_config(), load_config(PYRAMID_CONFIG)):
        if config.sensor.kind == "pyramid":
            config = replace(
                config,
                input=replace(config.input, shape=(64, 64)),
                numerics=replace(config.numerics, dtype="float64"),
            )
        sensor = WavefrontSensor(config)
        mask = np.asarray(sensor.engine.configured_pupil, dtype=np.float64)
        assert mask.shape == config.input.shape
        amplitude = 0.5 + 0.5 * rng.random(config.input.shape)
        np.save(tmp_path / "mask.npy", mask)
        np.save(tmp_path / "masked.npy", mask * amplitude)
        with_mask = WavefrontSensor(
            replace(
                config,
                telescope=replace(config.telescope, custom_mask_path=str(tmp_path / "mask.npy")),
            )
        )
        masked = WavefrontSensor(
            replace(
                config,
                telescope=replace(config.telescope, custom_mask_path=str(tmp_path / "masked.npy")),
            )
        )
        opd = _opd(config.input.shape)
        normalization = np.sum(mask**2) / np.sum((mask * amplitude) ** 2)
        np.testing.assert_allclose(
            with_mask.photon_rate(opd, amplitude=amplitude),
            masked.photon_rate(opd) / normalization,
            rtol=1e-10,
            atol=1e-12 * float(np.max(masked.photon_rate(opd))),
        )


def test_amplitude_intensity_is_area_averaged_onto_the_lenslet_grid() -> None:
    """Each lenslet receives the integral of the input intensity over it."""
    rng = np.random.default_rng(8)
    for size in (64, 100):  # integer (x2) and non-integer (104 / 100) refinement
        config = _square_config()
        config = replace(config, input=replace(config.input, shape=(size, size)))
        sensor = WavefrontSensor(config)
        intensity = 0.2 + rng.random((size, size))
        flux = sensor.subaperture_flux(np.zeros((size, size)), amplitude=np.sqrt(intensity))
        np.testing.assert_allclose(
            np.sum(flux.pupil_photons_s),
            sensor.engine.source_rate * np.mean(intensity),
            rtol=1e-12,
        )
        if size == 64:
            blocks = intensity.reshape(LENSLETS, 8, LENSLETS, 8).sum(axis=(1, 3))
            np.testing.assert_allclose(
                flux.pupil_photons_s,
                sensor.engine.source_rate * blocks / intensity.size,
                rtol=1e-12,
            )


def test_per_wavelength_amplitude_stack_scales_each_spectral_plane() -> None:
    config = _square_config(source={"wavelengths_m": [4.5e-7, 5.5e-7]})
    sensor = WavefrontSensor(config)
    assert sensor.wavelengths_m == (4.5e-7, 5.5e-7)
    opd = _opd(config.input.shape)
    reference = sensor.engine.render(opd)
    stack = np.stack([np.full(config.input.shape, 0.5), np.full(config.input.shape, 2.0)])
    result = sensor.engine.render(opd, amplitude=stack)
    assert result.spectral_photon_rate is not None
    assert reference.spectral_photon_rate is not None
    np.testing.assert_allclose(
        result.spectral_photon_rate,
        reference.spectral_photon_rate * np.array([0.25, 4.0])[:, None, None],
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        sensor.photon_rate(opd, amplitude=stack), np.sum(result.spectral_photon_rate, axis=0)
    )
    one = np.full(config.input.shape, 0.7)
    np.testing.assert_array_equal(
        sensor.photon_rate(opd, amplitude=one), sensor.photon_rate(opd, amplitude=one[None])
    )
    with pytest.raises(ValueError, match="wavelength nodes"):
        sensor.photon_rate(opd, amplitude=np.ones((3, *config.input.shape)))


def test_integrated_amplitudes_may_be_per_wavelength_or_mixed() -> None:
    """Temporal samples average; each may be one map or a per-wavelength stack."""
    config = _square_config(source={"wavelengths_m": [4.5e-7, 5.5e-7]})
    sensor = WavefrontSensor(config)
    shape = config.input.shape
    first, second = _opd(shape, seed=1), _opd(shape, seed=2)
    low = np.stack([np.full(shape, 0.5), np.full(shape, 0.8)])
    high = np.full(shape, 1.2)
    integrated = sensor.engine.render_integrated([first, second], amplitudes=[low, high])
    separate = [
        sensor.engine.render(first, amplitude=low),
        sensor.engine.render(second, amplitude=np.stack([high, high])),
    ]
    np.testing.assert_allclose(
        integrated.photon_rate,
        0.5 * (separate[0].photon_rate + separate[1].photon_rate),
        rtol=1e-12,
        atol=1e-12 * float(np.max(separate[0].photon_rate)),
    )
    frame = sensor.expose_integrated(np.stack([first, second]), amplitude=[low, high], seed=4)
    np.testing.assert_allclose(frame.truth.photon_rate, integrated.photon_rate, rtol=1e-12)
    with pytest.raises(ValueError, match="one amplitude per wavefront sample"):
        sensor.engine.render_integrated([first, second], amplitudes=[low])
    # One sample with one plain map.
    single = sensor.expose_integrated(first, amplitude=high, seed=4)
    np.testing.assert_array_equal(
        single.truth.photon_rate, sensor.photon_rate(first, amplitude=high)
    )


def test_filter_batches_mixing_wavelengths_are_filtered_one_by_one() -> None:
    """A batch of states at different wavelengths gets each its own passband."""
    config = _square_config(
        shack_hartmann={"spatial_filter_width_lambda_over_d": 1.0},
        source={"wavelengths_m": [4.0e-7, 6.0e-7]},
    )
    engine = WavefrontSensor(config).engine
    geometry = engine.geometry(0)
    rng = np.random.default_rng(6)
    phasor = np.exp(1j * rng.normal(size=(2, *engine.internal_shape)))
    batch = engine._lenslet_fields(phasor, geometry, None, (4.0e-7, 6.0e-7))
    for index, wavelength in enumerate((4.0e-7, 6.0e-7)):
        single = engine._lenslet_fields(phasor[index : index + 1], geometry, None, (wavelength,))
        np.testing.assert_array_equal(batch[index], single[0])
    assert engine._spatial_filter_half_widths == {4.0e-7: 9, 6.0e-7: 6}


def test_amplitude_inputs_are_validated() -> None:
    sensor = WavefrontSensor(load_config(SH_CONFIG))
    shape = sensor.config.input.shape
    opd = np.zeros(shape)
    bad = np.ones(shape)
    bad[3, 4] = -0.1
    with pytest.raises(ValueError, match="finite and non-negative"):
        sensor.photon_rate(opd, amplitude=bad)
    bad[3, 4] = np.nan
    with pytest.raises(ValueError, match="finite and non-negative"):
        sensor.photon_rate(opd, amplitude=bad)
    with pytest.raises(TypeError, match="must be real"):
        sensor.photon_rate(opd, amplitude=np.ones(shape, dtype=complex))
    with pytest.raises(TypeError, match="numeric"):
        sensor.photon_rate(opd, amplitude=np.full(shape, "a"))
    with pytest.raises(ValueError, match="amplitude must have shape"):
        sensor.photon_rate(opd, amplitude=np.ones((64, 64)))
    with pytest.raises(ValueError, match="one entry per temporal sample"):
        sensor.expose_integrated(np.stack([opd, opd]), amplitude=[np.ones(shape)])
    pyramid = WavefrontSensor(load_config(PYRAMID_CONFIG))
    with pytest.raises(ValueError, match="one entry per temporal sample"):
        pyramid.expose_integrated(np.stack([opd, opd]), amplitude=np.ones((3, *shape)))
    with pytest.raises(ValueError, match="only for Shack--Hartmann"):
        pyramid.subaperture_flux(opd)


def test_pyramid_pupil_images_carry_the_amplitude_with_the_input_orientation() -> None:
    config = load_config(PYRAMID_CONFIG)
    sensor = WavefrontSensor(config)
    height, width = config.input.shape
    amplitude = np.ones((height, width))
    amplitude[:, width // 2 :] = 0.5  # the +x half of the pupil is dimmer
    opd = np.zeros((height, width))
    flat = sensor.photon_rate(opd)
    dim = sensor.photon_rate(opd, amplitude=amplitude)
    assert config.pyramid is not None
    pixels = config.pyramid.pixels_across_pupil
    separation = config.pyramid.pupil_separation_pixels
    centre = (flat.shape[0] - 1) / 2
    for sign_y in (-1, 1):
        for sign_x in (-1, 1):
            y0 = round(centre + sign_y * separation / 2 - (pixels - 1) / 2)
            x0 = round(centre + sign_x * separation / 2 - (pixels - 1) / 2)
            ratio = dim[y0 : y0 + pixels, x0 : x0 + pixels] / np.maximum(
                flat[y0 : y0 + pixels, x0 : x0 + pixels], 1e-30
            )
            left = np.mean(ratio[:, : pixels // 2 - 2])
            right = np.mean(ratio[:, pixels // 2 + 2 :])
            # Each pupil image keeps the input's +x direction (towards larger
            # column index); diffraction across the step blurs the 0.25 ratio.
            assert right < 0.5 < left


def test_seeded_exposures_with_amplitude_repeat() -> None:
    sensor = WavefrontSensor(load_config(SH_CONFIG))
    shape = sensor.config.input.shape
    opd = _opd(shape)
    amplitude = np.sqrt(1.0 + 0.2 * np.sin(np.arange(shape[1]) / 7.0))[None, :] * np.ones(shape)
    first = sensor.expose(opd, amplitude=amplitude, seed=9)
    second = sensor.expose(opd, amplitude=amplitude, seed=9)
    other = sensor.expose(opd, amplitude=amplitude, seed=10)
    np.testing.assert_array_equal(first.data, second.data)
    assert not np.array_equal(first.data, other.data)
    frames = list(sensor.expose_many([opd, opd], seeds=[9, 9], amplitudes=[amplitude, None]))
    np.testing.assert_array_equal(frames[0].data, first.data)
    assert frames[0].metadata["wfs_input_amplitude"] is True
    assert frames[1].metadata["wfs_input_amplitude"] is False
    np.testing.assert_array_equal(frames[1].data, sensor.expose(opd, seed=9).data)
    one_shot = simulate(opd, sensor.config, seed=9, amplitude=amplitude)
    np.testing.assert_array_equal(one_shot.data, first.data)


def test_amplitude_keeps_the_working_precision() -> None:
    config = _with_dtype(load_config(SH_CONFIG), "float32")
    sensor = WavefrontSensor(config)
    engine = sensor.engine
    geometry = engine.geometry(0)
    maps = engine.lenslet_amplitudes([np.ones(config.input.shape)], geometry)
    assert maps.dtype == np.float32
    internal = np.zeros(engine.internal_shape, dtype=np.float32)
    field = engine._field(internal, engine.source_states[0], 0, geometry, amplitude=maps[0, 0])
    assert field.dtype == np.complex64
    filtered = square_spatial_filter(field[None], half_width=4)
    assert filtered.dtype == np.complex64


# -------------------------------------------------------------------- spatial filter


def _gaussian_carrier(size: int, frequency_bins: tuple[float, float]) -> np.ndarray:
    """A plane wave of ``frequency_bins`` (padded-grid units) under a Gaussian window."""
    coordinate = np.arange(size) - (size - 1) / 2
    xx, yy = np.meshgrid(coordinate, coordinate)
    window = np.exp(-(xx**2 + yy**2) / (2.0 * (size / 8.0) ** 2))
    fx, fy = frequency_bins
    return window * np.exp(2j * np.pi * (fx * xx + fy * yy) / (2 * size))


def test_square_spatial_filter_passes_inside_and_removes_outside_its_cutoff() -> None:
    """A single Fourier component, windowed so its spectrum is a few bins wide.

    The Gaussian window of width ``N/8`` has a spectral standard deviation of
    ``2N / (2 pi N / 8) = 2.5`` bins on the padded grid, so a carrier 15 bins
    (six of those) inside the passband keeps all but ~1e-9 of its energy and
    one 15 bins outside loses all but ~1e-9.
    """
    size = 128
    half_width = 40
    for fx, fy in ((0.0, 0.0), (half_width - 15, 0.0), (0.0, -(half_width - 15)), (20, 20)):
        field = _gaussian_carrier(size, (fx, fy))
        passed = square_spatial_filter(field, half_width=half_width)
        assert np.sum(np.abs(passed) ** 2) / np.sum(np.abs(field) ** 2) > 1 - 1e-8
        np.testing.assert_allclose(passed, field, atol=1e-3 * np.max(np.abs(field)))
    for fx, fy in ((half_width + 15, 0.0), (0.0, half_width + 15), (-(half_width + 15), 5.0)):
        field = _gaussian_carrier(size, (fx, fy))
        blocked = square_spatial_filter(field, half_width=half_width)
        assert np.sum(np.abs(blocked) ** 2) / np.sum(np.abs(field) ** 2) < 1e-8
    # Every passband line kept, every other one removed, checked on the
    # padded spectrum of a random field.
    rng = np.random.default_rng(1)
    field = rng.normal(size=(2, size, size)) + 1j * rng.normal(size=(2, size, size))
    filtered = square_spatial_filter(field, half_width=10)
    padded = np.zeros((2, 2 * size, 2 * size), dtype=complex)
    padded[:, :size, :size] = field
    spectrum = np.fft.fft2(padded)
    index = np.fft.fftfreq(2 * size) * 2 * size
    passband = (np.abs(index)[None, :] <= 10) & (np.abs(index)[:, None] <= 10)
    expected = np.fft.ifft2(spectrum * passband)[:, :size, :size]
    np.testing.assert_allclose(filtered, expected, atol=1e-12 * np.max(np.abs(expected)))
    assert square_spatial_filter(field, half_width=size) is field
    with pytest.raises(ValueError, match="square"):
        square_spatial_filter(field[:, :, :10], half_width=2)
    with pytest.raises(ValueError, match="non-negative"):
        square_spatial_filter(field, half_width=-1)


def test_spatial_filter_cutoff_is_a_fixed_angle() -> None:
    """``|m| < c n lambda_ref / lambda`` on the twice-padded grid; the edge is blocked."""
    kwargs = {"lenslets": 8, "reference_wavelength_m": 7e-7}
    assert spatial_filter_half_width(2.0, wavelength_m=7e-7, **kwargs) == 15
    assert spatial_filter_half_width(2.0, wavelength_m=14e-7, **kwargs) == 7
    assert spatial_filter_half_width(1.0, wavelength_m=7e-7, **kwargs) == 7
    assert (
        spatial_filter_half_width(1.1, wavelength_m=7e-7, lenslets=10, reference_wavelength_m=7e-7)
        == 10
    )
    assert spatial_filter_half_width(0.9375, wavelength_m=7e-7, **kwargs) == 7
    assert spatial_filter_half_width(0.01, wavelength_m=7e-7, **kwargs) == 0
    for bad in ({"width": 0.0}, {"width": float("inf")}):
        with pytest.raises(ValueError, match="finite and positive"):
            spatial_filter_half_width(bad["width"], wavelength_m=7e-7, **kwargs)
    with pytest.raises(ValueError, match="lenslets"):
        spatial_filter_half_width(1.0, wavelength_m=7e-7, lenslets=0, reference_wavelength_m=7e-7)
    with pytest.raises(ValueError, match="wavelengths"):
        spatial_filter_half_width(1.0, wavelength_m=0.0, **kwargs)


def test_sensor_spatial_filter_matches_an_independent_padded_fft() -> None:
    """The lenslet entrance flux behind the filter, against a direct NumPy model.

    The model follows the prototype it reproduces: the pupil field (pupil
    times amplitude times ``exp(2 pi i OPD / lambda)``) is zero-padded twice,
    transformed, masked to ``|f| < c / (2 d) lambda_ref / lambda``, transformed
    back and cropped, and each lenslet's intensity summed.
    """
    width = 1.5
    # At the reference wavelength the cutoff falls exactly on a padded-grid
    # frequency, which both models block; at 0.77 of it, between two.
    for wavelength in (WAVELENGTH_M, 0.77 * WAVELENGTH_M):
        config = _square_config(
            shack_hartmann={"spatial_filter_width_lambda_over_d": width},
            telescope={"pupil_diameter_m": 1.8, "central_obscuration_ratio": 0.2},
            source={"wavelengths_m": [wavelength]},
        )
        sensor = WavefrontSensor(config)
        size = config.input.shape[0]
        xx, yy = _coordinates(config)
        opd = 60e-9 * np.sin(2 * np.pi * (3 * xx + 2 * yy) / EXTENT_M) + _opd((size, size))
        amplitude = np.sqrt(1.0 + 0.3 * np.cos(2 * np.pi * 5 * xx / EXTENT_M))
        flux = sensor.subaperture_flux(opd, amplitude=amplitude)

        pupil = np.asarray(sensor.engine.configured_pupil)
        field = pupil * amplitude * np.exp(2j * np.pi * opd / wavelength)
        padded = np.zeros((2 * size, 2 * size), dtype=complex)
        padded[:size, :size] = field
        frequency = np.fft.fftfreq(2 * size, d=EXTENT_M / size)
        cutoff = width / (2 * EXTENT_M / LENSLETS) * WAVELENGTH_M / wavelength
        mask = (np.abs(frequency)[None, :] < cutoff) & (np.abs(frequency)[:, None] < cutoff)
        filtered = np.fft.ifft2(np.fft.fft2(padded) * mask)[:size, :size]
        energy = np.abs(filtered) ** 2
        per_lenslet = energy.reshape(LENSLETS, SAMPLES, LENSLETS, SAMPLES).sum(axis=(1, 3))
        expected = sensor.engine.source_rate * per_lenslet / np.sum(pupil**2)
        np.testing.assert_allclose(flux.pupil_photons_s, expected, rtol=1e-9, atol=1e-9 * 1e6)
        # The filter only removes light.
        unfiltered = WavefrontSensor(
            replace(
                config,
                shack_hartmann=replace(
                    config.shack_hartmann, spatial_filter_width_lambda_over_d=None
                ),
            )
        ).subaperture_flux(opd, amplitude=amplitude)
        assert np.sum(flux.pupil_photons_s) < np.sum(unfiltered.pupil_photons_s)


def test_wide_spatial_filter_is_transparent_and_fill_factor_applies_after_it() -> None:
    wide = _square_config(shack_hartmann={"spatial_filter_width_lambda_over_d": 4.0 * SAMPLES})
    plain = _square_config()
    opd = _opd(plain.input.shape)
    np.testing.assert_array_equal(
        WavefrontSensor(wide).photon_rate(opd), WavefrontSensor(plain).photon_rate(opd)
    )
    # With a lenslet fill factor, the filtered field is masked by the lenslet
    # apertures afterwards: a transparent filter then equals the plain sensor.
    filled = _square_config(
        shack_hartmann={
            "spatial_filter_width_lambda_over_d": 4.0 * SAMPLES,
            "lenslet_fill_factor": 0.75,
        }
    )
    unfiltered = _square_config(shack_hartmann={"lenslet_fill_factor": 0.75})
    np.testing.assert_allclose(
        WavefrontSensor(filled).photon_rate(opd),
        WavefrontSensor(unfiltered).photon_rate(opd),
        rtol=1e-14,
        atol=0.0,
    )


def test_spatial_filter_turns_phase_into_lenslet_intensity() -> None:
    """A phase grating beyond the cutoff loses its light to the filter.

    A pure phase ripple leaves the pupil intensity flat; its diffracted
    orders at ``+-f`` are blocked when ``f`` lies outside the passband, so the
    entrance flux drops by the orders' share, ``1 - J0(a)^2`` for a ripple of
    ``a`` radians, and stays when ``f`` is inside.
    """
    from scipy.special import j0

    width = 2.0
    radians = 0.3
    base = _square_config(shack_hartmann={"spatial_filter_width_lambda_over_d": width})
    sensor = WavefrontSensor(base)
    xx, _ = _coordinates(base)
    pitch = EXTENT_M / LENSLETS
    for cycles_per_pitch, kept in ((0.5, 1.0), (1.5, j0(radians) ** 2)):
        opd = (
            radians * WAVELENGTH_M / (2 * np.pi) * np.sin(2 * np.pi * cycles_per_pitch * xx / pitch)
        )
        flux = sensor.subaperture_flux(opd)
        # Interior lenslets, away from the ringing at the grid edge.
        interior = flux.pupil_photons_s[2:-2, 2:-2] / (sensor.engine.source_rate / LENSLETS**2)
        np.testing.assert_allclose(interior, kept, rtol=2e-2)


# ------------------------------------------------------------------------ cross-talk


def test_crosstalk_is_the_overlap_add_of_wider_spot_windows() -> None:
    """Reach 1 equals adding independently propagated 3x-wide windows at pitch spacing.

    A sensor with three times the pixels per subaperture and no cross-talk
    propagates exactly the same window for every lenslet. Overlap-adding those
    windows at the true pitch must give the cross-talk image, and the total
    detected light must equal theirs while every spot stays on the detector.
    """
    pixels = 8
    crosstalk = _square_config(
        pixels=pixels, margin=pixels, shack_hartmann={"crosstalk_reach_subapertures": 1}
    )
    wide = _square_config(pixels=3 * pixels)
    opd = _opd(crosstalk.input.shape, rms_m=80e-9)
    image = WavefrontSensor(crosstalk).photon_rate(opd)
    windows = WavefrontSensor(wide).photon_rate(opd)
    blocks = windows.reshape(LENSLETS, 3 * pixels, LENSLETS, 3 * pixels).transpose(0, 2, 1, 3)
    expected = np.zeros(((LENSLETS + 2) * pixels,) * 2)
    for row in range(LENSLETS):
        for column in range(LENSLETS):
            expected[
                row * pixels : row * pixels + 3 * pixels,
                column * pixels : column * pixels + 3 * pixels,
            ] += blocks[row, column]
    np.testing.assert_allclose(image, expected, rtol=1e-12, atol=1e-12 * np.max(expected))
    np.testing.assert_allclose(np.sum(image), np.sum(windows), rtol=1e-12)


def test_crosstalk_light_beyond_the_detector_is_lost_and_reported() -> None:
    pixels = 8
    opd = _opd((LENSLETS * SAMPLES,) * 2, rms_m=80e-9)
    kept = WavefrontSensor(
        _square_config(
            pixels=pixels, margin=pixels, shack_hartmann={"crosstalk_reach_subapertures": 1}
        )
    )
    cut = WavefrontSensor(
        _square_config(pixels=pixels, shack_hartmann={"crosstalk_reach_subapertures": 1})
    )
    wide_margin = WavefrontSensor(
        _square_config(
            pixels=pixels, margin=3 * pixels, shack_hartmann={"crosstalk_reach_subapertures": 1}
        )
    )
    full = kept.photon_rate(opd)
    cropped = cut.photon_rate(opd)
    # Without a margin the detector keeps exactly the central part.
    np.testing.assert_allclose(cropped, full[pixels:-pixels, pixels:-pixels], rtol=1e-12)
    np.testing.assert_allclose(
        wide_margin.photon_rate(opd)[2 * pixels : -2 * pixels, 2 * pixels : -2 * pixels],
        full,
        rtol=1e-12,
    )
    frame = cut.expose(opd, seed=1)
    assert frame.metadata["wfs_captured_photons_s"] == pytest.approx(np.sum(cropped), rel=1e-12)
    assert np.sum(cropped) < np.sum(full)
    assert cut.engine.output_shape == (LENSLETS * pixels,) * 2


def _single_lenslet(tmp_path: Path, row: int, column: int, *, reach: int) -> WavefrontSensor:
    mask = np.zeros((LENSLETS * SAMPLES,) * 2)
    mask[row * SAMPLES : (row + 1) * SAMPLES, column * SAMPLES : (column + 1) * SAMPLES] = 1.0
    path = tmp_path / f"lenslet_{row}_{column}.npy"
    np.save(path, mask)
    return WavefrontSensor(
        _square_config(
            pixels=8,
            shack_hartmann={
                "crosstalk_reach_subapertures": reach,
                "minimum_illuminated_fraction": 0.0,
            },
            telescope={"custom_mask_path": str(path)},
        )
    )


@pytest.mark.parametrize("axis", ["x", "y"])
def test_tilted_spot_moves_its_light_into_the_expected_neighbour(tmp_path: Path, axis: str) -> None:
    """A tilt of one window (4 lambda/d) along +x or +y moves the spot one block along it.

    With cross-talk the neighbour's block then holds the spot's core: for a
    square lenslet the fraction inside a 4 lambda/d window is
    ``(2/pi Si(4 pi))^2 = 0.90`` (sampled lenslet slightly more). Without it
    the light leaves the lit lenslet's window and is lost.
    """
    from scipy.special import sici

    row, column = 3, 4
    with_crosstalk = _single_lenslet(tmp_path, row, column, reach=1)
    field_stop = _single_lenslet(tmp_path, row, column, reach=0)
    config = with_crosstalk.config
    xx, yy = _coordinates(config)
    pitch = EXTENT_M / LENSLETS
    window_lambda_over_d = 4.0  # 8 pixels at 2 pixels per lambda/d
    slope = window_lambda_over_d * WAVELENGTH_M / pitch
    opd = slope * (xx if axis == "x" else yy)
    neighbour = (row, column + 1) if axis == "x" else (row + 1, column)
    flux = with_crosstalk.subaperture_flux(opd)
    stopped = field_stop.subaperture_flux(opd)
    rate = with_crosstalk.engine.source_rate
    core = (2.0 / np.pi * sici(2.0 * np.pi * window_lambda_over_d / 2.0)[0]) ** 2
    assert flux.detector_photons_s[neighbour] / rate == pytest.approx(core, abs=0.03)
    assert flux.detector_photons_s[neighbour] > 10 * flux.detector_photons_s[row, column]
    assert np.argmax(flux.detector_photons_s) == np.ravel_multi_index(neighbour, (LENSLETS,) * 2)
    assert stopped.detector_photons_s[neighbour] == 0.0
    assert np.sum(stopped.detector_photons_s) < 0.2 * np.sum(flux.detector_photons_s)
    # The pupil-plane flux does not depend on where the spot lands.
    np.testing.assert_allclose(flux.pupil_photons_s, stopped.pupil_photons_s, rtol=1e-12)
    assert flux.pupil_photons_s[row, column] == pytest.approx(rate, rel=1e-12)
