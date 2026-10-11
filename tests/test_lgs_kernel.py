"""Fast sodium-LGS kernel mode versus the range/angle quadrature (issue #19)."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from makewfs import ConfigError, WavefrontSensor, load_config
from makewfs.config import WFSConfig
from makewfs.sampling import _SpotPropagationPlan
from makewfs.source import iter_source_states, lgs_kernel_morphology

ROOT = Path(__file__).parents[1]
SH_CONFIG = ROOT / "examples" / "configs" / "shack_hartmann_minimal.toml"
LENSLETS = 8
PIXELS = 12
RANGES_M = tuple(float(value) for value in np.linspace(86e3, 94e3, 17))
RANGE_WEIGHTS = tuple(
    float(value) for value in np.exp(-0.5 * ((np.asarray(RANGES_M) - 90e3) / 2e3) ** 2)
)


def _config(
    *,
    dtype: str = "float64",
    rotation_deg: float = 0.0,
    pupil_samples: int | None = None,
    oversampling: int = 2,
    **source: object,
) -> WFSConfig:
    config = load_config(SH_CONFIG)
    assert config.shack_hartmann is not None
    return replace(
        config,
        numerics=replace(
            config.numerics,
            dtype=dtype,
            pupil_samples_per_lenslet=pupil_samples,
            fft_oversampling=oversampling,
        ),
        shack_hartmann=replace(
            config.shack_hartmann,
            pixels_per_subaperture=PIXELS,
            lenslet_grid_rotation_deg=rotation_deg,
        ),
        source=replace(config.source, kind="lgs", **source),  # type: ignore[arg-type]
    )


def _pair(**source: object) -> tuple[WavefrontSensor, WavefrontSensor]:
    rotation = float(source.pop("rotation_deg", 0.0))  # type: ignore[arg-type]
    samples = source.pop("pupil_samples", None)
    quadrature = WavefrontSensor(
        _config(rotation_deg=rotation, pupil_samples=samples, **source)  # type: ignore[arg-type]
    )
    kernel = WavefrontSensor(
        _config(
            rotation_deg=rotation,
            pupil_samples=samples,  # type: ignore[arg-type]
            lgs_model="kernel",
            **source,
        )
    )
    return quadrature, kernel


def _spots(image: np.ndarray) -> np.ndarray:
    return image.reshape(LENSLETS, PIXELS, LENSLETS, PIXELS).transpose(0, 2, 1, 3)


def _centroids(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validation-only centroids in pixels about each window centre."""
    spots = _spots(image)
    coordinate = np.arange(PIXELS) - (PIXELS - 1) / 2
    flux = spots.sum(axis=(2, 3))
    safe = np.where(flux > 0, flux, 1.0)
    x = (spots * coordinate[None, None, None, :]).sum(axis=(2, 3)) / safe
    y = (spots * coordinate[None, None, :, None]).sum(axis=(2, 3)) / safe
    return x, y, flux


def _total_spot_variance(image: np.ndarray, valid: np.ndarray) -> float:
    """Sum over valid subapertures of each spot's second central moment, pixels^2."""
    spots = _spots(image)[valid]
    coordinate = np.arange(PIXELS) - (PIXELS - 1) / 2
    yy, xx = np.meshgrid(coordinate, coordinate, indexing="ij")
    weights = spots / spots.sum(axis=(1, 2), keepdims=True)
    mean_x = (weights * xx).sum(axis=(1, 2), keepdims=True)
    mean_y = (weights * yy).sum(axis=(1, 2), keepdims=True)
    return float((weights * ((xx - mean_x) ** 2 + (yy - mean_y) ** 2)).sum())


def _gaussian_table(path: Path, fwhm_arcsec: float, *, step_sigma: float = 0.5) -> None:
    """A dense Gaussian source table: the quadrature reference for a continuous extent."""
    sigma = fwhm_arcsec / 2.3548200450309493
    nodes = np.arange(-4.0, 4.0 + 1e-9, step_sigma) * sigma
    x, y = np.meshgrid(nodes, nodes)
    weight = np.exp(-0.5 * (x**2 + y**2) / sigma**2)
    np.savetxt(path, np.c_[x.ravel(), y.ravel(), weight.ravel()])


def test_lgs_model_is_validated_and_serialized() -> None:
    data = _config(lgs_ranges_m=(89e3, 91e3)).to_dict()
    assert "lgs_model" not in data["source"]
    data["source"]["lgs_model"] = "KERNEL"
    config = WFSConfig.from_dict(data)
    assert config.source.lgs_model == "kernel"
    assert WFSConfig.from_dict(config.to_dict()).digest == config.digest
    data["source"]["lgs_model"] = "quadrature"
    assert WFSConfig.from_dict(data).digest != config.digest
    data["source"]["lgs_model"] = "fft"
    with pytest.raises(ConfigError, match=r"source\.lgs_model: expected"):
        WFSConfig.from_dict(data)
    data["source"].update(
        kind="ngs", lgs_model="kernel", lgs_ranges_m=[], lgs_launch_position_m=[0.0, 0.0]
    )
    with pytest.raises(ConfigError, match=r"requires source\.kind = 'lgs'"):
        WFSConfig.from_dict(data)
    pyramid = load_config(ROOT / "examples" / "configs" / "pyramid_minimal.toml").to_dict()
    pyramid["source"].update(kind="lgs", lgs_model="kernel")
    with pytest.raises(ConfigError, match="shack_hartmann' only"):
        WFSConfig.from_dict(pyramid)


def test_kernel_mode_propagates_only_wavelength_states(tmp_path: Path) -> None:
    table = tmp_path / "source.txt"
    np.savetxt(table, [[-0.1, 0.0, 1.0], [0.1, 0.0, 3.0]])
    config = _config(
        lgs_ranges_m=(89e3, 90e3, 91e3),
        lgs_range_weights=(1.0, 2.0, 1.0),
        wavelengths_m=(600e-9, 700e-9),
        angular_kernel_path=str(table),
        field_angle_arcsec=(0.2, -0.1),
        lgs_model="kernel",
    )
    states = iter_source_states(config)
    assert [state.wavelength_m for state in states] == [600e-9, 700e-9]
    assert all(state.range_m is None for state in states)
    assert all(math.isclose(state.angle_x_rad, math.radians(0.2 / 3600)) for state in states)
    morphology = lgs_kernel_morphology(config)
    assert morphology.sigma_rad == 0.0
    assert [round(weight, 12) for *_, weight in morphology.points] == [0.25, 0.75]
    assert math.isclose(morphology.points[0][0], -math.radians(0.1 / 3600))
    assert morphology.range_weights == (0.25, 0.5, 0.25)
    thin = lgs_kernel_morphology(_config(angular_fwhm_arcsec=0.5, lgs_model="kernel"))
    assert thin.points == ((0.0, 0.0, 1.0),)
    assert thin.ranges_m == ()
    assert math.isclose(thin.sigma_rad, math.radians(0.5 / 3600) / 2.3548200450309493)


@pytest.mark.parametrize(
    ("launch_m", "rotation_deg"),
    [((0.0, 0.0), 0.0), ((4.0, 1.0), 0.0), ((-3.0, -2.0), 0.0), ((4.0, 1.0), 10.0)],
)
def test_thick_profile_matches_the_range_quadrature(
    launch_m: tuple[float, float], rotation_deg: float
) -> None:
    """Without source extent, both models translate the same spot by the same offsets.

    The kernel deposits each translation by linear interpolation on the
    ``fft_oversampling = 2`` focal grid, a small extra smoothing of a sharp
    spot, so images agree to a few percent while centroids and fluxes, which
    linear deposition conserves, agree much more tightly.
    """
    quadrature, kernel = _pair(
        lgs_ranges_m=RANGES_M,
        lgs_range_weights=RANGE_WEIGHTS,
        lgs_launch_position_m=launch_m,
        rotation_deg=rotation_deg,
    )
    opd = np.zeros(quadrature.config.input.shape)
    reference = np.asarray(quadrature.photon_rate(opd))
    fast = np.asarray(kernel.photon_rate(opd))
    assert np.all(fast >= 0.0)
    assert np.abs(fast - reference).sum() / reference.sum() < 0.04
    valid = np.asarray(quadrature.valid_subapertures())
    reference_x, reference_y, reference_flux = _centroids(reference)
    fast_x, fast_y, fast_flux = _centroids(fast)
    assert np.max(np.abs(fast_x - reference_x)[valid]) < 0.01
    assert np.max(np.abs(fast_y - reference_y)[valid]) < 0.01
    assert np.max(np.abs(fast_flux - reference_flux)[valid]) / reference_flux.max() < 2e-3
    # The comparison is not trivially passed by unelongated spots: summed over
    # the valid subapertures, both images are far wider than a thin beacon's.
    thin = np.asarray(WavefrontSensor(_config(rotation_deg=rotation_deg)).photon_rate(opd))
    assert _total_spot_variance(reference, valid) > 1.5 * _total_spot_variance(thin, valid)
    assert _total_spot_variance(fast, valid) > 1.5 * _total_spot_variance(thin, valid)


def test_linear_deposition_converges_with_focal_oversampling() -> None:
    """The kernel's only approximation for point translations is second order."""
    source = {
        "lgs_ranges_m": RANGES_M,
        "lgs_range_weights": RANGE_WEIGHTS,
        "lgs_launch_position_m": (4.0, 1.0),
    }
    differences = []
    for oversampling in (2, 4):
        reference = np.asarray(
            WavefrontSensor(_config(oversampling=oversampling, **source)).photon_rate(
                np.zeros((128, 128))
            )
        )
        fast = np.asarray(
            WavefrontSensor(
                _config(oversampling=oversampling, lgs_model="kernel", **source)
            ).photon_rate(np.zeros((128, 128)))
        )
        differences.append(np.abs(fast - reference).sum() / reference.sum())
    # Halving the focal sample should cut the deposition error about four times.
    assert differences[1] < 0.3 * differences[0]


def test_source_extent_matches_a_dense_gaussian_quadrature(tmp_path: Path) -> None:
    table = tmp_path / "gaussian.txt"
    _gaussian_table(table, 0.25)
    reference = np.asarray(
        WavefrontSensor(_config(angular_kernel_path=str(table))).photon_rate(np.zeros((128, 128)))
    )
    fast = np.asarray(
        WavefrontSensor(_config(angular_fwhm_arcsec=0.25, lgs_model="kernel")).photon_rate(
            np.zeros((128, 128))
        )
    )
    assert np.abs(fast - reference).sum() / reference.sum() < 1e-4


@pytest.mark.validation
def test_thick_profile_with_extent_matches_dense_quadrature(tmp_path: Path) -> None:
    table = tmp_path / "gaussian.txt"
    _gaussian_table(table, 0.25)
    ranges = RANGES_M[::2]
    weights = RANGE_WEIGHTS[::2]
    reference = np.asarray(
        WavefrontSensor(
            _config(
                angular_kernel_path=str(table),
                lgs_ranges_m=ranges,
                lgs_range_weights=weights,
                lgs_launch_position_m=(4.0, 1.0),
            )
        ).photon_rate(np.zeros((128, 128)))
    )
    fast = np.asarray(
        WavefrontSensor(
            _config(
                angular_fwhm_arcsec=0.25,
                lgs_ranges_m=ranges,
                lgs_range_weights=weights,
                lgs_launch_position_m=(4.0, 1.0),
                lgs_model="kernel",
            )
        ).photon_rate(np.zeros((128, 128)))
    )
    assert np.abs(fast - reference).sum() / reference.sum() < 0.01
    reference_x, reference_y, _ = _centroids(reference)
    fast_x, fast_y, _ = _centroids(fast)
    assert np.max(np.abs(fast_x - reference_x)) < 5e-3
    assert np.max(np.abs(fast_y - reference_y)) < 5e-3


def test_thin_beacon_without_extent_is_the_point_source_image() -> None:
    """A unit-impulse kernel must leave the diffraction spots untouched."""
    opd = np.random.default_rng(5).normal(size=(128, 128)) * 30e-9
    point = WavefrontSensor(_config()).photon_rate(opd)
    kernel = WavefrontSensor(_config(lgs_model="kernel")).photon_rate(opd)
    np.testing.assert_allclose(kernel, point, rtol=0.0, atol=1e-12 * float(np.max(point)))


def test_window_truncation_is_reported_not_renormalized() -> None:
    """Light elongated out of a subaperture window is lost in both models alike.

    The streaks reach about 18 pixels, beyond the 12-pixel window. With 32
    pupil samples per lenslet the sampled field's far field repeats every 64
    pixels, so neither the quadrature tilts alias nor does the kernel margin
    (up to 26 pixels) cut off light that elongation moves into the window.
    """
    long_ranges = tuple(float(value) for value in np.linspace(85e3, 95e3, 21))
    quadrature, kernel = _pair(
        lgs_ranges_m=long_ranges, lgs_launch_position_m=(6.0, 0.0), pupil_samples=32
    )
    opd = np.zeros((128, 128))
    reference = quadrature.expose(opd, seed=0)
    fast = kernel.expose(opd, seed=0)
    launched = fast.metadata["wfs_launched_photons_s"]
    captured = fast.metadata["wfs_captured_photons_s"]
    thin_captured = (
        WavefrontSensor(_config(pupil_samples=32))
        .expose(opd, seed=0)
        .metadata["wfs_captured_photons_s"]
    )
    assert captured < 0.9 * thin_captured < launched
    assert math.isclose(captured, reference.metadata["wfs_captured_photons_s"], rel_tol=2e-3)
    assert math.isclose(float(np.sum(fast.truth.photon_rate)), captured, rel_tol=1e-9)
    assert fast.metadata["wfs_lgs_model"] == "kernel"
    assert reference.metadata["wfs_lgs_model"] == "quadrature"
    assert fast.metadata["wfs_lgs_kernel_margin_px"] == fast.metadata["wfs_lgs_kernel_radius_px"]
    assert fast.metadata["wfs_lgs_kernel_radius_px"] > 15.0
    # Every kernel holds unit weight before truncation by the window.
    spectrum = np.asarray(kernel.engine.geometry(0).lgs_kernel.kernel_spectrum)
    np.testing.assert_allclose(spectrum[:, 0, 0].real, 1.0, rtol=1e-12)


def test_centre_launch_image_has_the_pupil_mirror_symmetry() -> None:
    _, kernel = _pair(
        lgs_ranges_m=RANGES_M, lgs_range_weights=RANGE_WEIGHTS, angular_fwhm_arcsec=0.2
    )
    image = np.asarray(kernel.photon_rate(np.zeros((128, 128))))
    scale = float(image.max())
    np.testing.assert_allclose(image, image[:, ::-1], atol=1e-9 * scale)
    np.testing.assert_allclose(image, image[::-1, :], atol=1e-9 * scale)
    np.testing.assert_allclose(image, image.T, atol=1e-9 * scale)


def test_elongation_is_radial_from_the_launch() -> None:
    _, kernel = _pair(lgs_ranges_m=RANGES_M, lgs_range_weights=RANGE_WEIGHTS)
    spots = _spots(np.asarray(kernel.photon_rate(np.zeros((128, 128)))))
    coordinate = np.arange(PIXELS) - (PIXELS - 1) / 2
    yy, xx = np.meshgrid(coordinate, coordinate, indexing="ij")
    for row, column in ((6, 6), (1, 4), (3, 0)):
        spot = spots[row, column] / spots[row, column].sum()
        mean_x = float(np.sum(spot * xx))
        mean_y = float(np.sum(spot * yy))
        cxx = float(np.sum(spot * (xx - mean_x) ** 2))
        cyy = float(np.sum(spot * (yy - mean_y) ** 2))
        cxy = float(np.sum(spot * (xx - mean_x) * (yy - mean_y)))
        axis = 0.5 * math.atan2(2 * cxy, cxx - cyy)
        radial = math.atan2(row - 3.5, column - 3.5)
        difference = (axis - radial + math.pi / 2) % math.pi - math.pi / 2
        assert abs(difference) < math.radians(3)


def test_float32_kernel_matches_float64() -> None:
    source = {
        "lgs_ranges_m": RANGES_M,
        "lgs_range_weights": RANGE_WEIGHTS,
        "lgs_launch_position_m": (2.0, -1.0),
        "angular_fwhm_arcsec": 0.2,
        "lgs_model": "kernel",
    }
    single = WavefrontSensor(_config(dtype="float32", **source))
    double = WavefrontSensor(_config(**source))
    assert single.engine.geometry(0).lgs_kernel.kernel_spectrum.dtype == np.complex64
    opd = np.random.default_rng(7).normal(size=(128, 128)) * 30e-9
    reference = np.asarray(double.photon_rate(opd))
    np.testing.assert_allclose(
        single.photon_rate(opd), reference, rtol=0.0, atol=2e-5 * float(reference.max())
    )


def test_kernel_mode_matches_quadrature_with_field_angle_and_two_wavelengths() -> None:
    quadrature, kernel = _pair(
        lgs_ranges_m=RANGES_M[::4],
        lgs_range_weights=RANGE_WEIGHTS[::4],
        lgs_launch_position_m=(3.0, 3.0),
        field_angle_arcsec=(0.1, -0.05),
        wavelengths_m=(650e-9, 750e-9),
        wavelength_weights=(1.0, 2.0),
    )
    opd = np.random.default_rng(11).normal(size=(128, 128)) * 30e-9
    reference = quadrature.engine.render(opd)
    fast = kernel.engine.render(opd)
    assert fast.spectral_photon_rate is not None and reference.spectral_photon_rate is not None
    for index in range(2):
        expected = np.asarray(reference.spectral_photon_rate[index])
        actual = np.asarray(fast.spectral_photon_rate[index])
        assert np.abs(actual - expected).sum() / expected.sum() < 0.05
    reference_x, reference_y, _ = _centroids(np.asarray(reference.photon_rate))
    fast_x, fast_y, _ = _centroids(np.asarray(fast.photon_rate))
    valid = np.asarray(quadrature.valid_subapertures())
    # Five sodium nodes form a coarse comb; where the window clips it, linear
    # deposition moves a little light across the window edge (the first moment
    # is conserved only for unclipped light), so allow 0.06 focal samples.
    assert np.max(np.abs(fast_x - reference_x)[valid]) < 0.03
    assert np.max(np.abs(fast_y - reference_y)[valid]) < 0.03


def test_offset_states_each_get_their_own_kernel() -> None:
    config = _config(
        lgs_ranges_m=RANGES_M,
        lgs_range_weights=RANGE_WEIGHTS,
        lgs_launch_position_m=(4.0, 0.0),
        lgs_model="kernel",
    )
    assert config.shack_hartmann is not None
    states = replace(
        config,
        shack_hartmann=replace(
            config.shack_hartmann, lenslet_grid_offset_states_fraction=((0.0, 0.0), (0.5, 0.0))
        ),
    )
    shifted = replace(
        config,
        shack_hartmann=replace(config.shack_hartmann, lenslet_grid_offset_fraction=(0.5, 0.0)),
    )
    opd = np.random.default_rng(13).normal(size=(128, 128)) * 30e-9
    multi = WavefrontSensor(states)
    np.testing.assert_allclose(
        multi.photon_rate(opd, offset_state=1),
        WavefrontSensor(shifted).photon_rate(opd),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        multi.photon_rate(opd, offset_state=0), WavefrontSensor(config).photon_rate(opd), rtol=1e-12
    )


def test_integrated_render_applies_the_kernel_to_each_sample() -> None:
    _, kernel = _pair(lgs_ranges_m=RANGES_M, lgs_launch_position_m=(1.0, 2.0))
    rng = np.random.default_rng(17)
    samples = [rng.normal(size=(128, 128)) * 30e-9 for _ in range(3)]
    batched = kernel.engine.render_integrated(samples).photon_rate
    separate = np.mean([kernel.engine.render(sample).photon_rate for sample in samples], axis=0)
    np.testing.assert_allclose(batched, separate, rtol=1e-10, atol=1e-6)


def test_spot_window_margin_must_fit_one_period() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        _SpotPropagationPlan.build(
            pixels=8,
            samples_per_lenslet=16,
            sampling=2.0,
            oversampling=2,
            field_dtype=np.complex128,
            field_stop_radius_lambda_over_d=None,
            optical_blur_kernel=None,
            charge_diffusion_kernel=None,
            backend=WavefrontSensor(_config()).backend,
            window_margin=-1,
        )
    with pytest.raises(ValueError, match="beyond one FFT period"):
        _SpotPropagationPlan.build(
            pixels=8,
            samples_per_lenslet=16,
            sampling=2.0,
            oversampling=2,
            field_dtype=np.complex128,
            field_stop_radius_lambda_over_d=None,
            optical_blur_kernel=None,
            charge_diffusion_kernel=None,
            backend=WavefrontSensor(_config()).backend,
            window_margin=40,
        )
