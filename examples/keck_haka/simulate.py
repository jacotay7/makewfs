"""Animate open-loop Keck II HAKA natural-guide-star detector images.

The atmosphere comes from pyturb's traceable ``mauna-kea`` profile, makewfs
forms the 57x57 Shack-Hartmann mosaic, and getframes applies the OCAM2K EMCCD
detector model. Camera gain and exposure are selected from ``camera_modes.csv``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

import makewfs

HERE = Path(__file__).resolve().parent
KECK_DIAMETER_M = 10.95
# Fit to the amplifier-corrected live pupil. The shadow is the union of a
# circular secondary and a slightly larger pointy-top hexagonal component.
KECK_SECONDARY_CIRCLE_RADIUS_M = 1.2873048283643613
KECK_SECONDARY_HEX_CIRCUMRADIUS_M = 1.4550129501344706
KECK_SECONDARY_OFFSET_X_M = 0.03132835365923644
KECK_SECONDARY_OFFSET_Y_M = -0.032594058104417935
KECK_CENTRAL_OBSCURATION_DIAMETER_M = 2.0 * KECK_SECONDARY_CIRCLE_RADIUS_M
KECK_SPIDER_WIDTH_M = 0.026
HAKA_ILLUMINATED_LENSLETS_ACROSS = 54.0
HAKA_GRID_EXTENT_M = KECK_DIAMETER_M * 57.0 / HAKA_ILLUMINATED_LENSLETS_ACROSS
KECK_OCAM_AMPLIFIER_LAYOUT = (4, 2)
# The RTC ROI is [top:top+height, left:left+width] on the 240x240 detector.
KECK_OCAM_ROI_LEFT_PX = 4
KECK_OCAM_ROI_TOP_PX = 4
KECK_OCAM_ROI_WIDTH_PX = 228
KECK_OCAM_ROI_HEIGHT_PX = 228
KECK_OCAM_FULL_AMPLIFIER_BOUNDARIES_Y_PX = (60, 120, 180)
KECK_OCAM_FULL_AMPLIFIER_BOUNDARIES_X_PX = (120,)
KECK_OCAM_AMPLIFIER_BOUNDARIES_Y_PX = tuple(
    boundary - KECK_OCAM_ROI_TOP_PX for boundary in KECK_OCAM_FULL_AMPLIFIER_BOUNDARIES_Y_PX
)
KECK_OCAM_AMPLIFIER_BOUNDARIES_X_PX = tuple(
    boundary - KECK_OCAM_ROI_LEFT_PX for boundary in KECK_OCAM_FULL_AMPLIFIER_BOUNDARIES_X_PX
)
# Raw modes measured outside the generated pupil are [404, 407], [409, 408],
# [413, 408], [405, 404] ADU. These are offsets around the 408 ADU pedestal.
KECK_OCAM_AMPLIFIER_OFFSETS_ADU = (-4.0, -1.0, 1.0, 0.0, 5.0, 0.0, -3.0, -4.0)
# Relative e-/ADU factors inferred from estimated-dark-subtracted flux in fully
# illuminated, non-seam subapertures in the supplied eng519 V=10.16 cube. The
# corresponding ADU responses have arithmetic mean one, so this is not a global
# flux adjustment.
KECK_OCAM_AMPLIFIER_GAIN_FACTORS = (
    1.3737406261492462,
    1.2011866097543242,
    0.9705027257405701,
    0.9024763049728306,
    0.6262655226757171,
    1.6490741185051536,
    0.6628838205328217,
    1.6967367262098727,
)
OCAM2K_MAX_EM_GAIN = 600.0
OCAM2K_MAX_FRAME_RATE_HZ = 2067.0
HAKA_BAND_MIN_NM = 400.0
HAKA_BAND_MAX_NM = 950.0
HAKA_SPECTRAL_QUADRATURE_ORDER = 8
HAKA_SOURCE_TEMPERATURE_K = 6600.0
HAKA_REFERENCE_AIRMASS = 1.01
# Independently confirmed downstream HAKA bench throughput, after the telescope
# mirrors and before the OCAM2K detector QE. This is physical input, not a fit
# performed by the simulation or real-data comparison.
HAKA_DOWNSTREAM_THROUGHPUT = 0.287
KECK_ALUMINUM_MIRROR_REFLECTIVITY = 0.88
KECK_TELESCOPE_MIRROR_COUNT = 3
MAUNA_KEA_EXTINCTION_PATH = HERE / "mauna_kea_extinction.csv"
_H_PLANCK_J_S = 6.62607015e-34
_C_LIGHT_M_S = 299792458.0
_K_BOLTZMANN_J_K = 1.380649e-23


@dataclass(frozen=True)
class CameraMode:
    """One row of the supplied WATAO camera-mode table."""

    watao: int
    min_magnitude: float
    max_magnitude: float
    em_gain: float
    frame_rate_1_hz: float
    frame_rate_2_hz: float
    filter_name: str
    background_mode: int

    def frame_rate_hz(self, column: str) -> float:
        """Return the selected WFS frame-rate column."""
        if column == "WSFRRT1":
            return self.frame_rate_1_hz
        if column == "WSFRRT2":
            return self.frame_rate_2_hz
        raise ValueError("frame-rate column must be WSFRRT1 or WSFRRT2")


@dataclass(frozen=True)
class RenderedFrame:
    """Host-side data and labels for one animation frame."""

    magnitude: float
    mode: CameraMode
    time_s: float
    opd_m: NDArray[np.float64]
    counts: NDArray[np.float64]
    peak_counts: float
    launched_photons_per_s: float
    captured_photons_per_s: float
    expected_photoelectrons: float
    expected_unclipped_signal_counts: float
    measured_dark_subtracted_signal_counts: float
    image_full_well_e: float
    output_full_well_e: float | None
    gain_e_per_adu: float


@dataclass(frozen=True)
class BroadbandBudget:
    """Absolute photon budget and quadrature for one catalog magnitude."""

    wavelengths_m: tuple[float, ...]
    wavelength_weights: tuple[float, ...]
    collecting_area_m2: float
    above_atmosphere_photons_per_s_m2: float
    after_atmosphere_photons_per_s_m2: float
    after_telescope_mirrors_photons_per_s_m2: float
    detector_surface_photons_per_s: float
    photon_weighted_atmospheric_transmission: float
    telescope_mirror_throughput: float


def load_camera_modes(path: Path = HERE / "camera_modes.csv") -> tuple[CameraMode, ...]:
    """Load and validate the supplied WATAO magnitude lookup."""
    modes: list[CameraMode] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            mode = CameraMode(
                watao=int(row["watao"]),
                min_magnitude=float(row["minmag"]),
                max_magnitude=float(row["maxmag"]),
                em_gain=float(row["O1SMGN"]),
                frame_rate_1_hz=float(row["WSFRRT1"]),
                frame_rate_2_hz=float(row["WSFRRT2"]),
                filter_name=row["OBWNNAME"].strip(),
                background_mode=int(row["bkgnd"]),
            )
            if mode.min_magnitude >= mode.max_magnitude:
                raise ValueError(f"WATAO {mode.watao}: minmag must be below maxmag")
            if not 1.0 <= mode.em_gain <= OCAM2K_MAX_EM_GAIN:
                raise ValueError(f"WATAO {mode.watao}: EM gain is outside [1, 600]")
            if not (
                0.0 < mode.frame_rate_1_hz <= OCAM2K_MAX_FRAME_RATE_HZ
                and 0.0 < mode.frame_rate_2_hz <= OCAM2K_MAX_FRAME_RATE_HZ
            ):
                raise ValueError(f"WATAO {mode.watao}: frame rate exceeds 2067 Hz")
            modes.append(mode)
    if len({mode.watao for mode in modes}) != len(modes):
        raise ValueError("camera mode table contains duplicate WATAO indices")
    return tuple(modes)


def select_camera_mode(magnitude: float, modes: tuple[CameraMode, ...]) -> CameraMode:
    """Select a row using lower-inclusive, upper-exclusive magnitude bins."""
    matches = [mode for mode in modes if mode.min_magnitude <= magnitude < mode.max_magnitude]
    if len(matches) != 1:
        raise ValueError(f"magnitude {magnitude:g} matches {len(matches)} camera modes")
    return matches[0]


def _keck_segment_centres(radius_m: float) -> tuple[tuple[float, float], ...]:
    """Return the three axial rings of the 36-segment Keck primary."""
    centres: list[tuple[float, float]] = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            if max(abs(q), abs(r), abs(q + r)) <= 3 and (q != 0 or r != 0):
                centres.append(
                    (
                        math.sqrt(3.0) * radius_m * (q + r / 2.0),
                        1.5 * radius_m * r,
                    )
                )
    if len(centres) != 36:
        raise AssertionError("Keck pupil construction must contain 36 segments")
    return tuple(centres)


def _inside_hexagon(
    x_m: NDArray[np.float64], y_m: NDArray[np.float64], radius_m: float
) -> NDArray[np.bool_]:
    """Return membership in a pointy-top regular hexagon centred at zero."""
    abs_x = np.abs(x_m)
    abs_y = np.abs(y_m)
    return (abs_x <= math.sqrt(3.0) * radius_m / 2.0) & (
        abs_x + math.sqrt(3.0) * abs_y <= math.sqrt(3.0) * radius_m
    )


def make_keck_pupil(
    shape: tuple[int, int] = (228, 228),
    *,
    diameter_m: float = KECK_DIAMETER_M,
    segment_gap_m: float = 0.003,
    central_obscuration_diameter_m: float = KECK_CENTRAL_OBSCURATION_DIAMETER_M,
    central_hex_circumradius_m: float = KECK_SECONDARY_HEX_CIRCUMRADIUS_M,
    central_obscuration_offset_m: tuple[float, float] = (
        KECK_SECONDARY_OFFSET_X_M,
        KECK_SECONDARY_OFFSET_Y_M,
    ),
    spider_width_m: float = KECK_SPIDER_WIDTH_M,
    grid_extent_m: float = HAKA_GRID_EXTENT_M,
    supersampling: int = 4,
) -> NDArray[np.float32]:
    """Sample a 36-segment Keck pupil whose widest axis spans ``diameter_m``.

    The segment scale follows the official 10.95 m maximum primary diameter.
    The physical 3 mm inter-segment gaps and six 26 mm radial secondary-support
    arms begin from HCIPy's Keck aperture. The central shadow is the union of a
    circle and a pointy-top hexagon fitted to the supplied live HAKA pupil.
    """
    if shape[0] <= 0 or shape[1] <= 0:
        raise ValueError("pupil shape entries must be positive")
    if (
        diameter_m <= 0
        or segment_gap_m < 0
        or not 0 <= central_obscuration_diameter_m < diameter_m
        or not 0 <= central_hex_circumradius_m < diameter_m / 2.0
        or len(central_obscuration_offset_m) != 2
        or not all(math.isfinite(value) for value in central_obscuration_offset_m)
        or spider_width_m < 0
        or grid_extent_m < diameter_m
        or supersampling < 1
    ):
        raise ValueError("invalid Keck pupil diameter, gap, obscuration, spider, or sampling")
    # Three rings plus one segment vertex span 7*sqrt(3) circumradii.
    tiled_radius = diameter_m / (7.0 * math.sqrt(3.0))
    clear_radius = tiled_radius - segment_gap_m / math.sqrt(3.0)
    if clear_radius <= 0:
        raise ValueError("segment gap leaves no clear segment area")
    centres = _keck_segment_centres(tiled_radius)
    height, width = shape
    mask = np.zeros(shape, dtype=np.float64)
    for sub_y in range(supersampling):
        y = (
            (np.arange(height, dtype=np.float64) + (sub_y + 0.5) / supersampling) / height - 0.5
        ) * grid_extent_m
        for sub_x in range(supersampling):
            x = (
                (np.arange(width, dtype=np.float64) + (sub_x + 0.5) / supersampling) / width - 0.5
            ) * grid_extent_m
            xx, yy = np.meshgrid(x, y)
            illuminated = np.zeros(shape, dtype=np.bool_)
            for centre_x, centre_y in centres:
                illuminated |= _inside_hexagon(xx - centre_x, yy - centre_y, clear_radius)
            obstruction_x = xx - central_obscuration_offset_m[0]
            obstruction_y = yy - central_obscuration_offset_m[1]
            circular_shadow = (
                np.hypot(obstruction_x, obstruction_y) <= central_obscuration_diameter_m / 2.0
            )
            hexagonal_shadow = _inside_hexagon(
                obstruction_x,
                obstruction_y,
                central_hex_circumradius_m,
            )
            illuminated &= ~(circular_shadow | hexagonal_shadow)
            if spider_width_m > 0:
                # The HAKA RTC pupil shows the transposed sixfold support pattern:
                # vertical plus +/-30-degree arms rather than a horizontal arm.
                for angle_deg in range(30, 390, 60):
                    angle = math.radians(angle_deg)
                    along = xx * math.cos(angle) + yy * math.sin(angle)
                    across = -xx * math.sin(angle) + yy * math.cos(angle)
                    illuminated &= ~((along >= 0) & (np.abs(across) <= spider_width_m / 2.0))
            mask += illuminated
    mask /= float(supersampling * supersampling)
    return mask.astype(np.float32)


def pupil_collecting_area_m2(pupil: NDArray[np.float32]) -> float:
    """Return clear collecting area represented by a sampled pupil mask."""
    if pupil.ndim != 2 or pupil.shape[0] <= 0 or pupil.shape[1] <= 0:
        raise ValueError("pupil must be a non-empty two-dimensional array")
    pixel_area_m2 = (HAKA_GRID_EXTENT_M / pupil.shape[0]) * (HAKA_GRID_EXTENT_M / pupil.shape[1])
    return float(np.sum(pupil, dtype=np.float64) * pixel_area_m2)


def _blackbody_photon_shape(
    wavelength_nm: NDArray[np.float64], temperature_k: float
) -> NDArray[np.float64]:
    """Return a dimensionless Planck photon-density shape referenced at 550 nm."""
    if temperature_k <= 0:
        raise ValueError("source temperature must be positive")
    wavelength_m = wavelength_nm * 1e-9
    reference_m = 550e-9
    scale = _H_PLANCK_J_S * _C_LIGHT_M_S / (_K_BOLTZMANN_J_K * temperature_k)
    shape = wavelength_m**-4 / np.expm1(scale / wavelength_m)
    reference = reference_m**-4 / math.expm1(scale / reference_m)
    return np.asarray(shape / reference, dtype=np.float64)


def _gauss_legendre_interval(
    minimum: float, maximum: float, order: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    nodes, weights = np.polynomial.legendre.leggauss(order)
    half_width = 0.5 * (maximum - minimum)
    return half_width * nodes + 0.5 * (minimum + maximum), half_width * weights


def broadband_budget(
    magnitude: float,
    collecting_area_m2: float,
    *,
    normalization_band: str = "V",
    temperature_k: float = HAKA_SOURCE_TEMPERATURE_K,
    airmass: float = HAKA_REFERENCE_AIRMASS,
    mirror_reflectivity: float = KECK_ALUMINUM_MIRROR_REFLECTIVITY,
    mirror_count: int = KECK_TELESCOPE_MIRROR_COUNT,
    throughput: float = HAKA_DOWNSTREAM_THROUGHPUT,
) -> BroadbandBudget:
    """Build a Johnson-band-normalized 400--950 nm stellar photon quadrature.

    The catalog magnitude in ``normalization_band`` fixes the above-atmosphere
    normalization. The source is a blackbody photon spectrum and the tabulated
    Mauna Kea extinction in mag/airmass is applied at the observed frame's airmass. The primary,
    secondary, and tertiary are assigned the same band-averaged aluminum
    reflectivity. ``throughput`` is the independently measured remaining
    downstream HAKA bench term.
    """
    try:
        import getframes
    except ImportError as exc:  # pragma: no cover - required makewfs dependency
        raise ImportError("makewfs requires getframes") from exc
    if (
        collecting_area_m2 <= 0
        or airmass <= 0
        or not 0 <= mirror_reflectivity <= 1
        or mirror_count < 0
        or not 0 <= throughput <= 1
    ):
        raise ValueError(
            "collecting area and airmass must be positive; reflectivity and "
            "throughput in [0, 1]; mirror count non-negative"
        )

    normalized_band = normalization_band.upper()
    try:
        bandpass = getframes.Bandpass.johnson(normalized_band)
    except (KeyError, ValueError) as exc:
        raise ValueError(f"unsupported Johnson normalization band {normalization_band!r}") from exc
    if bandpass.response is None:
        raise AssertionError(f"getframes Johnson {normalized_band} must carry a spectral response")
    response = bandpass.response.response
    band_nodes, band_quadrature = _gauss_legendre_interval(
        float(response.wavelength_nm[0]),
        float(response.wavelength_nm[-1]),
        32,
    )
    band_response = np.interp(band_nodes, response.wavelength_nm, response.value)
    band_shape_integral = float(
        np.sum(band_quadrature * _blackbody_photon_shape(band_nodes, temperature_k) * band_response)
    )
    density_scale = bandpass.photon_flux(magnitude) / band_shape_integral

    wavelengths_nm, quadrature = _gauss_legendre_interval(
        HAKA_BAND_MIN_NM,
        HAKA_BAND_MAX_NM,
        HAKA_SPECTRAL_QUADRATURE_ORDER,
    )
    extinction = np.loadtxt(MAUNA_KEA_EXTINCTION_PATH, delimiter=",", comments="#")
    extinction_mag_per_airmass = np.interp(
        wavelengths_nm,
        extinction[:, 0],
        extinction[:, 1],
    )
    atmospheric_transmission = 10.0 ** (-0.4 * airmass * extinction_mag_per_airmass)
    above_density = density_scale * _blackbody_photon_shape(wavelengths_nm, temperature_k)
    above_contributions = quadrature * above_density
    transmitted_contributions = above_contributions * atmospheric_transmission
    above_flux = float(np.sum(above_contributions))
    transmitted_flux = float(np.sum(transmitted_contributions))
    mirror_throughput = mirror_reflectivity**mirror_count
    after_mirrors_flux = transmitted_flux * mirror_throughput
    return BroadbandBudget(
        wavelengths_m=tuple((wavelengths_nm * 1e-9).tolist()),
        wavelength_weights=tuple(transmitted_contributions.tolist()),
        collecting_area_m2=collecting_area_m2,
        above_atmosphere_photons_per_s_m2=above_flux,
        after_atmosphere_photons_per_s_m2=transmitted_flux,
        after_telescope_mirrors_photons_per_s_m2=after_mirrors_flux,
        detector_surface_photons_per_s=(after_mirrors_flux * collecting_area_m2 * throughput),
        photon_weighted_atmospheric_transmission=transmitted_flux / above_flux,
        telescope_mirror_throughput=mirror_throughput,
    )


def configured_sensor(
    base: makewfs.Config,
    *,
    magnitude: float,
    mode: CameraMode,
    frame_rate_column: str,
    pupil_path: Path,
    collecting_area_m2: float,
    source_temperature_k: float = HAKA_SOURCE_TEMPERATURE_K,
    airmass: float = HAKA_REFERENCE_AIRMASS,
    magnitude_band: str = "V",
) -> makewfs.WavefrontSensor:
    """Build one magnitude/camera-mode configuration from public sibling APIs."""
    try:
        import getframes
    except ImportError as exc:  # pragma: no cover - dependency is required by makewfs
        raise ImportError("makewfs requires getframes") from exc
    frame_rate_hz = mode.frame_rate_hz(frame_rate_column)
    if not math.isclose(
        base.source.throughput,
        HAKA_DOWNSTREAM_THROUGHPUT,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "Keck HAKA config must use the independently measured downstream "
            f"throughput {HAKA_DOWNSTREAM_THROUGHPUT:g}"
        )
    camera = getframes.load_preset("andor_ocam2k").replace(
        em_gain=mode.em_gain,
        bias_offset_adu=408.0,
        amplifier_layout=KECK_OCAM_AMPLIFIER_LAYOUT,
        amplifier_gain_factors=KECK_OCAM_AMPLIFIER_GAIN_FACTORS,
        amplifier_offsets_adu=KECK_OCAM_AMPLIFIER_OFFSETS_ADU,
    )
    detector = replace(
        base.detector,
        preset=None,
        inline=camera.to_dict(),
        exposure_s=1.0 / frame_rate_hz,
    )
    budget = broadband_budget(
        magnitude,
        collecting_area_m2,
        normalization_band=magnitude_band,
        temperature_k=source_temperature_k,
        airmass=airmass,
        throughput=base.source.throughput,
    )
    metadata = {
        **base.metadata,
        "watao": mode.watao,
        "em_gain": mode.em_gain,
        "frame_rate_hz": frame_rate_hz,
        "frame_rate_column": frame_rate_column,
        "obwnname": mode.filter_name,
        "bkgnd": mode.background_mode,
        f"catalog_{magnitude_band.lower()}_magnitude": magnitude,
        "magnitude_normalization_band": magnitude_band.upper(),
        "source_temperature_k": source_temperature_k,
        "airmass": airmass,
        "bandpass_nm": [HAKA_BAND_MIN_NM, HAKA_BAND_MAX_NM],
        "spectral_quadrature_order": HAKA_SPECTRAL_QUADRATURE_ORDER,
        "clear_collecting_area_m2": collecting_area_m2,
        "above_atmosphere_photons_per_s_m2": budget.above_atmosphere_photons_per_s_m2,
        "after_atmosphere_photons_per_s_m2": budget.after_atmosphere_photons_per_s_m2,
        "after_telescope_mirrors_photons_per_s_m2": (
            budget.after_telescope_mirrors_photons_per_s_m2
        ),
        "photon_weighted_atmospheric_transmission": (
            budget.photon_weighted_atmospheric_transmission
        ),
        "aluminum_mirror_reflectivity_each": KECK_ALUMINUM_MIRROR_REFLECTIVITY,
        "telescope_mirror_count": KECK_TELESCOPE_MIRROR_COUNT,
        "telescope_mirror_throughput": budget.telescope_mirror_throughput,
        "downstream_haka_throughput": base.source.throughput,
    }
    config = replace(
        base,
        telescope=replace(base.telescope, custom_mask_path=str(pupil_path)),
        source=replace(
            base.source,
            normalization="detector_photon_rate",
            detector_photon_rate_per_s=budget.detector_surface_photons_per_s,
            magnitude=None,
            band=None,
            wavelengths_m=budget.wavelengths_m,
            wavelength_weights=budget.wavelength_weights,
            sed_path=None,
            transmission_path=None,
        ),
        detector=detector,
        metadata=metadata,
    )
    return makewfs.WavefrontSensor(config)


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(HERE / "keck_haka.gif"))
    parser.add_argument("--manifest", help="JSON manifest path (default: output with .json)")
    parser.add_argument(
        "--magnitudes",
        nargs="+",
        type=float,
        default=[value / 2.0 for value in range(10, 31)],
        help="Vega V magnitudes to animate (default: 5 to 15 in 0.5-mag steps)",
    )
    parser.add_argument("--frames-per-magnitude", type=int, default=1)
    parser.add_argument("--samples-per-exposure", type=int, default=3)
    parser.add_argument(
        "--atmosphere-step-s",
        type=float,
        default=1.0 / 30.0,
        help=(
            "minimum atmosphere time between displayed frames; prevents bright-star "
            "sub-millisecond exposures from looking frozen (default: 1/30 s)"
        ),
    )
    parser.add_argument(
        "--master-dark-frames",
        type=int,
        default=32,
        help="exposure-matched dark frames median-combined for each WATAO mode",
    )
    parser.add_argument("--seeing", type=float, default=0.65, help="500 nm seeing (arcsec)")
    parser.add_argument("--profile", choices=["mauna-kea", "keck"], default="mauna-kea")
    parser.add_argument("--source-temperature-k", type=float, default=HAKA_SOURCE_TEMPERATURE_K)
    parser.add_argument("--airmass", type=float, default=HAKA_REFERENCE_AIRMASS)
    parser.add_argument("--atmosphere-engine", choices=["spectral", "extrude"], default="extrude")
    parser.add_argument("--seed", type=int, default=57)
    parser.add_argument("--playback-fps", type=int, default=6)
    parser.add_argument("--frame-rate-column", choices=["WSFRRT1", "WSFRRT2"], default="WSFRRT1")
    return parser.parse_args()


def _validate_arguments(args: argparse.Namespace, modes: tuple[CameraMode, ...]) -> None:
    if (
        args.frames_per_magnitude < 1
        or args.samples_per_exposure < 1
        or args.master_dark_frames < 1
    ):
        raise SystemExit("frame and exposure sample counts must be >= 1")
    if (
        args.playback_fps < 1
        or args.seeing <= 0
        or args.atmosphere_step_s <= 0
        or args.source_temperature_k <= 0
        or args.airmass <= 0
    ):
        raise SystemExit(
            "--playback-fps, --seeing, --atmosphere-step-s, source temperature, "
            "and airmass must be positive"
        )
    for magnitude in args.magnitudes:
        select_camera_mode(magnitude, modes)


def _atmosphere_display_gap_s(exposure_s: float, atmosphere_step_s: float) -> float:
    """Return extra frozen-flow evolution after a physically integrated exposure."""
    return max(0.0, atmosphere_step_s - exposure_s)


def _render_frames(
    args: argparse.Namespace,
    modes: tuple[CameraMode, ...],
    pupil: NDArray[np.float32],
    pupil_path: Path,
) -> list[RenderedFrame]:
    try:
        import getframes
        import pyturb
    except ImportError as exc:
        raise SystemExit("install makewfs[examples,interop] to run this example") from exc
    base = makewfs.load_config(HERE / "keck_haka.toml")
    collecting_area_m2 = pupil_collecting_area_m2(pupil)
    atmosphere = pyturb.Atmosphere.from_profile(
        args.profile,
        seeing=args.seeing,
        diameter=base.input.grid_extent_m,
        n=base.input.shape[0],
        seed=args.seed,
        engine=args.atmosphere_engine,
    )
    rendered: list[RenderedFrame] = []
    master_darks: dict[int, Any] = {}
    time_s = 0.0
    frame_index = 0
    for magnitude in args.magnitudes:
        mode = select_camera_mode(magnitude, modes)
        sensor = configured_sensor(
            base,
            magnitude=magnitude,
            mode=mode,
            frame_rate_column=args.frame_rate_column,
            pupil_path=pupil_path,
            collecting_area_m2=collecting_area_m2,
            source_temperature_k=args.source_temperature_k,
            airmass=args.airmass,
        )
        exposure_s = sensor.config.detector.exposure_s
        if mode.watao not in master_darks:
            master_darks[mode.watao] = sensor.detector.camera.master_dark(
                exposure_s,
                args.master_dark_frames,
                sensor.config.detector.temperature_c,
                seed=args.seed * 100_000 + mode.watao,
                method="median",
            )
        for _ in range(args.frames_per_magnitude):
            samples: list[NDArray[np.float64]] = []
            for _ in range(args.samples_per_exposure):
                opd = atmosphere.evolve(exposure_s / args.samples_per_exposure)
                samples.append(np.asarray(pyturb.to_numpy(opd), dtype=np.float64))
            detector_frame = sensor.expose_integrated(
                samples, seed=args.seed * 10_000 + frame_index
            )
            calibrated = getframes.calibrate(detector_frame, dark=master_darks[mode.watao])
            counts = np.asarray(getframes.to_numpy(calibrated.data), dtype=np.float64)
            if detector_frame.truth is None:
                raise AssertionError("Keck HAKA flux audit requires detector truth")
            expected_photoelectrons = float(
                np.sum(getframes.to_numpy(detector_frame.truth.mean_photoelectrons))
            )
            camera_config = sensor.detector.camera.config
            time_s += exposure_s
            rendered.append(
                RenderedFrame(
                    magnitude=magnitude,
                    mode=mode,
                    time_s=time_s,
                    opd_m=samples[-1],
                    counts=counts,
                    peak_counts=float(np.max(counts)),
                    launched_photons_per_s=float(detector_frame.metadata["wfs_launched_photons_s"]),
                    captured_photons_per_s=float(detector_frame.metadata["wfs_captured_photons_s"]),
                    expected_photoelectrons=expected_photoelectrons,
                    expected_unclipped_signal_counts=(
                        expected_photoelectrons * mode.em_gain / camera_config.gain_e_per_adu
                    ),
                    measured_dark_subtracted_signal_counts=float(np.sum(counts)),
                    image_full_well_e=camera_config.full_well_e,
                    output_full_well_e=camera_config.output_full_well_e,
                    gain_e_per_adu=camera_config.gain_e_per_adu,
                )
            )
            display_gap_s = _atmosphere_display_gap_s(exposure_s, args.atmosphere_step_s)
            if display_gap_s > 0:
                atmosphere.evolve(display_gap_s)
                time_s += display_gap_s
            frame_index += 1
        del sensor
    if pupil.shape != rendered[0].opd_m.shape:
        raise AssertionError("pupil and pyturb OPD grids must match")
    return rendered


def _save_animation(
    args: argparse.Namespace,
    rendered: list[RenderedFrame],
    pupil: NDArray[np.float32],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
    except ImportError as exc:
        raise SystemExit("install makewfs[examples,interop] to run this example") from exc
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    opd_stack = np.stack([frame.opd_m for frame in rendered]) * 1e9
    count_stack = np.stack([frame.counts for frame in rendered])
    illuminated = pupil > 0
    opd_scale = max(float(np.percentile(np.abs(opd_stack[:, illuminated]), 99.5)), 1.0)
    count_max = max(float(np.percentile(count_stack, 99.8)), 1.0)

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 5.4), constrained_layout=True)
    masked_opd = np.ma.masked_where(~illuminated, opd_stack[0])
    opd_artist = axes[0].imshow(
        masked_opd,
        origin="lower",
        cmap="coolwarm",
        vmin=-opd_scale,
        vmax=opd_scale,
        interpolation="nearest",
    )
    figure.colorbar(opd_artist, ax=axes[0], fraction=0.046, label="open-loop OPD (nm)")
    frame_artist = axes[1].imshow(
        count_stack[0],
        origin="lower",
        vmin=0.0,
        vmax=count_max,
        interpolation="nearest",
    )
    figure.colorbar(
        frame_artist,
        ax=axes[1],
        fraction=0.046,
        label="dark-subtracted OCAM2K signal (count / ADU)",
    )
    axes[0].set_title("Keck pupil: uncorrected Maunakea OPD")
    axes[1].set_title("HAKA 57x57 SH: 228x228-pixel ROI")
    for axis in axes:
        axis.set_xlabel("x (native pixel)")
        axis.set_ylabel("y (native pixel)")
    title = figure.suptitle("", fontsize=12)

    def update(index: int) -> tuple[Any, ...]:
        frame = rendered[index]
        rate = frame.mode.frame_rate_hz(args.frame_rate_column)
        opd_artist.set_data(np.ma.masked_where(~illuminated, frame.opd_m * 1e9))
        frame_artist.set_data(frame.counts)
        title.set_text(
            f"Keck II HAKA open loop | V={frame.magnitude:g} | 400-950 nm, "
            f"X={args.airmass:g} | "
            f"WATAO {frame.mode.watao} | EM x{frame.mode.em_gain:g} | {rate:g} Hz "
            f"({1e3 / rate:.2f} ms) | t={1e3 * frame.time_s:.1f} ms | "
            f"peak={frame.peak_counts:.0f} count"
        )
        return opd_artist, frame_artist, title

    animation = FuncAnimation(figure, update, frames=len(rendered), blit=False)
    suffix = output.suffix.lower()
    if suffix == ".gif":
        writer: Any = PillowWriter(fps=args.playback_fps)
    elif suffix == ".mp4":
        if not FFMpegWriter.isAvailable():
            raise SystemExit("FFmpeg is unavailable; choose a .gif output or install ffmpeg")
        writer = FFMpegWriter(fps=args.playback_fps, bitrate=2400)
    else:
        raise SystemExit("--output must end in .gif or .mp4")
    animation.save(output, writer=writer, dpi=110)
    plt.close(figure)


def _write_manifest(
    args: argparse.Namespace,
    modes: tuple[CameraMode, ...],
    rendered: list[RenderedFrame],
    pupil: NDArray[np.float32],
) -> Path:
    output = Path(args.output).resolve()
    manifest = Path(args.manifest).resolve() if args.manifest else output.with_suffix(".json")
    used_modes = {frame.mode.watao: frame.mode for frame in rendered}
    collecting_area_m2 = pupil_collecting_area_m2(pupil)
    spectral_budget = broadband_budget(
        0.0,
        collecting_area_m2,
        temperature_k=args.source_temperature_k,
        airmass=args.airmass,
    )
    spectral_weights = np.asarray(spectral_budget.wavelength_weights, dtype=np.float64)
    spectral_weights /= np.sum(spectral_weights)
    payload = {
        "output": output.name,
        "seed": args.seed,
        "atmosphere_profile": args.profile,
        "atmosphere_engine": args.atmosphere_engine,
        "seeing_arcsec_at_500_nm": args.seeing,
        "magnitudes_vega_v": args.magnitudes,
        "frames_per_magnitude": args.frames_per_magnitude,
        "samples_per_exposure": args.samples_per_exposure,
        "minimum_atmosphere_step_between_displayed_frames_s": args.atmosphere_step_s,
        "source_spectrum": {
            "model": "Planck photon spectrum normalized to catalog Johnson V magnitude",
            "temperature_k": args.source_temperature_k,
            "bandpass_nm": [HAKA_BAND_MIN_NM, HAKA_BAND_MAX_NM],
            "quadrature_order": HAKA_SPECTRAL_QUADRATURE_ORDER,
            "wavelengths_nm": [
                wavelength_m * 1e9 for wavelength_m in spectral_budget.wavelengths_m
            ],
            "normalized_post_atmosphere_photon_weights": spectral_weights.tolist(),
        },
        "atmospheric_extinction": {
            "site": "Mauna Kea",
            "airmass": args.airmass,
            "curve": MAUNA_KEA_EXTINCTION_PATH.name,
            "curve_sha256": hashlib.sha256(MAUNA_KEA_EXTINCTION_PATH.read_bytes()).hexdigest(),
            "units": "magnitude per airmass",
            "photon_weighted_transmission": (
                spectral_budget.photon_weighted_atmospheric_transmission
            ),
            "red_edge_extrapolation": (
                "Gemini table ends at 900 nm; 0.05 mag/airmass is held constant to 950 nm"
            ),
            "reference": "Gemini Observatory Mauna Kea optical extinction curve",
        },
        "telescope_mirrors": {
            "coating_assumption": "bare aluminum, scalar band-averaged reflectivity",
            "count": KECK_TELESCOPE_MIRROR_COUNT,
            "surfaces": ["primary", "secondary", "tertiary"],
            "reflectivity_each": KECK_ALUMINUM_MIRROR_REFLECTIVITY,
            "combined_throughput": spectral_budget.telescope_mirror_throughput,
            "combined_loss_fraction": 1.0 - spectral_budget.telescope_mirror_throughput,
            "downstream_haka_throughput": HAKA_DOWNSTREAM_THROUGHPUT,
            "downstream_haka_throughput_basis": ("team-confirmed independent bench measurements"),
        },
        "clear_collecting_area_m2": collecting_area_m2,
        "master_dark_frames_per_mode": args.master_dark_frames,
        "master_dark_combine": "median",
        "frame_rate_column": args.frame_rate_column,
        "playback_fps_not_physical": args.playback_fps,
        "pupil_sha256": hashlib.sha256(pupil.tobytes()).hexdigest(),
        "pupil_shape": list(pupil.shape),
        "detector_roi_shape": [228, 228],
        "detector_roi_full_sensor_xywh_px": [
            KECK_OCAM_ROI_LEFT_PX,
            KECK_OCAM_ROI_TOP_PX,
            KECK_OCAM_ROI_WIDTH_PX,
            KECK_OCAM_ROI_HEIGHT_PX,
        ],
        "amplifier_model": {
            "layout_rows_columns": list(KECK_OCAM_AMPLIFIER_LAYOUT),
            "roi_boundaries_y_px": list(KECK_OCAM_AMPLIFIER_BOUNDARIES_Y_PX),
            "roi_boundaries_x_px": list(KECK_OCAM_AMPLIFIER_BOUNDARIES_X_PX),
            "offsets_adu_row_major": list(KECK_OCAM_AMPLIFIER_OFFSETS_ADU),
            "relative_e_per_adu_row_major": list(KECK_OCAM_AMPLIFIER_GAIN_FACTORS),
            "response_normalization": (
                "reciprocal ADU/e- responses have arithmetic mean one; no global flux scaling"
            ),
        },
        "fallback_reference_wavelength_nm": 673.0,
        "dark_subtracted": True,
        "detector_saturation": {
            "image_area_full_well_e": rendered[0].image_full_well_e,
            "output_full_well_e": rendered[0].output_full_well_e,
            "gain_e_per_adu": rendered[0].gain_e_per_adu,
            "dark_subtracted_output_ceiling_counts": (
                None
                if rendered[0].output_full_well_e is None
                else rendered[0].output_full_well_e / rendered[0].gain_e_per_adu
            ),
        },
        "frame_flux_audit": [
            {
                "magnitude": frame.magnitude,
                "watao": frame.mode.watao,
                "atmosphere_time_s": frame.time_s,
                "launched_photons_per_s": frame.launched_photons_per_s,
                "captured_photons_per_s": frame.captured_photons_per_s,
                "optical_capture_fraction": (
                    frame.captured_photons_per_s / frame.launched_photons_per_s
                ),
                "incident_photons_per_exposure": (
                    frame.captured_photons_per_s / frame.mode.frame_rate_hz(args.frame_rate_column)
                ),
                "expected_photoelectrons": frame.expected_photoelectrons,
                "expected_unclipped_signal_counts": frame.expected_unclipped_signal_counts,
                "measured_dark_subtracted_signal_counts": (
                    frame.measured_dark_subtracted_signal_counts
                ),
                "peak_dark_subtracted_counts": frame.peak_counts,
            }
            for frame in rendered
        ],
        "camera_modes": [asdict(used_modes[index]) for index in sorted(used_modes)],
        "assumptions": [
            (
                "Guide-star magnitudes are catalog Vega V magnitudes; all showcase "
                "stars use the eng519-like 6600 K spectral shape."
            ),
            (
                "Mauna Kea atmospheric extinction is modeled across 400--950 nm at "
                f"airmass {args.airmass:g}. Three aluminum telescope reflections use "
                f"R={KECK_ALUMINUM_MIRROR_REFLECTIVITY:g} each; downstream HAKA "
                f"throughput is {HAKA_DOWNSTREAM_THROUGHPUT:g} from independent "
                "bench measurements."
            ),
            "Exposure equals the inverse selected frame rate (100% duty cycle).",
            (
                "The pupil has 36 segments, 3 mm gaps, a live-data-fitted circular-plus-"
                "hexagonal central shadow, and six 26 mm support arms and spans about "
                "54 of 57 lenslets; relay rotation is omitted."
            ),
            (
                "The centred RTC crop preserves eight OCAM outputs in a 4x2 layout; "
                "relative gain and pedestal are measured from the supplied cube."
            ),
            (
                "Each mode uses an exposure-matched "
                f"{args.master_dark_frames}-frame median master dark."
            ),
            "Other OCAM2K preset parameters are held fixed across modes.",
            "The supplied bkgnd flag is recorded but not interpreted as a photon background.",
        ],
        "available_mode_count": len(modes),
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    """Run the deterministic parameter-sweep animation."""
    args = _parse_arguments()
    modes = load_camera_modes()
    _validate_arguments(args, modes)
    pupil = make_keck_pupil()
    with tempfile.TemporaryDirectory(prefix="makewfs-keck-haka-") as temporary:
        pupil_path = Path(temporary) / "keck_primary.npy"
        np.save(pupil_path, pupil)
        rendered = _render_frames(args, modes, pupil, pupil_path)
    _save_animation(args, rendered, pupil)
    manifest = _write_manifest(args, modes, rendered, pupil)
    print(f"wrote {args.output} ({len(rendered)} frames) and {manifest}")


if __name__ == "__main__":
    main()
