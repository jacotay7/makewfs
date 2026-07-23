"""Quantify and optimize the empirical Keck HAKA OCAM2K camera-mode LUT.

The metric is detector-limited intensity SNR for the dark-subtracted sum of all
16 pixels in each active 4x4 lenslet region. Photon shot noise, EMCCD excess
noise, dark current, CIC, output-amplifier read noise, amplifier conversion
differences, and ADC quantization are included. The reported value is the
arithmetic mean over active lenslets and generated open-loop atmosphere states.
It is not a centroid, slope, or reconstructed-wavefront-error metric.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from simulate import (
    HAKA_REFERENCE_AIRMASS,
    OCAM2K_MAX_EM_GAIN,
    OCAM2K_MAX_FRAME_RATE_HZ,
    CameraMode,
    configured_sensor,
    load_camera_modes,
    make_keck_pupil,
    pupil_collecting_area_m2,
    select_camera_mode,
)

import makewfs

HERE = Path(__file__).resolve().parent
REFERENCE_R_MAGNITUDE = 10.0
SIGMA_SATURATION_MARGIN = 5.0
HAKA_OPTIMIZATION_MAX_FRAME_RATE_HZ = OCAM2K_MAX_FRAME_RATE_HZ
EMPIRICAL_TAIL_MIN_MAGNITUDE_R = 10.0


@dataclass(frozen=True)
class GuideStarSED:
    """One representative main-sequence continuum approximation."""

    label: str
    spectral_type: str
    temperature_k: float
    color: str


REPRESENTATIVE_SEDS = (
    GuideStarSED("A0 V", "blue", 9500.0, "#3977b9"),
    GuideStarSED("F6 V", "eng519-like", 6600.0, "#35a16b"),
    GuideStarSED("G2 V", "solar", 5800.0, "#d18f00"),
    GuideStarSED("M3 V", "red", 3400.0, "#b44747"),
)


@dataclass(frozen=True)
class RateEnsemble:
    """Reference photoelectron-rate maps for one R-normalized SED."""

    sed: GuideStarSED
    photoelectron_rate_per_s: NDArray[np.float64]


@dataclass(frozen=True)
class SNRStats:
    """Distribution summary across active lenslets and atmosphere states."""

    mean: float
    median: float
    percentile_10: float
    percentile_90: float


@dataclass(frozen=True)
class EmpiricalFrameRateFit:
    """Tail-derived smooth broken power law with the detector ceiling."""

    maximum_frame_rate_hz: float
    transition_relative_flux: float
    flux_exponent: float
    transition_sharpness: float
    rms_log10_residual: float
    anchor_magnitudes_r: tuple[float, ...]
    anchor_frame_rates_hz: tuple[float, ...]

    def relative_flux(self, magnitude_r: float | NDArray[np.float64]) -> Any:
        """Return R-band flux relative to a magnitude-10 source."""
        return 10.0 ** (-0.4 * (np.asarray(magnitude_r) - REFERENCE_R_MAGNITUDE))

    def frame_rate_hz(self, magnitude_r: float | NDArray[np.float64]) -> Any:
        """Evaluate the fitted cadence in Hz."""
        flux = self.relative_flux(magnitude_r)
        result = self.maximum_frame_rate_hz * (
            1.0 + (self.transition_relative_flux / flux) ** self.transition_sharpness
        ) ** (-self.flux_exponent / self.transition_sharpness)
        return float(result) if np.ndim(result) == 0 else result


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(HERE / "haka_lut_snr.png"))
    parser.add_argument("--manifest", default=str(HERE / "haka_lut_snr.json"))
    parser.add_argument(
        "--candidate-lut", default=str(HERE / "camera_modes_empirical_floor_continuous.csv")
    )
    parser.add_argument("--states", type=int, default=4)
    parser.add_argument("--seeing", type=float, default=0.65)
    parser.add_argument("--airmass", type=float, default=HAKA_REFERENCE_AIRMASS)
    parser.add_argument("--atmosphere-step-s", type=float, default=1.0 / 30.0)
    parser.add_argument("--seed", type=int, default=57)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--target-snr", type=float, default=4.5)
    parser.add_argument("--minimum-frame-rate-hz", type=float, default=1.0)
    parser.add_argument("--minimum-magnitude-r", type=float, default=5.0)
    parser.add_argument("--maximum-magnitude-r", type=float, default=15.0)
    parser.add_argument("--magnitude-step", type=float, default=0.05)
    parser.add_argument("--frame-rate-column", choices=["WSFRRT1", "WSFRRT2"], default="WSFRRT1")
    return parser.parse_args()


def _validate_arguments(args: argparse.Namespace) -> None:
    if args.states < 1:
        raise SystemExit("--states must be at least one")
    if (
        args.seeing <= 0
        or args.airmass <= 0
        or args.atmosphere_step_s <= 0
        or args.target_snr <= 0
        or args.minimum_frame_rate_hz < 1
        or args.minimum_frame_rate_hz > HAKA_OPTIMIZATION_MAX_FRAME_RATE_HZ
        or args.minimum_magnitude_r >= args.maximum_magnitude_r
        or args.magnitude_step <= 0
    ):
        raise SystemExit("seeing, airmass, SNR, rate, magnitude range, and step are invalid")


def _amplifier_maps(camera: Any) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return public-config conversion and offset maps in detector (y, x) order."""
    height, width = camera.resolution
    y_edges = (0, *camera.amplifier_boundaries_y_px, height)
    x_edges = (0, *camera.amplifier_boundaries_x_px, width)
    rows = len(y_edges) - 1
    columns = len(x_edges) - 1
    factors = np.asarray(camera.amplifier_gain_factors, dtype=np.float64).reshape(rows, columns)
    offsets = np.asarray(camera.amplifier_offsets_adu, dtype=np.float64).reshape(rows, columns)
    gain_map = np.empty((height, width), dtype=np.float64)
    offset_map = np.empty((height, width), dtype=np.float64)
    for row in range(rows):
        for column in range(columns):
            block = np.s_[y_edges[row] : y_edges[row + 1], x_edges[column] : x_edges[column + 1]]
            gain_map[block] = camera.gain_e_per_adu * factors[row, column]
            offset_map[block] = offsets[row, column]
    return gain_map, offset_map


def _lenslet_sum(array: NDArray[np.float64]) -> NDArray[np.float64]:
    """Sum 4x4 detector regions into a 57x57 lenslet array."""
    if array.shape != (228, 228):
        raise ValueError("HAKA detector arrays must have shape (228, 228)")
    return array.reshape(57, 4, 57, 4).sum(axis=(1, 3))


def lenslet_snr(
    photoelectron_rate_per_s: NDArray[np.float64],
    *,
    magnitude_r: float,
    frame_rate_hz: float,
    em_gain: float,
    reference_magnitude_r: float,
    camera: Any,
    active_lenslets: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Return total-intensity SNR samples for active HAKA lenslets.

    The input can contain one or more atmosphere states on axis zero. A perfect
    exposure-matched dark mean is subtracted; dark and CIC shot variance remain.
    """
    rates = np.asarray(photoelectron_rate_per_s, dtype=np.float64)
    if rates.ndim == 2:
        rates = rates[np.newaxis, ...]
    if rates.ndim != 3 or rates.shape[1:] != (228, 228):
        raise ValueError("photoelectron rate must have shape (state, 228, 228)")
    if frame_rate_hz <= 0 or em_gain < 1 or active_lenslets.shape != (57, 57):
        raise ValueError("invalid frame rate, EM gain, or active-lenslet mask")

    gain_map, _ = _amplifier_maps(camera)
    exposure_s = 1.0 / frame_rate_hz
    magnitude_scale = 10.0 ** (-0.4 * (magnitude_r - reference_magnitude_r))
    photoelectrons = rates * (magnitude_scale * exposure_s)
    dark_electrons = camera.dark_current_at(-45.0) * exposure_s
    cic_electrons = camera.clock_induced_charge_e
    excess_noise_factor = camera.gain_excess_noise_factor if em_gain > 1 else 1.0

    signal_adu = photoelectrons * em_gain / gain_map
    variance_adu2 = (
        em_gain**2
        * excess_noise_factor**2
        * (photoelectrons + dark_electrons + cic_electrons)
        / gain_map**2
        + camera.read_noise_e**2 / gain_map**2
        + 1.0 / 12.0
    )
    samples: list[NDArray[np.float64]] = []
    for state_signal, state_variance in zip(signal_adu, variance_adu2, strict=True):
        summed_signal = _lenslet_sum(state_signal)
        summed_variance = _lenslet_sum(state_variance)
        samples.append(summed_signal[active_lenslets] / np.sqrt(summed_variance[active_lenslets]))
    return np.concatenate(samples)


def snr_stats(samples: NDArray[np.float64]) -> SNRStats:
    """Summarize a non-empty finite SNR sample array."""
    values = np.asarray(samples, dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("SNR samples must be non-empty and finite")
    return SNRStats(
        mean=float(np.mean(values)),
        median=float(np.median(values)),
        percentile_10=float(np.percentile(values, 10.0)),
        percentile_90=float(np.percentile(values, 90.0)),
    )


def fit_empirical_frame_rate_model(
    modes: tuple[CameraMode, ...],
    *,
    frame_rate_column: str,
) -> EmpiricalFrameRateFit:
    """Fit faint-tail cadence with a ceiling-limited smooth broken power law.

    Only open-filter rows centred at R >= 10 and below the broad R=15.5--24
    catch-all row are fitted. This excludes the hand-rounded bright 2000 Hz
    ceiling and coarse 1500 Hz operational plateau. Bin centres are weighted by
    bin width. The model approaches the true 2067 Hz OCAM2K ceiling for abundant
    flux and a fitted power law in the faint limit; a third parameter controls
    the smoothness of that transition.
    """
    from scipy.optimize import least_squares

    fitted_modes = tuple(
        mode
        for mode in modes
        if mode.filter_name == "open"
        and 0.5 * (mode.min_magnitude + mode.max_magnitude) >= EMPIRICAL_TAIL_MIN_MAGNITUDE_R
        and mode.min_magnitude < 15.5
    )
    anchors = sorted(
        (
            (0.5 * (mode.min_magnitude + mode.max_magnitude), mode.frame_rate_hz(frame_rate_column))
            for mode in fitted_modes
        ),
        key=lambda item: item[0],
    )
    if len(anchors) < 2:
        raise ValueError("continuous empirical frame-rate fit requires at least two open rows")
    magnitudes = np.asarray([anchor[0] for anchor in anchors], dtype=np.float64)
    frame_rates = np.asarray([anchor[1] for anchor in anchors], dtype=np.float64)
    if np.any(np.diff(frame_rates) > 0):
        raise ValueError("empirical open-filter frame rates must not rise toward fainter stars")
    widths = np.asarray(
        [
            mode.max_magnitude - mode.min_magnitude
            for mode in sorted(
                fitted_modes,
                key=lambda item: 0.5 * (item.min_magnitude + item.max_magnitude),
            )
        ],
        dtype=np.float64,
    )
    relative_flux = 10.0 ** (-0.4 * (magnitudes - REFERENCE_R_MAGNITUDE))

    def prediction(parameters: NDArray[np.float64]) -> NDArray[np.float64]:
        transition_flux = 10.0 ** parameters[0]
        flux_exponent = parameters[1]
        transition_sharpness = parameters[2]
        return HAKA_OPTIMIZATION_MAX_FRAME_RATE_HZ * (
            1.0 + (transition_flux / relative_flux) ** transition_sharpness
        ) ** (-flux_exponent / transition_sharpness)

    result = least_squares(
        lambda parameters: (
            (np.log10(prediction(parameters)) - np.log10(frame_rates)) * np.sqrt(widths)
        ),
        np.asarray([-0.9, 1.3, 0.5], dtype=np.float64),
        bounds=(np.asarray([-5.0, 0.01, 0.02]), np.asarray([5.0, 5.0, 20.0])),
    )
    fitted = prediction(np.asarray(result.x, dtype=np.float64))
    rms = float(
        np.sqrt(
            np.average(
                (np.log10(fitted) - np.log10(frame_rates)) ** 2,
                weights=widths,
            )
        )
    )
    return EmpiricalFrameRateFit(
        maximum_frame_rate_hz=HAKA_OPTIMIZATION_MAX_FRAME_RATE_HZ,
        transition_relative_flux=10.0 ** float(result.x[0]),
        flux_exponent=float(result.x[1]),
        transition_sharpness=float(result.x[2]),
        rms_log10_residual=rms,
        anchor_magnitudes_r=tuple(magnitudes.tolist()),
        anchor_frame_rates_hz=tuple(frame_rates.tolist()),
    )


def evaluate_mode(
    ensemble: RateEnsemble,
    *,
    magnitude_r: float,
    frame_rate_hz: float,
    em_gain: float,
    camera: Any,
    active_lenslets: NDArray[np.bool_],
) -> SNRStats:
    """Evaluate one camera mode for one representative SED."""
    return snr_stats(
        lenslet_snr(
            ensemble.photoelectron_rate_per_s,
            magnitude_r=magnitude_r,
            frame_rate_hz=frame_rate_hz,
            em_gain=em_gain,
            reference_magnitude_r=REFERENCE_R_MAGNITUDE,
            camera=camera,
            active_lenslets=active_lenslets,
        )
    )


def saturation_fractions(
    ensembles: tuple[RateEnsemble, ...],
    *,
    magnitude_r: float,
    frame_rate_hz: float,
    em_gain: float,
    camera: Any,
) -> dict[str, float]:
    """Return worst five-sigma fractions of all physical/digital ceilings."""
    gain_map, offset_map = _amplifier_maps(camera)
    exposure_s = 1.0 / frame_rate_hz
    scale = 10.0 ** (-0.4 * (magnitude_r - REFERENCE_R_MAGNITUDE))
    excess_noise_factor = camera.gain_excess_noise_factor if em_gain > 1 else 1.0
    dark_and_cic = camera.dark_current_at(-45.0) * exposure_s + camera.clock_induced_charge_e
    image_fraction = 0.0
    output_fraction = 0.0
    adc_fraction = 0.0
    for ensemble in ensembles:
        mean_input = ensemble.photoelectron_rate_per_s * (scale * exposure_s) + dark_and_cic
        image_upper = mean_input + SIGMA_SATURATION_MARGIN * np.sqrt(mean_input)
        output_mean = em_gain * mean_input
        output_sigma = np.sqrt(
            em_gain**2 * excess_noise_factor**2 * mean_input + camera.read_noise_e**2
        )
        output_upper = output_mean + SIGMA_SATURATION_MARGIN * output_sigma
        image_fraction = max(image_fraction, float(np.max(image_upper / camera.full_well_e)))
        output_ceiling = (
            camera.full_well_e if camera.output_full_well_e is None else camera.output_full_well_e
        )
        output_fraction = max(output_fraction, float(np.max(output_upper / output_ceiling)))
        adc_ceiling_e = (camera.max_adu - camera.bias_offset_adu - offset_map) * gain_map
        adc_fraction = max(adc_fraction, float(np.max(output_upper / adc_ceiling_e)))
    return {
        "image_area": image_fraction,
        "output_register": output_fraction,
        "adc": adc_fraction,
        "maximum": max(image_fraction, output_fraction, adc_fraction),
    }


def optimize_at_magnitude(
    magnitude_r: float,
    *,
    target_snr: float,
    minimum_frame_rate_hz: float,
    ensembles: tuple[RateEnsemble, ...],
    camera: Any,
    active_lenslets: NDArray[np.bool_],
) -> dict[str, Any]:
    """Find the fastest continuous camera mode meeting an absolute SNR target.

    Every representative SED must meet ``target_snr``. Among modes at the
    fastest feasible frame rate, the lowest continuous EM gain is selected. A
    five-sigma realization must remain below the image, output-register, and ADC
    ceilings for every SED, atmosphere state, and pixel.
    """
    if target_snr <= 0 or minimum_frame_rate_hz < 1:
        raise ValueError("target SNR and minimum frame rate must be positive")

    def gain_meets_snr(frame_rate_hz: float, gain: float) -> bool:
        for ensemble in ensembles:
            achieved = evaluate_mode(
                ensemble,
                magnitude_r=magnitude_r,
                frame_rate_hz=frame_rate_hz,
                em_gain=gain,
                camera=camera,
                active_lenslets=active_lenslets,
            ).mean
            if achieved < target_snr * (1.0 - 1e-10):
                return False
        return True

    def minimum_feasible_gain(frame_rate_hz: float) -> tuple[float, dict[str, float]] | None:
        if gain_meets_snr(frame_rate_hz, 1.0):
            saturation = saturation_fractions(
                ensembles,
                magnitude_r=magnitude_r,
                frame_rate_hz=frame_rate_hz,
                em_gain=1.0,
                camera=camera,
            )
            if saturation["maximum"] < 1.0:
                return 1.0, saturation
        if not gain_meets_snr(frame_rate_hz, OCAM2K_MAX_EM_GAIN):
            return None
        low = np.nextafter(1.0, 2.0)
        high = OCAM2K_MAX_EM_GAIN
        for _ in range(36):
            middle = 0.5 * (low + high)
            if gain_meets_snr(frame_rate_hz, middle):
                high = middle
            else:
                low = middle
        saturation = saturation_fractions(
            ensembles,
            magnitude_r=magnitude_r,
            frame_rate_hz=frame_rate_hz,
            em_gain=high,
            camera=camera,
        )
        return None if saturation["maximum"] >= 1.0 else (high, saturation)

    if not gain_meets_snr(minimum_frame_rate_hz, OCAM2K_MAX_EM_GAIN):
        raise ValueError(
            f"SNR {target_snr:g} is unattainable at R={magnitude_r:g}, "
            f"{minimum_frame_rate_hz} Hz, and EM gain 600"
        )
    if gain_meets_snr(HAKA_OPTIMIZATION_MAX_FRAME_RATE_HZ, OCAM2K_MAX_EM_GAIN):
        proposed_rate = HAKA_OPTIMIZATION_MAX_FRAME_RATE_HZ
        best = minimum_feasible_gain(proposed_rate)
    else:
        low_rate = minimum_frame_rate_hz
        high_rate = HAKA_OPTIMIZATION_MAX_FRAME_RATE_HZ
        for _ in range(36):
            middle_rate = 0.5 * (low_rate + high_rate)
            if gain_meets_snr(middle_rate, OCAM2K_MAX_EM_GAIN):
                low_rate = middle_rate
            else:
                high_rate = middle_rate
        proposed_rate = low_rate
        best = minimum_feasible_gain(proposed_rate)
    if best is None:
        raise ValueError(
            f"no unsaturated gain meets SNR {target_snr:g} at R={magnitude_r:g} "
            f"and the SNR-limited {proposed_rate:g} Hz rate"
        )

    proposed_gain, proposed_saturation = best
    achieved = {
        ensemble.sed.label: evaluate_mode(
            ensemble,
            magnitude_r=magnitude_r,
            frame_rate_hz=proposed_rate,
            em_gain=proposed_gain,
            camera=camera,
            active_lenslets=active_lenslets,
        ).mean
        for ensemble in ensembles
    }
    return {
        "magnitude_r": magnitude_r,
        "target_snr": target_snr,
        "proposed_em_gain": proposed_gain,
        "proposed_frame_rate_hz": proposed_rate,
        "mean_snr_by_sed": achieved,
        "minimum_snr_across_seds": min(achieved.values()),
        "mean_snr_across_seds": float(np.mean(tuple(achieved.values()))),
        "maximum_snr_across_seds": max(achieved.values()),
        "five_sigma_saturation_fraction": proposed_saturation,
    }


def apply_empirical_frame_rate_floor(
    target_result: dict[str, Any],
    *,
    empirical_floor_hz: float,
    ensembles: tuple[RateEnsemble, ...],
    camera: Any,
    active_lenslets: NDArray[np.bool_],
) -> dict[str, Any]:
    """Keep the target-SNR solution only when it is above the empirical floor."""
    magnitude_r = float(target_result["magnitude_r"])
    target_snr = float(target_result["target_snr"])
    target_rate_hz = float(target_result["proposed_frame_rate_hz"])
    if target_rate_hz >= empirical_floor_hz:
        return {
            **target_result,
            "target_snr_frame_rate_hz": target_rate_hz,
            "empirical_frame_rate_floor_hz": empirical_floor_hz,
            "empirical_floor_active": False,
            "target_snr_met": True,
        }

    proposed_gain = OCAM2K_MAX_EM_GAIN
    saturation = saturation_fractions(
        ensembles,
        magnitude_r=magnitude_r,
        frame_rate_hz=empirical_floor_hz,
        em_gain=proposed_gain,
        camera=camera,
    )
    if saturation["maximum"] >= 1.0:
        low_gain = 1.0
        high_gain = OCAM2K_MAX_EM_GAIN
        for _ in range(36):
            middle_gain = 0.5 * (low_gain + high_gain)
            middle_saturation = saturation_fractions(
                ensembles,
                magnitude_r=magnitude_r,
                frame_rate_hz=empirical_floor_hz,
                em_gain=middle_gain,
                camera=camera,
            )
            if middle_saturation["maximum"] < 1.0:
                low_gain = middle_gain
            else:
                high_gain = middle_gain
        proposed_gain = low_gain
        saturation = saturation_fractions(
            ensembles,
            magnitude_r=magnitude_r,
            frame_rate_hz=empirical_floor_hz,
            em_gain=proposed_gain,
            camera=camera,
        )
    achieved = {
        ensemble.sed.label: evaluate_mode(
            ensemble,
            magnitude_r=magnitude_r,
            frame_rate_hz=empirical_floor_hz,
            em_gain=proposed_gain,
            camera=camera,
            active_lenslets=active_lenslets,
        ).mean
        for ensemble in ensembles
    }
    minimum_snr = min(achieved.values())
    return {
        **target_result,
        "proposed_em_gain": proposed_gain,
        "proposed_frame_rate_hz": empirical_floor_hz,
        "mean_snr_by_sed": achieved,
        "minimum_snr_across_seds": minimum_snr,
        "mean_snr_across_seds": float(np.mean(tuple(achieved.values()))),
        "maximum_snr_across_seds": max(achieved.values()),
        "five_sigma_saturation_fraction": saturation,
        "target_snr_frame_rate_hz": target_rate_hz,
        "empirical_frame_rate_floor_hz": empirical_floor_hz,
        "empirical_floor_active": True,
        "target_snr_met": minimum_snr >= target_snr * (1.0 - 1e-10),
    }


def _generate_opd_states(
    args: argparse.Namespace, base: makewfs.Config
) -> list[NDArray[np.float64]]:
    try:
        import pyturb
    except ImportError as exc:  # pragma: no cover - example dependency
        raise SystemExit("install makewfs[examples,interop] to run this analysis") from exc
    atmosphere = pyturb.Atmosphere.from_profile(
        "mauna-kea",
        seeing=args.seeing,
        diameter=base.input.grid_extent_m,
        n=base.input.shape[0],
        seed=args.seed,
        engine="extrude",
        dtype=base.numerics.dtype,
        device=args.device,
    )
    return [
        np.asarray(pyturb.to_numpy(atmosphere.evolve(args.atmosphere_step_s)), dtype=np.float64)
        for _ in range(args.states)
    ]


def _build_rate_ensembles(
    args: argparse.Namespace,
    *,
    base: makewfs.Config,
    pupil_path: Path,
    collecting_area_m2: float,
    opd_states: list[NDArray[np.float64]],
) -> tuple[tuple[RateEnsemble, ...], NDArray[np.bool_], Any]:
    try:
        import getframes
    except ImportError as exc:  # pragma: no cover - required dependency
        raise SystemExit("install makewfs[examples,interop] to run this analysis") from exc
    reference_mode = replace(
        select_camera_mode(REFERENCE_R_MAGNITUDE, load_camera_modes()),
        em_gain=600.0,
        frame_rate_1_hz=750.0,
        frame_rate_2_hz=750.0,
    )
    ensembles: list[RateEnsemble] = []
    active_lenslets: NDArray[np.bool_] | None = None
    camera_config: Any = None
    for sed_index, sed in enumerate(REPRESENTATIVE_SEDS):
        sensor = configured_sensor(
            base,
            magnitude=REFERENCE_R_MAGNITUDE,
            magnitude_band="R",
            mode=reference_mode,
            frame_rate_column="WSFRRT1",
            pupil_path=pupil_path,
            collecting_area_m2=collecting_area_m2,
            source_temperature_k=sed.temperature_k,
            airmass=args.airmass,
        )
        state_rates: list[NDArray[np.float64]] = []
        exposure_s = sensor.config.detector.exposure_s
        for state_index, opd_m in enumerate(opd_states):
            frame = sensor.expose(opd_m, seed=args.seed * 10_000 + sed_index * 100 + state_index)
            if frame.truth is None:
                raise AssertionError("HAKA LUT analysis requires detector truth")
            state_rates.append(
                np.asarray(getframes.to_numpy(frame.truth.mean_photoelectrons), dtype=np.float64)
                / exposure_s
            )
        resolved_active = np.asarray(
            getframes.to_numpy(sensor.engine.lenslet_valid), dtype=np.bool_
        )
        if active_lenslets is None:
            active_lenslets = resolved_active
        elif not np.array_equal(active_lenslets, resolved_active):
            raise AssertionError("active HAKA lenslets changed between SEDs")
        camera_config = sensor.detector.camera.config
        ensembles.append(RateEnsemble(sed, np.stack(state_rates)))
    if active_lenslets is None:
        raise AssertionError("at least one SED is required")
    return tuple(ensembles), active_lenslets, camera_config


def _curve(
    magnitudes: NDArray[np.float64],
    modes: tuple[CameraMode, ...],
    ensembles: tuple[RateEnsemble, ...],
    *,
    frame_rate_column: str,
    camera: Any,
    active_lenslets: NDArray[np.bool_],
) -> dict[str, list[float]]:
    values = {ensemble.sed.label: [] for ensemble in ensembles}
    for magnitude in magnitudes:
        mode = select_camera_mode(float(magnitude), modes)
        for ensemble in ensembles:
            values[ensemble.sed.label].append(
                evaluate_mode(
                    ensemble,
                    magnitude_r=float(magnitude),
                    frame_rate_hz=mode.frame_rate_hz(frame_rate_column),
                    em_gain=mode.em_gain,
                    camera=camera,
                    active_lenslets=active_lenslets,
                ).mean
            )
    return values


def _save_candidate_lut(path: Path, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "magnitude_R",
        "target_snr",
        "empirical_frame_rate_floor_hz",
        "target_snr_frame_rate_hz",
        "proposed_em_gain",
        "proposed_framerate_hz",
        "empirical_floor_active",
        "target_snr_met",
        "minimum_snr_across_seds",
        "mean_snr_across_seds",
        "maximum_snr_across_seds",
        "max_5sigma_saturation_fraction",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "magnitude_R": result["magnitude_r"],
                    "target_snr": result["target_snr"],
                    "empirical_frame_rate_floor_hz": result["empirical_frame_rate_floor_hz"],
                    "target_snr_frame_rate_hz": result["target_snr_frame_rate_hz"],
                    "proposed_em_gain": result["proposed_em_gain"],
                    "proposed_framerate_hz": result["proposed_frame_rate_hz"],
                    "empirical_floor_active": result["empirical_floor_active"],
                    "target_snr_met": result["target_snr_met"],
                    "minimum_snr_across_seds": result["minimum_snr_across_seds"],
                    "mean_snr_across_seds": result["mean_snr_across_seds"],
                    "maximum_snr_across_seds": result["maximum_snr_across_seds"],
                    "max_5sigma_saturation_fraction": result["five_sigma_saturation_fraction"][
                        "maximum"
                    ],
                }
            )


def _save_plot(
    path: Path,
    *,
    magnitudes: NDArray[np.float64],
    current_curve: dict[str, list[float]],
    proposed_curve: dict[str, list[float]],
    current_modes: tuple[CameraMode, ...],
    proposal: list[dict[str, Any]],
    frame_rate_column: str,
    target_snr: float,
) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(3, 1, figsize=(10.5, 10.0), sharex=True, layout="constrained")
    figure.suptitle("Keck II HAKA: empirical LUT SNR and smooth flux-fitted cadence floor")
    current_matrix = np.asarray([current_curve[sed.label] for sed in REPRESENTATIVE_SEDS])
    proposed_matrix = np.asarray([proposed_curve[sed.label] for sed in REPRESENTATIVE_SEDS])
    for sed in REPRESENTATIVE_SEDS:
        axes[0].plot(
            magnitudes,
            current_curve[sed.label],
            color=sed.color,
            linewidth=1.4,
            alpha=0.85,
            label=f"{sed.label} ({sed.temperature_k:.0f} K)",
        )
    axes[0].plot(
        magnitudes,
        np.mean(current_matrix, axis=0),
        color="black",
        linewidth=2.7,
        label="SED mean, current LUT",
    )
    axes[0].plot(
        magnitudes,
        np.mean(proposed_matrix, axis=0),
        color="black",
        linestyle="--",
        linewidth=2.0,
        label="SED mean, empirical-floor policy",
    )
    axes[0].fill_between(
        magnitudes,
        np.min(proposed_matrix, axis=0),
        np.max(proposed_matrix, axis=0),
        color="#7b2cbf",
        alpha=0.14,
        label="Empirical-floor-policy SED range",
    )
    axes[0].axhspan(
        4.0,
        5.0,
        color="#f2c14e",
        alpha=0.15,
        label=f"Accepted SNR 4--5 (reference {target_snr:g})",
    )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Mean 4x4 lenslet intensity SNR")
    axes[0].grid(alpha=0.25, which="both")
    axes[0].legend(ncol=2, fontsize=8)
    flux_axis = axes[0].secondary_xaxis(
        "top",
        functions=(
            lambda magnitude: 10.0 ** (-0.4 * (magnitude - REFERENCE_R_MAGNITUDE)),
            lambda flux: (
                REFERENCE_R_MAGNITUDE - 2.5 * np.log10(np.maximum(flux, np.finfo(float).tiny))
            ),
        ),
    )
    flux_axis.set_xscale("log")
    flux_axis.invert_xaxis()
    flux_axis.set_xlabel("Relative Johnson R flux (R=10 is 1; logarithmic axis)")

    current_rates = [
        select_camera_mode(float(value), current_modes).frame_rate_hz(frame_rate_column)
        for value in magnitudes
    ]
    current_gains = [
        select_camera_mode(float(value), current_modes).em_gain for value in magnitudes
    ]
    proposed_rates = [result["proposed_frame_rate_hz"] for result in proposal]
    empirical_floor_rates = [result["empirical_frame_rate_floor_hz"] for result in proposal]
    proposed_gains = [result["proposed_em_gain"] for result in proposal]
    axes[1].step(magnitudes, current_rates, where="post", color="#555555", label="Current")
    axes[1].plot(
        magnitudes,
        empirical_floor_rates,
        color="#d18f00",
        linestyle=":",
        linewidth=2.0,
        label="Faint-tail smooth broken power law (2067 Hz limit)",
    )
    axes[1].plot(
        magnitudes, proposed_rates, color="#7b2cbf", label="Final cadence (floor or faster)"
    )
    axes[1].set_ylabel("Frame rate (Hz)")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    axes[2].step(magnitudes, current_gains, where="post", color="#555555", label="Current")
    axes[2].plot(magnitudes, proposed_gains, color="#7b2cbf", label="Optimized EM gain")
    axes[2].set_ylabel("EM gain")
    axes[2].set_xlabel("Vega Johnson R magnitude")
    axes[2].set_xlim(float(magnitudes[0]), float(magnitudes[-1]))
    axes[2].grid(alpha=0.25)
    axes[2].legend()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    args = _parse_arguments()
    _validate_arguments(args)
    base = makewfs.load_config(HERE / "keck_haka.toml")
    base = replace(base, numerics=replace(base.numerics, device=args.device))
    pupil = make_keck_pupil()
    collecting_area_m2 = pupil_collecting_area_m2(pupil)
    current_modes = load_camera_modes()
    empirical_fit = fit_empirical_frame_rate_model(
        current_modes, frame_rate_column=args.frame_rate_column
    )

    with tempfile.TemporaryDirectory(prefix="makewfs-haka-lut-") as temporary:
        pupil_path = Path(temporary) / "keck-primary.npy"
        np.save(pupil_path, pupil)
        opd_states = _generate_opd_states(args, base)
        ensembles, active_lenslets, camera = _build_rate_ensembles(
            args,
            base=base,
            pupil_path=pupil_path,
            collecting_area_m2=collecting_area_m2,
            opd_states=opd_states,
        )

    sample_count = math.floor(
        (args.maximum_magnitude_r - args.minimum_magnitude_r) / args.magnitude_step
    )
    magnitudes = args.minimum_magnitude_r + np.arange(sample_count + 1) * args.magnitude_step
    if magnitudes[-1] < args.maximum_magnitude_r - 1e-12:
        magnitudes = np.append(magnitudes, args.maximum_magnitude_r)
    magnitudes[-1] = min(magnitudes[-1], args.maximum_magnitude_r)
    target_solutions = [
        optimize_at_magnitude(
            float(magnitude),
            target_snr=args.target_snr,
            minimum_frame_rate_hz=args.minimum_frame_rate_hz,
            ensembles=ensembles,
            camera=camera,
            active_lenslets=active_lenslets,
        )
        for magnitude in magnitudes
    ]
    optimization = [
        apply_empirical_frame_rate_floor(
            target_result,
            empirical_floor_hz=float(empirical_fit.frame_rate_hz(float(magnitude))),
            ensembles=ensembles,
            camera=camera,
            active_lenslets=active_lenslets,
        )
        for magnitude, target_result in zip(magnitudes, target_solutions, strict=True)
    ]
    current_curve = _curve(
        magnitudes,
        current_modes,
        ensembles,
        frame_rate_column=args.frame_rate_column,
        camera=camera,
        active_lenslets=active_lenslets,
    )
    proposed_curve = {
        ensemble.sed.label: [
            result["mean_snr_by_sed"][ensemble.sed.label] for result in optimization
        ]
        for ensemble in ensembles
    }

    output_path = Path(args.output).resolve()
    manifest_path = Path(args.manifest).resolve()
    candidate_path = Path(args.candidate_lut).resolve()
    _save_plot(
        output_path,
        magnitudes=magnitudes,
        current_curve=current_curve,
        proposed_curve=proposed_curve,
        current_modes=current_modes,
        proposal=optimization,
        frame_rate_column=args.frame_rate_column,
        target_snr=args.target_snr,
    )
    _save_candidate_lut(candidate_path, optimization)
    manifest = {
        "metric": {
            "name": "mean active-lenslet 4x4 total-intensity SNR",
            "signal": "expected dark-subtracted sum of all 16 native pixels per lenslet",
            "noise": [
                "photon shot noise",
                "sqrt(2) high-gain EMCCD excess noise (gain > 1)",
                "dark-current shot noise",
                "clock-induced-charge shot noise",
                "output-amplifier read noise",
                "ADC quantization",
            ],
            "average": "arithmetic mean over active lenslets and open-loop atmosphere states",
            "excluded": [
                "master-dark estimation uncertainty (perfect matched mean subtraction assumed)",
                "sky background",
                "centroid/slope algorithm",
                "wavefront reconstruction and temporal-control error",
            ],
        },
        "magnitude_system": "Vega Johnson R",
        "reference_magnitude_r": REFERENCE_R_MAGNITUDE,
        "representative_seds": [asdict(sed) for sed in REPRESENTATIVE_SEDS],
        "atmosphere": {
            "profile": "mauna-kea",
            "engine": "extrude",
            "seeing_arcsec_at_500_nm": args.seeing,
            "states": args.states,
            "state_separation_s": args.atmosphere_step_s,
            "seed": args.seed,
        },
        "airmass": args.airmass,
        "active_lenslets": int(np.count_nonzero(active_lenslets)),
        "frame_rate_column": args.frame_rate_column,
        "empirical_frame_rate_model": {
            "kind": "faint-tail smooth broken power law with 2067 Hz asymptote",
            "formula": (
                "fps = 2067 * (1 + (transition_flux/relative_R_flux)^sharpness)"
                "^(-faint_exponent/sharpness)"
            ),
            "relative_flux_definition": "10^(-0.4*(R-10))",
            "fitted_tail_minimum_magnitude_r": EMPIRICAL_TAIL_MIN_MAGNITUDE_R,
            "excluded_bright_rows_basis": (
                "2000 Hz is a human-rounded representation of the 2067 Hz camera ceiling; "
                "1500 Hz is a coarse convenient step"
            ),
            "maximum_frame_rate_hz": empirical_fit.maximum_frame_rate_hz,
            "transition_relative_flux": empirical_fit.transition_relative_flux,
            "transition_magnitude_r": (
                REFERENCE_R_MAGNITUDE - 2.5 * math.log10(empirical_fit.transition_relative_flux)
            ),
            "faint_flux_exponent": empirical_fit.flux_exponent,
            "transition_sharpness": empirical_fit.transition_sharpness,
            "rms_log10_frame_rate_residual": empirical_fit.rms_log10_residual,
            "rms_multiplicative_factor": 10.0**empirical_fit.rms_log10_residual,
            "anchors": [
                {
                    "magnitude_r": magnitude,
                    "relative_flux_r10": float(empirical_fit.relative_flux(magnitude)),
                    "empirical_frame_rate_hz": frame_rate,
                    "fitted_frame_rate_hz": float(empirical_fit.frame_rate_hz(magnitude)),
                    "fitted_to_empirical_ratio": (
                        float(empirical_fit.frame_rate_hz(magnitude)) / frame_rate
                    ),
                }
                for magnitude, frame_rate in zip(
                    empirical_fit.anchor_magnitudes_r,
                    empirical_fit.anchor_frame_rates_hz,
                    strict=True,
                )
            ],
            "role": "hard lower bound on proposed frame rate",
        },
        "optimization_constraints": {
            "objective": "maximum continuous frame rate, then minimum continuous EM gain",
            "snr": (
                "meet the absolute target for every SED when possible without crossing below "
                "the empirical frame-rate floor; otherwise accept the floor's lower SNR"
            ),
            "target_snr": args.target_snr,
            "gain": "continuous EM gain from 1 through 600",
            "minimum_frame_rate_hz": args.minimum_frame_rate_hz,
            "maximum_frame_rate_hz": HAKA_OPTIMIZATION_MAX_FRAME_RATE_HZ,
            "saturation": (
                "mean plus five sigma below image-area full well, output-register "
                "full well, and ADC ceiling for every SED/state/pixel"
            ),
        },
        "optimization": optimization,
        "curve": {
            "magnitude_r": magnitudes.tolist(),
            "current_mean_snr_by_sed": current_curve,
            "proposed_mean_snr_by_sed": proposed_curve,
        },
        "outputs": {"plot": output_path.name, "candidate_lut": candidate_path.name},
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output_path}")
    print(f"wrote {manifest_path}")
    print(f"wrote {candidate_path}")
    target_met = sum(bool(result["target_snr_met"]) for result in optimization)
    print(
        f"continuous R={magnitudes[0]:g}--{magnitudes[-1]:g} policy meets SNR "
        f"{args.target_snr:g} at {target_met}/{len(optimization)} samples and never drops "
        "below the empirical cadence model"
    )


if __name__ == "__main__":
    main()
