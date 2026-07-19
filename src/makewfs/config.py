"""Validated, serializable configuration for :mod:`makewfs`.

The configuration layer intentionally contains no runtime arrays, FFT plans,
random generators, or detector objects.  It is the boundary at which all
physical values are validated and all paths are made relative to the TOML file.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib  # type: ignore[import-not-found]


class ConfigError(ValueError):
    """Raised when a configuration is invalid or uses an unknown field."""


def _strict(table: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(table) - allowed)
    if unknown:
        names = ", ".join(unknown)
        raise ConfigError(f"{path}: unknown key(s): {names}")


def _finite(
    value: Any, path: str, *, minimum: float | None = None, maximum: float | None = None
) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{path}: expected a finite number") from exc
    if not math.isfinite(number):
        raise ConfigError(f"{path}: expected a finite number")
    if minimum is not None and number < minimum:
        raise ConfigError(f"{path}: must be >= {minimum}")
    if maximum is not None and number > maximum:
        raise ConfigError(f"{path}: must be <= {maximum}")
    return number


def _positive_int(value: Any, path: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(f"{path}: expected an integer >= {minimum}")
    return int(value)


def _shape(value: Any, path: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ConfigError(f"{path}: expected [height, width]")
    return (_positive_int(value[0], f"{path}[0]"), _positive_int(value[1], f"{path}[1]"))


def _tuple_floats(value: Any, path: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        raise ConfigError(f"{path}: expected an array of numbers")
    return tuple(_finite(item, f"{path}[{i}]") for i, item in enumerate(value))


@dataclass(frozen=True)
class InputConfig:
    """Input array units and physical sampling."""

    quantity: str
    unit: str
    shape: tuple[int, int]
    grid_extent_m: float
    reference_wavelength_m: float | None = None
    static_opd_path: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, base: Path) -> InputConfig:
        _strict(
            data,
            {
                "quantity",
                "unit",
                "shape",
                "grid_extent_m",
                "reference_wavelength_m",
                "static_opd_path",
            },
            "input",
        )
        quantity = str(data.get("quantity", "opd")).lower()
        unit = str(data.get("unit", "m" if quantity == "opd" else "rad")).lower()
        if quantity not in {"opd", "phase"}:
            raise ConfigError("input.quantity: expected 'opd' or 'phase'")
        expected_unit = "m" if quantity == "opd" else "rad"
        if unit != expected_unit:
            raise ConfigError(f"input.unit: {quantity!r} requires {expected_unit!r}")
        shape = _shape(data.get("shape"), "input.shape")
        extent = _finite(data.get("grid_extent_m", 0), "input.grid_extent_m", minimum=1e-15)
        reference = data.get("reference_wavelength_m")
        reference_value = (
            None
            if reference is None
            else _finite(reference, "input.reference_wavelength_m", minimum=1e-15)
        )
        if quantity == "phase" and reference_value is None:
            raise ConfigError("input.reference_wavelength_m: required for phase input")
        static = data.get("static_opd_path")
        static_path = None if static is None else str((base / str(static)).resolve())
        return cls(quantity, unit, shape, extent, reference_value, static_path)


@dataclass(frozen=True)
class SpiderConfig:
    """One radial spider vane."""

    angle_deg: float
    width_fraction: float

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], path: str) -> SpiderConfig:
        _strict(data, {"angle_deg", "width_fraction"}, path)
        return cls(
            _finite(data.get("angle_deg", 0), f"{path}.angle_deg"),
            _finite(data.get("width_fraction", 0), f"{path}.width_fraction", minimum=0, maximum=1),
        )


@dataclass(frozen=True)
class TelescopeConfig:
    """Entrance-pupil geometry."""

    pupil_diameter_m: float
    central_obscuration_ratio: float = 0.0
    spiders: tuple[SpiderConfig, ...] = ()
    pupil_rotation_deg: float = 0.0
    custom_mask_path: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, base: Path) -> TelescopeConfig:
        _strict(
            data,
            {
                "pupil_diameter_m",
                "central_obscuration_ratio",
                "spiders",
                "pupil_rotation_deg",
                "custom_mask_path",
            },
            "telescope",
        )
        raw_spiders = data.get("spiders", [])
        if not isinstance(raw_spiders, list):
            raise ConfigError("telescope.spiders: expected an array of tables")
        spiders = tuple(
            SpiderConfig.from_dict(item, f"telescope.spiders[{i}]")
            for i, item in enumerate(raw_spiders)
        )
        custom = data.get("custom_mask_path")
        custom_path = None if custom is None else str((base / str(custom)).resolve())
        return cls(
            _finite(data.get("pupil_diameter_m"), "telescope.pupil_diameter_m", minimum=1e-15),
            _finite(
                data.get("central_obscuration_ratio", 0),
                "telescope.central_obscuration_ratio",
                minimum=0,
                maximum=0.999999,
            ),
            spiders,
            _finite(data.get("pupil_rotation_deg", 0), "telescope.pupil_rotation_deg"),
            custom_path,
        )


@dataclass(frozen=True)
class SourceConfig:
    """Guide-source normalization and optional source morphology."""

    kind: str
    normalization: str
    detector_photon_rate_per_s: float | None = None
    magnitude: float | None = None
    magnitude_system: str = "vega"
    band: str | None = None
    throughput: float = 1.0
    field_angle_arcsec: tuple[float, float] = (0.0, 0.0)
    angular_fwhm_arcsec: float = 0.0
    angular_quadrature_order: int = 3
    wavelengths_m: tuple[float, ...] = ()
    wavelength_weights: tuple[float, ...] = ()
    lgs_ranges_m: tuple[float, ...] = ()
    lgs_range_weights: tuple[float, ...] = ()
    lgs_launch_position_m: tuple[float, float] = (0.0, 0.0)
    sed_path: str | None = None
    transmission_path: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, base: Path | None = None) -> SourceConfig:
        _strict(
            data,
            {
                "kind",
                "normalization",
                "detector_photon_rate_per_s",
                "magnitude",
                "magnitude_system",
                "band",
                "throughput",
                "field_angle_arcsec",
                "angular_fwhm_arcsec",
                "angular_quadrature_order",
                "wavelengths_m",
                "wavelength_weights",
                "lgs_ranges_m",
                "lgs_range_weights",
                "lgs_launch_position_m",
                "sed_path",
                "transmission_path",
            },
            "source",
        )
        kind = str(data.get("kind", "ngs")).lower()
        if kind not in {"ngs", "lgs"}:
            raise ConfigError("source.kind: expected 'ngs' or 'lgs'")
        normalization = str(data.get("normalization", "detector_photon_rate")).lower()
        if normalization not in {"detector_photon_rate", "magnitude"}:
            raise ConfigError(
                "source.normalization: expected 'detector_photon_rate' or 'magnitude'"
            )
        rate = data.get("detector_photon_rate_per_s")
        magnitude = data.get("magnitude")
        if normalization == "detector_photon_rate":
            if rate is None or magnitude is not None:
                raise ConfigError(
                    "source: direct-rate normalization requires only detector_photon_rate_per_s"
                )
            rate_value = _finite(rate, "source.detector_photon_rate_per_s", minimum=0)
            magnitude_value = None
        else:
            if magnitude is None or rate is not None:
                raise ConfigError("source: magnitude normalization requires only magnitude")
            if kind == "lgs":
                raise ConfigError(
                    "source: LGS return flux must use detector_photon_rate normalization"
                )
            rate_value = None
            magnitude_value = _finite(magnitude, "source.magnitude")
        system = str(data.get("magnitude_system", "vega")).lower()
        if system not in {"vega", "ab"}:
            raise ConfigError("source.magnitude_system: expected 'vega' or 'ab'")
        band = None if data.get("band") is None else str(data["band"])
        if normalization == "magnitude" and not band:
            raise ConfigError("source.band: required for magnitude normalization")
        angle = _tuple_floats(
            data.get("field_angle_arcsec", [0.0, 0.0]), "source.field_angle_arcsec"
        )
        if len(angle) != 2:
            raise ConfigError("source.field_angle_arcsec: expected [x, y]")
        angular_fwhm = _finite(
            data.get("angular_fwhm_arcsec", 0),
            "source.angular_fwhm_arcsec",
            minimum=0,
        )
        angular_order = _positive_int(
            data.get("angular_quadrature_order", 3),
            "source.angular_quadrature_order",
        )
        if angular_fwhm > 0 and angular_order < 2:
            raise ConfigError(
                "source.angular_quadrature_order: finite source extent requires order >= 2"
            )
        wavelengths = _tuple_floats(data.get("wavelengths_m", []), "source.wavelengths_m")
        if any(value <= 0 for value in wavelengths):
            raise ConfigError("source.wavelengths_m: values must be positive")
        weights = _tuple_floats(data.get("wavelength_weights", []), "source.wavelength_weights")
        if weights and (
            len(weights) != len(wavelengths)
            or any(value < 0 for value in weights)
            or sum(weights) <= 0
        ):
            raise ConfigError(
                "source.wavelength_weights: must be non-negative and match wavelengths_m"
            )
        lgs_ranges = _tuple_floats(data.get("lgs_ranges_m", []), "source.lgs_ranges_m")
        if any(value <= 0 for value in lgs_ranges):
            raise ConfigError("source.lgs_ranges_m: values must be positive")
        lgs_weights = _tuple_floats(data.get("lgs_range_weights", []), "source.lgs_range_weights")
        if lgs_weights and (
            len(lgs_weights) != len(lgs_ranges)
            or any(value < 0 for value in lgs_weights)
            or sum(lgs_weights) <= 0
        ):
            raise ConfigError(
                "source.lgs_range_weights: must be non-negative and match lgs_ranges_m"
            )
        launch = _tuple_floats(
            data.get("lgs_launch_position_m", [0.0, 0.0]),
            "source.lgs_launch_position_m",
        )
        if len(launch) != 2:
            raise ConfigError("source.lgs_launch_position_m: expected [x, y]")
        if kind != "lgs" and (lgs_ranges or lgs_weights or any(value != 0 for value in launch)):
            raise ConfigError("source: LGS geometry fields require source.kind = 'lgs'")
        base_path = Path.cwd() if base is None else base
        sed = data.get("sed_path")
        transmission = data.get("transmission_path")
        sed_path = None if sed is None else str((base_path / str(sed)).resolve())
        transmission_path = (
            None if transmission is None else str((base_path / str(transmission)).resolve())
        )
        return cls(
            kind,
            normalization,
            rate_value,
            magnitude_value,
            system,
            band,
            _finite(data.get("throughput", 1), "source.throughput", minimum=0, maximum=1),
            (angle[0], angle[1]),
            angular_fwhm,
            angular_order,
            wavelengths,
            weights,
            lgs_ranges,
            lgs_weights,
            (launch[0], launch[1]),
            sed_path,
            transmission_path,
        )


@dataclass(frozen=True)
class SensorConfig:
    """Common sensor selection."""

    kind: str
    wavelength_m: float

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SensorConfig:
        _strict(data, {"kind", "wavelength_m"}, "sensor")
        kind = str(data.get("kind", "shack_hartmann")).lower()
        if kind not in {"shack_hartmann", "pyramid"}:
            raise ConfigError("sensor.kind: expected 'shack_hartmann' or 'pyramid'")
        return cls(kind, _finite(data.get("wavelength_m"), "sensor.wavelength_m", minimum=1e-15))


@dataclass(frozen=True)
class ShackHartmannConfig:
    """Normalized square-lenslet Shack-Hartmann settings."""

    lenslets_across_pupil: int
    pixels_per_subaperture: int
    spot_sampling_pixels_per_lambda_over_d: float | None
    minimum_illuminated_fraction: float
    lenslet_fill_factor: float = 1.0
    lenslet_focal_length_m: float | None = None
    detector_pixel_pitch_m: float | None = None
    relay_magnification: float = 1.0
    field_stop_radius_lambda_over_d: float | None = None
    optical_blur_fwhm_pixels: float = 0.0
    detector_margin_pixels: int = 0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ShackHartmannConfig:
        _strict(
            data,
            {
                "lenslets_across_pupil",
                "pixels_per_subaperture",
                "spot_sampling_pixels_per_lambda_over_d",
                "minimum_illuminated_fraction",
                "lenslet_fill_factor",
                "lenslet_focal_length_m",
                "detector_pixel_pitch_m",
                "relay_magnification",
                "field_stop_radius_lambda_over_d",
                "optical_blur_fwhm_pixels",
                "detector_margin_pixels",
            },
            "shack_hartmann",
        )
        raw_sampling = data.get("spot_sampling_pixels_per_lambda_over_d")
        focal = data.get("lenslet_focal_length_m")
        pixel_pitch = data.get("detector_pixel_pitch_m")
        if raw_sampling is None and (focal is None or pixel_pitch is None):
            raise ConfigError(
                "shack_hartmann: provide spot_sampling_pixels_per_lambda_over_d or "
                "both lenslet_focal_length_m and detector_pixel_pitch_m"
            )
        if raw_sampling is not None and (focal is not None or pixel_pitch is not None):
            raise ConfigError(
                "shack_hartmann: normalized and physical sampling modes are mutually exclusive"
            )
        return cls(
            _positive_int(
                data.get("lenslets_across_pupil"), "shack_hartmann.lenslets_across_pupil", minimum=1
            ),
            _positive_int(
                data.get("pixels_per_subaperture"),
                "shack_hartmann.pixels_per_subaperture",
                minimum=2,
            ),
            None
            if raw_sampling is None
            else _finite(
                raw_sampling,
                "shack_hartmann.spot_sampling_pixels_per_lambda_over_d",
                minimum=0.5,
            ),
            _finite(
                data.get("minimum_illuminated_fraction", 0.25),
                "shack_hartmann.minimum_illuminated_fraction",
                minimum=0,
                maximum=1,
            ),
            _finite(
                data.get("lenslet_fill_factor", 1),
                "shack_hartmann.lenslet_fill_factor",
                minimum=0,
                maximum=1,
            ),
            None
            if focal is None
            else _finite(focal, "shack_hartmann.lenslet_focal_length_m", minimum=1e-15),
            None
            if pixel_pitch is None
            else _finite(pixel_pitch, "shack_hartmann.detector_pixel_pitch_m", minimum=1e-15),
            _finite(
                data.get("relay_magnification", 1),
                "shack_hartmann.relay_magnification",
                minimum=1e-15,
            ),
            None
            if data.get("field_stop_radius_lambda_over_d") is None
            else _finite(
                data["field_stop_radius_lambda_over_d"],
                "shack_hartmann.field_stop_radius_lambda_over_d",
                minimum=0,
            ),
            _finite(
                data.get("optical_blur_fwhm_pixels", 0),
                "shack_hartmann.optical_blur_fwhm_pixels",
                minimum=0,
            ),
            _positive_int(
                data.get("detector_margin_pixels", 0),
                "shack_hartmann.detector_margin_pixels",
                minimum=0,
            ),
        )


@dataclass(frozen=True)
class PyramidConfig:
    """Four-face pyramid settings."""

    pixels_across_pupil: int
    pupil_separation_pixels: int
    modulation_radius_lambda_over_d: float = 0.0
    modulation_samples: int = 1
    detector_margin_pixels: int = 0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PyramidConfig:
        _strict(
            data,
            {
                "pixels_across_pupil",
                "pupil_separation_pixels",
                "modulation_radius_lambda_over_d",
                "modulation_samples",
                "detector_margin_pixels",
            },
            "pyramid",
        )
        radius = _finite(
            data.get("modulation_radius_lambda_over_d", 0),
            "pyramid.modulation_radius_lambda_over_d",
            minimum=0,
        )
        samples = _positive_int(data.get("modulation_samples", 1), "pyramid.modulation_samples")
        if radius == 0 and samples != 1:
            raise ConfigError(
                "pyramid.modulation_samples: zero modulation requires exactly one sample"
            )
        if radius > 0 and samples < 4:
            raise ConfigError(
                "pyramid.modulation_samples: modulated mode requires at least four samples"
            )
        return cls(
            _positive_int(
                data.get("pixels_across_pupil"), "pyramid.pixels_across_pupil", minimum=8
            ),
            _positive_int(
                data.get("pupil_separation_pixels"), "pyramid.pupil_separation_pixels", minimum=1
            ),
            radius,
            samples,
            _positive_int(
                data.get("detector_margin_pixels", 0),
                "pyramid.detector_margin_pixels",
                minimum=0,
            ),
        )


@dataclass(frozen=True)
class DetectorConfig:
    """Detector construction and exposure settings."""

    preset: str | None
    inline: dict[str, Any]
    exposure_s: float
    temperature_c: float | None
    binning: int
    binning_mode: str
    precision: str
    include_truth: bool

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DetectorConfig:
        allowed = {
            "preset",
            "camera",
            "exposure_s",
            "temperature_c",
            "binning",
            "binning_mode",
            "precision",
            "include_truth",
        }
        _strict(data, allowed, "detector")
        preset = None if data.get("preset") is None else str(data["preset"])
        inline = dict(data.get("camera", {}))
        if preset is None and not inline:
            raise ConfigError("detector: provide preset or inline camera config")
        if preset is not None and inline:
            raise ConfigError("detector: provide preset or inline camera config, not both")
        mode = str(data.get("binning_mode", "digital"))
        if mode not in {"digital", "on_chip"}:
            raise ConfigError("detector.binning_mode: expected 'digital' or 'on_chip'")
        precision = str(data.get("precision", "float64"))
        if precision not in {"float32", "float64"}:
            raise ConfigError("detector.precision: expected 'float32' or 'float64'")
        include_truth = data.get("include_truth", True)
        if not isinstance(include_truth, bool):
            raise ConfigError("detector.include_truth: expected a boolean")
        return cls(
            preset,
            inline,
            _finite(data.get("exposure_s"), "detector.exposure_s", minimum=0),
            None
            if data.get("temperature_c") is None
            else _finite(data["temperature_c"], "detector.temperature_c"),
            _positive_int(data.get("binning", 1), "detector.binning"),
            mode,
            precision,
            include_truth,
        )


@dataclass(frozen=True)
class NumericsConfig:
    """Numerical precision and FFT controls."""

    dtype: str = "float64"
    fft_oversampling: int = 2
    fft_workers: int = 1
    pupil_samples_per_lenslet: int | None = None
    pupil_supersampling: int = 1

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> NumericsConfig:
        _strict(
            data,
            {
                "dtype",
                "fft_oversampling",
                "fft_workers",
                "pupil_samples_per_lenslet",
                "pupil_supersampling",
            },
            "numerics",
        )
        dtype = str(data.get("dtype", "float64"))
        if dtype not in {"float32", "float64"}:
            raise ConfigError("numerics.dtype: expected 'float32' or 'float64'")
        samples = data.get("pupil_samples_per_lenslet")
        sample_value = (
            None
            if samples is None
            else _positive_int(samples, "numerics.pupil_samples_per_lenslet", minimum=4)
        )
        return cls(
            dtype,
            _positive_int(data.get("fft_oversampling", 2), "numerics.fft_oversampling"),
            _positive_int(data.get("fft_workers", 1), "numerics.fft_workers"),
            sample_value,
            _positive_int(data.get("pupil_supersampling", 1), "numerics.pupil_supersampling"),
        )


@dataclass(frozen=True)
class WFSConfig:
    """Complete immutable makewfs configuration."""

    schema_version: int
    input: InputConfig
    telescope: TelescopeConfig
    source: SourceConfig
    sensor: SensorConfig
    detector: DetectorConfig
    numerics: NumericsConfig
    shack_hartmann: ShackHartmannConfig | None = None
    pyramid: PyramidConfig | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    source_path: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, base: Path | None = None) -> WFSConfig:
        base_path = Path.cwd() if base is None else base
        allowed = {
            "schema_version",
            "input",
            "telescope",
            "source",
            "sensor",
            "detector",
            "numerics",
            "shack_hartmann",
            "pyramid",
            "metadata",
        }
        _strict(data, allowed, "config")
        version = data.get("schema_version")
        if version != 1:
            raise ConfigError("schema_version: only version 1 is supported")

        def table(name: str) -> Mapping[str, Any]:
            value = data.get(name, {})
            if not isinstance(value, Mapping):
                raise ConfigError(f"{name}: expected a table")
            return value

        input_config = InputConfig.from_dict(table("input"), base=base_path)
        telescope_config = TelescopeConfig.from_dict(table("telescope"), base=base_path)
        source_config = SourceConfig.from_dict(table("source"), base=base_path)
        sensor = SensorConfig.from_dict(table("sensor"))
        if sensor.kind == "shack_hartmann" and data.get("pyramid") is not None:
            raise ConfigError("pyramid: remove this table for a Shack-Hartmann sensor")
        if sensor.kind == "pyramid" and data.get("shack_hartmann") is not None:
            raise ConfigError("shack_hartmann: remove this table for a pyramid sensor")
        sh_data = table("shack_hartmann") if sensor.kind == "shack_hartmann" else None
        pyramid_data = table("pyramid") if sensor.kind == "pyramid" else None
        if sensor.kind == "shack_hartmann" and sh_data is None:
            raise ConfigError("shack_hartmann: required for Shack-Hartmann sensor")
        if sensor.kind == "pyramid" and pyramid_data is None:
            raise ConfigError("pyramid: required for pyramid sensor")
        return cls(
            1,
            input_config,
            telescope_config,
            source_config,
            sensor,
            DetectorConfig.from_dict(table("detector")),
            NumericsConfig.from_dict(table("numerics")),
            None if sh_data is None else ShackHartmannConfig.from_dict(sh_data),
            None if pyramid_data is None else PyramidConfig.from_dict(pyramid_data),
            dict(table("metadata")),
        )

    @classmethod
    def from_toml(cls, path: str | Path) -> WFSConfig:
        """Load and validate a TOML configuration, resolving relative paths."""
        config_path = Path(path).expanduser().resolve()
        with config_path.open("rb") as handle:
            data = tomllib.load(handle)
        return cls.from_dict(data, base=config_path.parent)._with_source_path(config_path)

    def _with_source_path(self, path: Path) -> WFSConfig:
        return WFSConfig(
            self.schema_version,
            self.input,
            self.telescope,
            self.source,
            self.sensor,
            self.detector,
            self.numerics,
            self.shack_hartmann,
            self.pyramid,
            dict(self.metadata),
            str(path),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/TOML-friendly representation."""

        def serialise(value: Any) -> Any:
            if isinstance(value, tuple):
                return [serialise(item) for item in value]
            if isinstance(value, dict):
                return {key: serialise(item) for key, item in value.items()}
            return value

        data = cast(dict[str, Any], serialise(asdict(self)))
        data.pop("source_path", None)
        detector = data.get("detector")
        if isinstance(detector, dict):
            detector["camera"] = detector.pop("inline", {})
        return data

    @property
    def digest(self) -> str:
        """Short SHA-256 digest of the normalized configuration."""
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), default=str
        ).encode()
        return hashlib.sha256(payload).hexdigest()[:16]


Config = WFSConfig


def load_config(path: str | Path) -> WFSConfig:
    """Load a validated TOML configuration."""
    return WFSConfig.from_toml(path)


__all__ = [
    "Config",
    "ConfigError",
    "DetectorConfig",
    "InputConfig",
    "NumericsConfig",
    "PyramidConfig",
    "ShackHartmannConfig",
    "SourceConfig",
    "TelescopeConfig",
    "WFSConfig",
    "load_config",
]
