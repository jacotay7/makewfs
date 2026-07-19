"""Guide-source photon normalization using public ``getframes`` radiometry."""

from __future__ import annotations

from .config import SourceConfig, TelescopeConfig


def source_rate_per_s(source: SourceConfig, telescope: TelescopeConfig) -> float:
    """Return the configured total detector-surface photon rate."""
    if source.normalization == "detector_photon_rate":
        assert source.detector_photon_rate_per_s is not None
        return source.detector_photon_rate_per_s
    if source.band is None or source.magnitude is None:
        raise ValueError("magnitude source needs band and magnitude")
    try:
        import getframes as gf
    except ImportError as exc:  # pragma: no cover - dependency is required at install time
        raise ImportError("magnitude normalization requires getframes") from exc
    band = (
        gf.Bandpass.ab(source.band)
        if source.magnitude_system == "ab"
        else gf.Bandpass.johnson(source.band)
    )
    optical = gf.Telescope(
        aperture_diameter_m=telescope.pupil_diameter_m,
        plate_scale_arcsec_per_pixel=1.0,
        throughput=source.throughput,
        central_obstruction=telescope.central_obscuration_ratio,
        band=band,
    )
    return optical.photon_rate_from_magnitude(source.magnitude)


__all__ = ["source_rate_per_s"]
