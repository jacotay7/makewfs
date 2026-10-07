"""Guide-source photon normalization using public ``getframes`` radiometry."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .backend import ArrayBackend
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
    return float(optical.photon_rate_from_magnitude(source.magnitude))


def clear_aperture_fraction(
    pupil: Any,
    telescope: TelescopeConfig,
    shape: tuple[int, int],
    extent_m: float,
    *,
    supersampling: int,
    backend: ArrayBackend,
) -> float:
    """Return the sampled pupil's transmitted fraction of its unobstructed annulus.

    Magnitude normalization collects light over the analytic annulus
    ``pi / 4 D^2 (1 - eps^2)``, which knows nothing of spiders, segment gaps or
    a custom mask. This is the ratio of ``sum |pupil|^2`` to the same sum for
    the annulus alone, sampled identically, so a plain annulus gives exactly 1
    and every additional obstruction removes its share of the photons.
    """
    if (
        telescope.custom_mask_path is None
        and not telescope.spiders
        and (telescope.segments_across_pupil is None or telescope.segment_gap_fraction == 0.0)
    ):
        return 1.0
    from .pupil import make_pupil

    annulus = replace(
        telescope,
        spiders=(),
        custom_mask_path=None,
        segments_across_pupil=None,
        segment_gap_fraction=0.0,
    )
    reference = make_pupil(
        annulus, shape, extent_m, supersampling=supersampling, backend=backend, dtype=pupil.dtype
    )
    clear = backend.scalar(backend.sum(backend.abs(pupil) ** 2))
    full = backend.scalar(backend.sum(backend.abs(reference) ** 2))
    if full <= 0.0:
        raise ValueError("the telescope annulus has no illuminated pixels")
    return float(clear / full)


__all__ = ["clear_aperture_fraction", "source_rate_per_s"]
