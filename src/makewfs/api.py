"""Small public facade for configured wavefront-sensor simulations."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .config import WFSConfig, load_config
from .detector import DetectorAdapter
from .provenance import metadata
from .sensors.base import OpticalResult, SensorEngine
from .sensors.pyramid import PyramidEngine
from .sensors.shack_hartmann import ShackHartmannEngine
from .wavefront import iter_phase_samples


class WavefrontSensor:
    """Configured wavefront-sensor facade.

    Construct once and call :meth:`expose` for each closed-loop residual OPD.
    ``photon_rate`` exposes the deterministic optical result for workflows that
    use another detector or need an ideal reference image.
    """

    def __init__(self, config: WFSConfig) -> None:
        self.config = config
        self.engine: SensorEngine
        if config.sensor.kind == "shack_hartmann":
            self.engine = ShackHartmannEngine(config)
        elif config.sensor.kind == "pyramid":
            self.engine = PyramidEngine(config)
        else:
            raise NotImplementedError(f"unsupported sensor kind {config.sensor.kind!r}")
        self.detector = DetectorAdapter(config.detector, self.engine.output_shape)

    @classmethod
    def from_toml(cls, path: str | Path) -> WavefrontSensor:
        """Build a sensor from a validated TOML file."""
        return cls(load_config(path))

    def _render(self, wavefront: ArrayLike) -> OpticalResult:
        return self.engine.render(np.asarray(wavefront, dtype=np.float64))

    def photon_rate(self, wavefront: ArrayLike) -> NDArray[np.float64]:
        """Return the ideal native-pixel photon rate before detector noise."""
        return self._render(wavefront).photon_rate

    def reference(self) -> NDArray[np.float64]:
        """Return the ideal image for a zero dynamic OPD."""
        return self.photon_rate(np.zeros(self.config.input.shape, dtype=np.float64))

    def expose(self, wavefront: ArrayLike, *, seed: int | None = None) -> Any:
        """Render one wavefront and expose it through the configured detector."""
        result = self._render(wavefront)
        frame_metadata = metadata(
            self.config,
            sensor_kind=self.config.sensor.kind,
            launched_rate=result.launched_rate_per_s,
            captured_rate=result.captured_rate_per_s,
            opd_m=result.opd_m,
            seed=seed,
        )
        return self.detector.expose(result.photon_rate, metadata=frame_metadata, seed=seed)

    def expose_many(
        self, phases: Iterable[ArrayLike], seeds: Iterable[int | None] | None = None
    ) -> Iterator[Any]:
        """Yield one detector frame per phase sample without stacking the stream."""
        seed_iter = iter(seeds) if seeds is not None else None
        for phase in phases:
            seed = next(seed_iter) if seed_iter is not None else None
            yield self.expose(phase, seed=seed)

    def expose_integrated(
        self, phase_samples: ArrayLike | Iterable[ArrayLike], *, seed: int | None = None
    ) -> Any:
        """Expose one detector frame after uniformly averaging temporal OPD samples."""
        rates: list[NDArray[np.float64]] = []
        opds: list[NDArray[np.float64]] = []
        launched = 0.0
        captured = 0.0
        for sample in iter_phase_samples(phase_samples, self.config.input.shape):
            result = self._render(sample)
            rates.append(result.photon_rate)
            opds.append(result.opd_m)
            launched = result.launched_rate_per_s
            captured += result.captured_rate_per_s
        if not rates:
            raise ValueError("phase_samples must contain at least one sample")
        average_rate = np.mean(np.stack(rates, axis=0), axis=0)
        average_opd = np.mean(np.stack(opds, axis=0), axis=0)
        frame_metadata = metadata(
            self.config,
            sensor_kind=self.config.sensor.kind,
            launched_rate=launched,
            captured_rate=captured / len(rates),
            opd_m=average_opd,
            seed=seed,
        )
        frame_metadata["wfs_temporal_samples"] = len(rates)
        return self.detector.expose(average_rate, metadata=frame_metadata, seed=seed)


def simulate(
    wavefront: ArrayLike, config: WFSConfig | str | Path, *, seed: int | None = None
) -> Any:
    """One-shot convenience wrapper around :class:`WavefrontSensor`."""
    resolved = load_config(config) if isinstance(config, (str, Path)) else config
    return WavefrontSensor(resolved).expose(wavefront, seed=seed)


__all__ = ["WavefrontSensor", "simulate"]
