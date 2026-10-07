"""Private CUDA-graph replay of a fixed-shape device computation.

On a GPU, a small sensor's frame is a few dozen short kernels. Their arithmetic
is cheap, so the frame rate is set by Python dispatch and kernel-launch cost on
the host. Capturing the fixed sequence once as a CUDA graph and replaying it
with one launch removes that cost while running exactly the same kernels, so
replayed results equal eager ones.

The capture runs on a private stream under a private memory pool that lives as
long as the graph: temporaries the graph writes on replay are never handed to
other arrays. Every call copies its outputs into new arrays, so a returned
result never aliases graph storage.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class _CudaGraphReplay:  # pragma: no cover - optional CUDA execution
    """Replay ``function(value)`` from a captured graph.

    ``function`` must take one device array of a fixed shape and dtype, return
    a tuple of device arrays, and never synchronize with the host. It is run
    eagerly once on the private stream (warming FFT plans, compiled kernels
    and the pool) and then captured. A device or driver that cannot capture
    falls back to eager calls for the rest of the sensor's life, recording the
    reason in ``failure``.
    """

    def __init__(self, xp: Any, function: Callable[[Any], tuple[Any, ...]]) -> None:
        self._xp = xp
        self._function = function
        self._graph: Any | None = None
        self._input: Any | None = None
        self._outputs: tuple[Any, ...] = ()
        self.failure: str | None = None

    def __call__(self, value: Any) -> tuple[Any, ...]:
        if self.failure is None and self._graph is None:
            self._capture(value)
        if self._graph is None or self._input is None:
            return self._function(value)
        if value.shape != self._input.shape or value.dtype != self._input.dtype:
            raise ValueError("captured graph input changed shape or dtype")
        current = self._xp.cuda.get_current_stream()
        # Order the replay after the caller's producer of ``value`` and after
        # the copies that read the previous replay's outputs.
        self._stream.wait_event(current.record())
        with self._stream:
            self._input[...] = value
            self._graph.launch()
        current.wait_event(self._stream.record())
        return tuple(output.copy() for output in self._outputs)

    def _capture(self, value: Any) -> None:
        xp = self._xp
        try:
            self._stream = xp.cuda.Stream(non_blocking=True)
            self._pool = xp.cuda.MemoryPool()
            current = xp.cuda.get_current_stream()
            with xp.cuda.using_allocator(self._pool.malloc), self._stream:
                self._stream.wait_event(current.record())
                staged = xp.empty_like(value)
                staged[...] = value
                self._function(staged)
                self._stream.synchronize()
                self._stream.begin_capture()
                try:
                    outputs = self._function(staged)
                finally:
                    graph = self._stream.end_capture()
            self._input = staged
            self._outputs = outputs
            self._graph = graph
        except Exception as exc:  # driver/runtime without usable graph capture
            self._graph = None
            self._input = None
            self._outputs = ()
            self.failure = f"{type(exc).__name__}: {exc}"


__all__ = ["_CudaGraphReplay"]
