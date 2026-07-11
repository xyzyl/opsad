"""Streaming/online detection: score data as it arrives."""

from __future__ import annotations

import numpy as np

from .core.anomaly_result import AnomalyResult
from .core.base import BaseDetector
from .core.signal_frame import SignalFrame

__all__ = ["StreamingDetector"]


class StreamingDetector:
    """Wrap any fitted detector for incremental scoring.

    Keeps a rolling context of the most recent ``context_size`` samples;
    each :meth:`update` scores the new chunk *with* that context (so
    windowed and rolling detectors see enough history) and returns an
    AnomalyResult for just the new samples.

    >>> stream = StreamingDetector(detector.fit(history), context_size=2000)
    >>> for t_chunk, x_chunk in feed:
    ...     result = stream.update(t_chunk, x_chunk)
    ...     if result.n_anomalies:
    ...         alert(result)
    """

    def __init__(self, detector: BaseDetector, context_size: int = 2000):
        if context_size < 10:
            raise ValueError("context_size must be at least 10")
        self.detector = detector
        self.context_size = int(context_size)
        self._time: list = []
        self._values: dict[str, list] = {}
        self.n_seen = 0

    def _append(self, sf: SignalFrame) -> None:
        if not self._values:
            self._values = {c: [] for c in sf.channels}
        if set(sf.channels) != set(self._values):
            raise ValueError(
                f"chunk channels {sf.channels} differ from stream channels "
                f"{list(self._values)}"
            )
        self._time.extend(list(sf.time))
        for c in sf.channels:
            self._values[c].extend(sf[c].tolist())
        # trim to context window
        excess = len(self._time) - self.context_size
        if excess > 0:
            self._time = self._time[excess:]
            for c in self._values:
                self._values[c] = self._values[c][excess:]

    def update(self, time, values) -> AnomalyResult:
        """Feed a new chunk; returns detection results for that chunk only."""
        sf_chunk = SignalFrame(time=time, values=values)
        n_new = len(sf_chunk)
        self._append(sf_chunk)
        self.n_seen += n_new

        context = SignalFrame(
            time=self._time,
            values={c: np.asarray(v) for c, v in self._values.items()},
            name="stream",
        )
        full = self.detector.detect(context)
        # slice out just the freshly arrived samples
        tail = SignalFrame(
            time=context.time[-n_new:],
            values={c: context[c].to_numpy()[-n_new:] for c in context.channels},
            name="stream",
        )
        return AnomalyResult(
            labels=full.labels[-n_new:],
            scores=full.scores[-n_new:],
            threshold=full.threshold,
            detector_name=full.detector_name,
            parameters=full.parameters,
            signal=tail,
            computation_time=full.computation_time,
        )
