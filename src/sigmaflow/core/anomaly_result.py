"""AnomalyResult: the output of every detector."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .signal_frame import SignalFrame

__all__ = ["AnomalyResult", "labels_to_intervals"]


def labels_to_intervals(
    labels: np.ndarray, time: pd.Index, scores: np.ndarray | None = None
) -> list[tuple]:
    """Merge contiguous anomalous timesteps into (start, end, severity) intervals.

    Severity is the maximum score inside the interval (1.0 if no scores given).
    """
    labels = np.asarray(labels).astype(int)
    intervals = []
    n = len(labels)
    i = 0
    while i < n:
        if labels[i] == 1:
            j = i
            while j + 1 < n and labels[j + 1] == 1:
                j += 1
            severity = float(np.max(scores[i : j + 1])) if scores is not None else 1.0
            intervals.append((time[i], time[j], severity))
            i = j + 1
        else:
            i += 1
    return intervals


class AnomalyResult:
    """Detection output: per-timestep scores and labels plus context."""

    def __init__(
        self,
        labels: np.ndarray,
        scores: np.ndarray,
        threshold: float,
        detector_name: str,
        parameters: dict[str, Any],
        signal: SignalFrame,
        computation_time: float = 0.0,
    ):
        self.labels = np.asarray(labels).astype(int)
        self.scores = np.asarray(scores, dtype=np.float64)
        self.threshold = float(threshold)
        self.detector_name = detector_name
        self.parameters = dict(parameters)
        self.signal_name = signal.name
        self.computation_time = float(computation_time)
        self._time = signal.time
        self._values = signal.values.copy()
        self.intervals = labels_to_intervals(self.labels, self._time, self.scores)

    @property
    def n_anomalies(self) -> int:
        return len(self.intervals)

    def __repr__(self) -> str:
        return (
            f"AnomalyResult(detector={self.detector_name!r}, "
            f"anomalies={self.n_anomalies}, threshold={self.threshold:.4g})"
        )

    def summary(self) -> str:
        """Print anomaly count, intervals, total anomalous duration, max score."""
        from .signal_frame import time_to_seconds

        name = self.signal_name or "signal"
        lines = [f"Detected {self.n_anomalies} anomalies in {name} ({self.detector_name}):"]
        secs = time_to_seconds(self._time)
        total = 0.0
        max_listed = 10
        for k, (start, end, severity) in enumerate(self.intervals, 1):
            i0 = self._time.get_indexer([start])[0]
            i1 = self._time.get_indexer([end])[0]
            total += secs[i1] - secs[i0]
            if k <= max_listed:
                lines.append(f"  {k}. [{start} -> {end}] score: {severity:.3g}")
        if self.n_anomalies > max_listed:
            lines.append(f"  ... and {self.n_anomalies - max_listed} more intervals")
        lines.append(f"  total anomalous duration: {total:.6g} s")
        if len(self.scores):
            lines.append(f"  max score: {float(np.max(self.scores)):.4g} "
                         f"(threshold: {self.threshold:.4g})")
        text = "\n".join(lines)
        print(text)
        return text

    def to_dataframe(self) -> pd.DataFrame:
        """Export as DataFrame with time, per-channel values, score, label."""
        df = self._values.copy()
        df["score"] = self.scores
        df["label"] = self.labels
        df.index.name = "time"
        return df

    # ------------------------------------------------------------------ #
    # Plotting (matplotlib imported lazily)
    # ------------------------------------------------------------------ #

    def plot(self, ax=None, channel: str | None = None):
        """Plot signal with detected anomaly intervals highlighted."""
        from ..viz.plots import plot_result

        return plot_result(self, ax=ax, channel=channel)

    def plot_scores(self, ax=None):
        """Plot anomaly scores with the threshold line."""
        from ..viz.plots import plot_result_scores

        return plot_result_scores(self, ax=ax)

    def plot_distribution(self, ax=None, bins: int = 50):
        """Histogram of anomaly scores with the threshold marked."""
        from ..viz.plots import plot_score_distribution

        return plot_score_distribution(self, ax=ax, bins=bins)
