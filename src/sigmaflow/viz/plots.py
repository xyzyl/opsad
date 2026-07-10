"""Static matplotlib plots (minimal MVP set)."""

from __future__ import annotations

import numpy as np

__all__ = ["plot_signal", "plot_result", "plot_result_scores", "plot_score_distribution"]


def _get_ax(ax, figsize=(10, 4)):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


def plot_signal(sf, ax=None, channel: str | None = None):
    """Plot one channel of a SignalFrame (first channel by default)."""
    ax = _get_ax(ax)
    channel = channel or sf.channels[0]
    ax.plot(sf.time, sf[channel], lw=0.8)
    unit = sf.units.get(channel, "")
    ax.set_ylabel(f"{channel} [{unit}]" if unit else channel)
    ax.set_xlabel("time")
    ax.set_title(sf.name or "signal")
    return ax


def plot_result(result, ax=None, channel: str | None = None):
    """Signal with detected anomaly intervals shaded."""
    ax = _get_ax(ax)
    channel = channel or result._values.columns[0]
    ax.plot(result._time, result._values[channel], lw=0.8, label=channel)
    for start, end, severity in result.intervals:
        ax.axvspan(start, end, color="crimson", alpha=0.25)
    ax.set_xlabel("time")
    ax.set_ylabel(channel)
    ax.set_title(
        f"{result.signal_name or 'signal'} — {result.n_anomalies} anomalies "
        f"({result.detector_name})"
    )
    return ax


def plot_result_scores(result, ax=None):
    """Anomaly scores with the decision threshold."""
    ax = _get_ax(ax)
    ax.plot(result._time, result.scores, lw=0.8, color="tab:blue", label="score")
    ax.axhline(result.threshold, color="crimson", ls="--", label=f"threshold={result.threshold:.3g}")
    ax.set_xlabel("time")
    ax.set_ylabel("anomaly score")
    ax.legend(loc="upper left")
    return ax


def plot_score_distribution(result, ax=None, bins: int = 50):
    """Histogram of scores with the threshold marked."""
    ax = _get_ax(ax, figsize=(6, 4))
    scores = np.asarray(result.scores)
    ax.hist(scores, bins=bins, color="tab:blue", alpha=0.75)
    ax.axvline(result.threshold, color="crimson", ls="--", label=f"threshold={result.threshold:.3g}")
    ax.set_xlabel("anomaly score")
    ax.set_ylabel("count")
    ax.set_yscale("log")
    ax.legend()
    return ax
