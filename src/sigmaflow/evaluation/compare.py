"""Comparative evaluation: run several detectors on one labeled signal."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.base import BaseDetector
from ..core.signal_frame import SignalFrame
from .metrics import evaluate

__all__ = ["compare_detectors", "DetectorComparison"]

_DEFAULT_METRICS = ["f1", "auc_pr", "detection_latency", "event_recall",
                    "range_f1", "fpr"]


class DetectorComparison:
    """Results of a multi-detector comparison on one signal."""

    def __init__(self, signal: SignalFrame, results: dict, metrics: list[str]):
        self.signal = signal
        self.results = results          # name -> AnomalyResult
        self.metric_names = metrics
        self.metrics = {
            name: evaluate(result, signal.anomaly_labels)
            for name, result in results.items()
        }

    def summary_table(self) -> pd.DataFrame:
        rows = {
            name: {m: vals.get(m, float("nan")) for m in self.metric_names}
            for name, vals in self.metrics.items()
        }
        df = pd.DataFrame(rows).T
        df.index.name = "detector"
        return df.round(4)

    def to_markdown(self) -> str:
        return self.summary_table().to_markdown()

    def to_latex(self) -> str:
        return self.summary_table().to_latex()

    def best(self, metric: str = "f1") -> str:
        """Name of the detector with the highest value of ``metric``."""
        return max(self.metrics, key=lambda n: np.nan_to_num(
            self.metrics[n].get(metric, float("nan")), nan=-np.inf))

    # ---------------------------------------------------------------- plots
    def plot_roc(self, ax=None):
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 5))
        truth = self.signal.anomaly_labels
        for name, result in self.results.items():
            fpr, tpr, _ = roc_curve(truth, result.scores)
            auc = self.metrics[name].get("auc_roc", float("nan"))
            ax.plot(fpr, tpr, lw=1.6, label=f"{name} (AUC {auc:.3f})")
        ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1)
        ax.set_xlabel("false positive rate")
        ax.set_ylabel("true positive rate")
        ax.legend(fontsize=9)
        return ax

    def plot_pr(self, ax=None):
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 5))
        truth = self.signal.anomaly_labels
        for name, result in self.results.items():
            precision, recall, _ = precision_recall_curve(truth, result.scores)
            auc_pr = self.metrics[name].get("auc_pr", float("nan"))
            ax.plot(recall, precision, lw=1.6, label=f"{name} (AUC-PR {auc_pr:.3f})")
        ax.set_xlabel("recall")
        ax.set_ylabel("precision")
        ax.legend(fontsize=9)
        return ax

    def plot_detections(self, channel: str | None = None):
        import matplotlib.pyplot as plt

        channel = channel or self.signal.channels[0]
        n = len(self.results)
        fig, axes = plt.subplots(n, 1, figsize=(11, 2.2 * n), sharex=True)
        axes = np.atleast_1d(axes)
        for ax, (name, result) in zip(axes, self.results.items()):
            ax.plot(self.signal.time, self.signal[channel], lw=0.7)
            for start, end, _ in result.intervals:
                ax.axvspan(start, end, color="crimson", alpha=0.25)
            ax.set_ylabel(name, fontsize=9)
        fig.tight_layout()
        return fig

    def __repr__(self) -> str:
        return f"DetectorComparison({list(self.results)}, best_f1={self.best('f1')!r})"


def compare_detectors(
    signal: SignalFrame,
    detectors: list[BaseDetector],
    ground_truth: np.ndarray | None = None,
    metrics: list[str] | None = None,
    fit_on: SignalFrame | None = None,
) -> DetectorComparison:
    """Fit and run every detector on ``signal``, scoring against labels.

    ``ground_truth`` defaults to the signal's own ``anomaly_labels``.
    ``fit_on`` optionally fits detectors on a separate (clean) signal.
    """
    if ground_truth is not None:
        signal.add_labels(ground_truth)
    if signal.anomaly_labels is None:
        raise ValueError("comparison needs ground-truth labels "
                         "(pass ground_truth or use signal.add_labels)")
    results = {}
    for det in detectors:
        det.fit(fit_on if fit_on is not None else signal)
        results[det.name] = det.detect(signal)
    return DetectorComparison(signal, results, metrics or list(_DEFAULT_METRICS))
