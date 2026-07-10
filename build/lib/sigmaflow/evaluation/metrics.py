"""Detection quality metrics: point-based, event-based, threshold-independent."""

from __future__ import annotations

import numpy as np

__all__ = ["evaluate", "point_metrics", "event_metrics", "score_metrics"]


def _runs(labels: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous runs of 1s as (start, end) inclusive index pairs."""
    labels = np.asarray(labels).astype(int)
    runs = []
    i, n = 0, len(labels)
    while i < n:
        if labels[i] == 1:
            j = i
            while j + 1 < n and labels[j + 1] == 1:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return runs


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def point_metrics(predicted: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Per-timestep precision/recall/F1, FPR/FNR, and MCC."""
    predicted = np.asarray(predicted).astype(int)
    truth = np.asarray(truth).astype(int)
    if len(predicted) != len(truth):
        raise ValueError("predicted and truth labels must have the same length")

    tp = int(np.sum((predicted == 1) & (truth == 1)))
    fp = int(np.sum((predicted == 1) & (truth == 0)))
    fn = int(np.sum((predicted == 0) & (truth == 1)))
    tn = int(np.sum((predicted == 0) & (truth == 0)))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    fnr = fn / (fn + tp) if fn + tp else 0.0

    mcc_denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / mcc_denom if mcc_denom else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "fnr": fnr,
        "mcc": float(mcc),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def event_metrics(
    predicted: np.ndarray, truth: np.ndarray, time_seconds: np.ndarray | None = None
) -> dict[str, float]:
    """Per-anomaly-interval metrics.

    - ``event_precision``: fraction of detected events overlapping a true anomaly.
    - ``event_recall``: fraction of true anomalies with at least one detection.
    - ``detection_latency``: mean delay from anomaly onset to first detection
      (seconds when ``time_seconds`` is given, otherwise samples).
    - ``over_detection_ratio``: detected events per detected true anomaly
      (1.0 = no fragmentation).
    """
    pred_events = _runs(np.asarray(predicted))
    true_events = _runs(np.asarray(truth))

    matched_pred = sum(1 for p in pred_events if any(_overlaps(p, t) for t in true_events))
    event_precision = matched_pred / len(pred_events) if pred_events else 0.0

    detected_true = [t for t in true_events if any(_overlaps(p, t) for p in pred_events)]
    event_recall = len(detected_true) / len(true_events) if true_events else 0.0

    latencies = []
    for t in detected_true:
        first_hits = [max(p[0], t[0]) for p in pred_events if _overlaps(p, t)]
        first = min(first_hits)
        if time_seconds is not None:
            latencies.append(float(time_seconds[first] - time_seconds[t[0]]))
        else:
            latencies.append(float(first - t[0]))
    detection_latency = float(np.mean(latencies)) if latencies else float("nan")

    over_detection = matched_pred / len(detected_true) if detected_true else float("nan")

    return {
        "event_precision": event_precision,
        "event_recall": event_recall,
        "detection_latency": detection_latency,
        "over_detection_ratio": over_detection,
        "n_true_events": len(true_events),
        "n_detected_events": len(pred_events),
    }


def score_metrics(scores: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Threshold-independent metrics from raw scores: AUC-ROC, AUC-PR, best-F1."""
    from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

    scores = np.asarray(scores, dtype=np.float64)
    truth = np.asarray(truth).astype(int)
    if len(np.unique(truth)) < 2:
        return {
            "auc_roc": float("nan"),
            "auc_pr": float("nan"),
            "best_f1": float("nan"),
            "best_f1_threshold": float("nan"),
        }

    auc_roc = float(roc_auc_score(truth, scores))
    precision, recall, thresholds = precision_recall_curve(truth, scores)
    auc_pr = float(auc(recall, precision))

    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = 2 * precision * recall / (precision + recall)
    f1 = np.nan_to_num(f1[:-1])  # last point has no threshold
    best_idx = int(np.argmax(f1)) if len(f1) else 0
    best_f1 = float(f1[best_idx]) if len(f1) else 0.0
    best_thr = float(thresholds[best_idx]) if len(thresholds) else float("nan")

    return {
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "best_f1": best_f1,
        "best_f1_threshold": best_thr,
    }


def evaluate(result, ground_truth: np.ndarray) -> dict[str, float]:
    """Evaluate a detection result against ground-truth labels.

    ``result`` may be an :class:`AnomalyResult` (scores and timing are
    used automatically) or a plain 0/1 label array.
    """
    from ..core.anomaly_result import AnomalyResult
    from ..core.signal_frame import time_to_seconds

    truth = np.asarray(ground_truth).astype(int)

    if isinstance(result, AnomalyResult):
        predicted = result.labels
        scores = result.scores
        time_seconds = time_to_seconds(result._time)
    else:
        predicted = np.asarray(result).astype(int)
        scores = None
        time_seconds = None

    metrics = {}
    metrics.update(point_metrics(predicted, truth))
    metrics.update(event_metrics(predicted, truth, time_seconds))
    if scores is not None:
        metrics.update(score_metrics(scores, truth))
    return metrics
