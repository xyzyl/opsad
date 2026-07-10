import numpy as np
import pytest

from sigmaflow.detectors import ZScoreDetector
from sigmaflow.evaluation import evaluate, event_metrics, point_metrics, score_metrics


def test_point_metrics_hand_computed():
    truth =     np.array([0, 0, 1, 1, 0, 0, 1, 0])
    predicted = np.array([0, 1, 1, 0, 0, 0, 1, 0])
    m = point_metrics(predicted, truth)
    # tp=2 fp=1 fn=1 tn=4
    assert m["tp"] == 2 and m["fp"] == 1 and m["fn"] == 1 and m["tn"] == 4
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["fpr"] == pytest.approx(1 / 5)
    assert m["fnr"] == pytest.approx(1 / 3)
    expected_mcc = (2 * 4 - 1 * 1) / np.sqrt(3 * 3 * 5 * 5)
    assert m["mcc"] == pytest.approx(expected_mcc)


def test_point_metrics_perfect():
    truth = np.array([0, 1, 1, 0])
    m = point_metrics(truth, truth)
    assert m["f1"] == 1.0 and m["mcc"] == 1.0 and m["fpr"] == 0.0


def test_point_metrics_no_positives():
    m = point_metrics(np.zeros(10), np.zeros(10))
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1"] == 0.0


def test_point_metrics_length_mismatch():
    with pytest.raises(ValueError):
        point_metrics(np.zeros(5), np.zeros(6))


def test_event_metrics_hand_computed():
    #                 event A (2-4)     event B (8-9)
    truth =     np.array([0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0])
    # detection: one overlapping A (late by 1), one false alarm at 6, B missed
    predicted = np.array([0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0])
    m = event_metrics(predicted, truth)
    assert m["n_true_events"] == 2
    assert m["n_detected_events"] == 2
    assert m["event_recall"] == pytest.approx(0.5)     # only A found
    assert m["event_precision"] == pytest.approx(0.5)  # 1 of 2 detections real
    assert m["detection_latency"] == pytest.approx(1.0)  # samples
    assert m["over_detection_ratio"] == pytest.approx(1.0)


def test_event_metrics_fragmentation():
    truth =     np.array([0, 1, 1, 1, 1, 1, 1, 0])
    predicted = np.array([0, 1, 0, 1, 0, 1, 0, 0])  # 3 fragments for 1 event
    m = event_metrics(predicted, truth)
    assert m["over_detection_ratio"] == pytest.approx(3.0)
    assert m["event_recall"] == 1.0


def test_event_metrics_latency_in_seconds():
    truth =     np.array([0, 1, 1, 1, 0])
    predicted = np.array([0, 0, 0, 1, 0])
    secs = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    m = event_metrics(predicted, truth, time_seconds=secs)
    assert m["detection_latency"] == pytest.approx(20.0)


def test_event_metrics_empty():
    m = event_metrics(np.zeros(10), np.zeros(10))
    assert m["event_precision"] == 0.0 and m["event_recall"] == 0.0
    assert np.isnan(m["detection_latency"])


def test_score_metrics_separable():
    truth = np.array([0] * 90 + [1] * 10)
    scores = np.concatenate([np.zeros(90), np.ones(10)])
    m = score_metrics(scores, truth)
    assert m["auc_roc"] == 1.0
    assert m["auc_pr"] == pytest.approx(1.0)
    assert m["best_f1"] == pytest.approx(1.0)


def test_score_metrics_single_class():
    m = score_metrics(np.random.default_rng(0).random(50), np.zeros(50))
    assert np.isnan(m["auc_roc"]) and np.isnan(m["auc_pr"])


def test_evaluate_with_result(simple_signal):
    truth = np.zeros(len(simple_signal), dtype=int)
    truth[500] = 1
    result = ZScoreDetector().fit_detect(simple_signal)
    m = evaluate(result, truth)
    assert m["recall"] == 1.0
    assert m["auc_roc"] > 0.99
    assert "detection_latency" in m


def test_evaluate_with_plain_labels():
    truth = np.array([0, 1, 1, 0])
    predicted = np.array([0, 1, 0, 0])
    m = evaluate(predicted, truth)
    assert m["recall"] == pytest.approx(0.5)
    assert "auc_roc" not in m  # no scores available
