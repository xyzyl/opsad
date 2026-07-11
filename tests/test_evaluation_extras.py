"""Tests for range metrics, comparative evaluation, streaming, benchmarks."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from sigmaflow import SignalFrame, StreamingDetector
from sigmaflow.detectors import (
    CUSUMDetector,
    IsolationForestDetector,
    ModifiedZScoreDetector,
    ZScoreDetector,
)
from sigmaflow.evaluation import compare_detectors, evaluate, range_metrics


# ---------------------------------------------------------------- range metrics

def test_range_metrics_partial_credit():
    truth = np.zeros(100, dtype=int)
    truth[20:40] = 1                      # one 20-sample anomaly
    predicted = np.zeros(100, dtype=int)
    predicted[30:40] = 1                  # detects the second half only
    m = range_metrics(predicted, truth, alpha=0.5)
    # existence (0.5) + half coverage (0.5 * 0.5) = 0.75
    assert m["range_recall"] == pytest.approx(0.75)
    assert m["range_precision"] == pytest.approx(1.0)  # detection fully inside truth
    # point recall would be 0.5 — range metrics give the partial credit
    assert m["range_recall"] > 0.5


def test_range_metrics_false_alarm():
    truth = np.zeros(100, dtype=int)
    truth[20:30] = 1
    predicted = np.zeros(100, dtype=int)
    predicted[80:90] = 1                  # entirely wrong place
    m = range_metrics(predicted, truth)
    assert m["range_recall"] == 0.0
    assert m["range_precision"] == 0.0
    assert m["range_f1"] == 0.0


def test_range_metrics_perfect_and_empty():
    truth = np.zeros(50, dtype=int)
    truth[10:20] = 1
    m = range_metrics(truth, truth, alpha=0.5)
    assert m["range_f1"] == pytest.approx(1.0)
    m2 = range_metrics(np.zeros(50), np.zeros(50))
    assert m2["range_f1"] == 0.0


def test_range_metrics_alpha_validation():
    with pytest.raises(ValueError):
        range_metrics(np.zeros(5), np.zeros(5), alpha=2.0)


def test_evaluate_includes_range_metrics(simple_signal):
    truth = np.zeros(len(simple_signal), dtype=int)
    truth[500] = 1
    result = ZScoreDetector().fit_detect(simple_signal)
    m = evaluate(result, truth)
    assert "range_f1" in m and "range_recall" in m


# ---------------------------------------------------------------- comparison

@pytest.fixture
def labeled_signal(rng):
    t = np.arange(1500.0)
    y = rng.normal(0, 1, 1500)
    y[700] = 13.0
    labels = np.zeros(1500, dtype=int)
    labels[700] = 1
    return SignalFrame(time=t, values=y, name="labeled").add_labels(labels)


def test_compare_detectors(labeled_signal):
    comparison = compare_detectors(
        signal=labeled_signal,
        detectors=[ZScoreDetector(), ModifiedZScoreDetector(),
                   IsolationForestDetector(contamination=0.01)],
    )
    table = comparison.summary_table()
    assert set(table.index) == {"zscore", "modified_zscore", "isolation_forest"}
    assert "f1" in table.columns
    assert comparison.best("f1") in table.index
    md = comparison.to_markdown()
    assert "zscore" in md
    tex = comparison.to_latex()
    assert "tabular" in tex


def test_compare_requires_labels(rng):
    sf = SignalFrame(time=np.arange(100.0), values=rng.normal(0, 1, 100))
    with pytest.raises(ValueError, match="ground-truth"):
        compare_detectors(sf, [ZScoreDetector()])


def test_compare_plots(labeled_signal):
    comparison = compare_detectors(labeled_signal, [ZScoreDetector(),
                                                    ModifiedZScoreDetector()])
    ax = comparison.plot_roc()
    assert ax.get_xlabel() == "false positive rate"
    ax2 = comparison.plot_pr()
    assert ax2.get_ylabel() == "precision"
    fig = comparison.plot_detections()
    assert len(fig.axes) == 2
    plt.close("all")


# ---------------------------------------------------------------- streaming

def test_streaming_detects_in_new_chunk(rng):
    history = SignalFrame(time=np.arange(1000.0), values=rng.normal(0, 1, 1000))
    detector = ZScoreDetector(threshold=4.0).fit(history)
    stream = StreamingDetector(detector, context_size=1000)

    # prime with normal data, then feed a chunk containing a spike
    r1 = stream.update(np.arange(1000.0, 1100.0), rng.normal(0, 1, 100))
    assert len(r1.labels) == 100
    assert r1.labels.sum() == 0

    chunk = rng.normal(0, 1, 100)
    chunk[50] = 12.0
    r2 = stream.update(np.arange(1100.0, 1200.0), chunk)
    assert len(r2.labels) == 100
    assert r2.labels[50] == 1
    assert stream.n_seen == 200


def test_streaming_context_trim(rng):
    detector = ZScoreDetector().fit(
        SignalFrame(time=np.arange(100.0), values=rng.normal(0, 1, 100)))
    stream = StreamingDetector(detector, context_size=50)
    for k in range(5):
        stream.update(np.arange(k * 30.0, (k + 1) * 30.0), rng.normal(0, 1, 30))
    assert len(stream._time) == 50  # trimmed to context


def test_streaming_channel_mismatch(rng):
    detector = ZScoreDetector()
    stream = StreamingDetector(detector, context_size=100)
    stream.update(np.arange(20.0), {"a": rng.normal(0, 1, 20)})
    with pytest.raises(ValueError, match="differ"):
        stream.update(np.arange(20.0, 40.0), {"b": rng.normal(0, 1, 20)})


def test_streaming_validation():
    with pytest.raises(ValueError):
        StreamingDetector(ZScoreDetector(), context_size=5)


def test_streaming_with_cusum_drift(rng):
    # the canonical online use: watch for drift as data arrives
    history = SignalFrame(time=np.arange(500.0), values=rng.normal(0, 1, 500))
    detector = CUSUMDetector(threshold=8.0).fit(history)
    stream = StreamingDetector(detector, context_size=600)
    alarms = []
    for k in range(6):
        values = rng.normal(0 if k < 3 else 2.5, 1, 100)  # drift begins chunk 3
        result = stream.update(np.arange(500 + k * 100.0, 600 + k * 100.0), values)
        alarms.append(result.labels.sum() > 0)
    assert not any(alarms[:3])
    assert any(alarms[3:])


# ---------------------------------------------------------------- benchmarks

def test_yahoo_instructions():
    from sigmaflow.evaluation import load_yahoo_s5

    with pytest.raises(RuntimeError, match="license"):
        load_yahoo_s5()


def test_nab_parsing_from_cache(tmp_path, monkeypatch):
    """Exercise load_nab fully offline via a pre-seeded cache."""
    import json as _json

    from sigmaflow.evaluation import load_nab

    series = "realKnownCause/nyc_taxi.csv"
    csv = tmp_path / series.replace("/", "__")
    csv.write_text(
        "timestamp,value\n" + "\n".join(
            f"2015-01-01 {h:02d}:00:00,{100 + h}" for h in range(24))
    )
    (tmp_path / "nab_combined_windows.json").write_text(_json.dumps({
        series: [["2015-01-01 05:00:00", "2015-01-01 07:00:00"]],
    }))
    sf = load_nab(series, data_dir=str(tmp_path))
    assert len(sf) == 24
    assert sf.anomaly_labels.sum() == 3  # hours 5, 6, 7
    assert "Numenta" in sf.metadata["source"]
