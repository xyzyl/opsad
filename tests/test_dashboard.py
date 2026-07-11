import numpy as np
import pytest
from click.testing import CliRunner

from sigmaflow.cli.main import cli
from sigmaflow.detectors import IsolationForestDetector, ZScoreDetector
from sigmaflow.viz.dashboard import (
    DETECTOR_PARAMS,
    build_detector,
    build_narrative,
    characterize_interval,
    compute_scores,
    demo_signals,
    intervals_records,
    metric_tiles,
    resolve_threshold,
)

dash = pytest.importorskip("dash", reason="dashboard extra not installed")


def test_detector_params_cover_registry():
    from sigmaflow.detectors import DETECTOR_REGISTRY

    assert set(DETECTOR_PARAMS) == set(DETECTOR_REGISTRY)


def test_build_detector_defaults_and_overrides():
    det = build_detector("zscore", {})
    assert isinstance(det, ZScoreDetector)
    assert det.threshold == 3.0
    det = build_detector("zscore", {"threshold": 2.5, "window_size": 50})
    assert det.threshold == 2.5 and det.window_size == 50


def test_build_detector_blank_contamination_is_auto():
    det = build_detector("isolation_forest", {"contamination": None})
    assert isinstance(det, IsolationForestDetector)
    assert det.contamination == "auto"


def test_build_detector_unknown():
    with pytest.raises(ValueError):
        build_detector("psychic", {})


def test_compute_scores(simple_signal):
    scores, auto_thr = compute_scores(simple_signal, "zscore", {"threshold": 3.0})
    assert len(scores) == len(simple_signal)
    assert np.isfinite(scores).all()
    assert auto_thr == 3.0


def test_resolve_threshold():
    scores = np.linspace(0, 1, 101)
    assert resolve_threshold(scores, "auto", 5.0, 2.0) == 2.0
    assert resolve_threshold(scores, "fixed", 0.5, 2.0) == 0.5
    assert resolve_threshold(scores, "fixed", None, 2.0) == 2.0  # fallback
    assert resolve_threshold(scores, "percentile", 90, 2.0) == pytest.approx(0.9)
    assert resolve_threshold(scores, "percentile", None, 2.0) == pytest.approx(
        np.percentile(scores, 99)
    )


def test_intervals_records(simple_signal):
    scores, _ = compute_scores(simple_signal, "zscore", {})
    labels = (scores > 3.0).astype(int)
    records = intervals_records(simple_signal, labels, scores)
    assert any(r["start"] == "500.0" for r in records)
    assert all({"#", "start", "end", "duration (s)", "peak score"} <= set(r) for r in records)


def test_metric_tiles_with_truth(simple_signal):
    truth = np.zeros(len(simple_signal), dtype=int)
    truth[500] = 1
    simple_signal.add_labels(truth)
    scores, _ = compute_scores(simple_signal, "zscore", {})
    labels = (scores > 3.0).astype(int)
    tiles = dict(metric_tiles(simple_signal, labels, scores))
    assert "anomalies" in tiles and "F1 vs truth" in tiles


def test_metric_tiles_without_truth(simple_signal):
    simple_signal.anomaly_labels = None
    scores, _ = compute_scores(simple_signal, "zscore", {})
    tiles = dict(metric_tiles(simple_signal, (scores > 3).astype(int), scores))
    assert "F1 vs truth" not in tiles


def test_demo_signals_have_labels():
    demos = demo_signals()
    assert len(demos) == 3
    for sf in demos.values():
        assert sf.anomaly_labels is not None


def test_figures_build(simple_signal):
    from sigmaflow.viz.dashboard import histogram_figure, signal_score_figure

    scores, _ = compute_scores(simple_signal, "zscore", {})
    labels = (scores > 3.0).astype(int)
    fig = signal_score_figure(simple_signal, "value", scores, 3.0, labels)
    assert len(fig.data) == 3  # signal + anomaly markers + score traces
    quiet = signal_score_figure(simple_signal, "value", scores, 99.0,
                                np.zeros_like(labels))
    assert len(quiet.data) == 2  # no markers when nothing is flagged
    hist = histogram_figure(scores, 3.0)
    assert len(hist.data) == 1


def test_create_app_layout(simple_signal):
    from sigmaflow.viz.dashboard import create_app

    app = create_app({"my signal": simple_signal}, live=False)
    layout = str(app.layout)
    for component_id in ("signal-select", "detector-select", "score-store",
                        "main-graph", "hist-graph", "anomaly-table", "run-button"):
        assert component_id in layout


def test_characterize_interval_shapes(rng):
    x = rng.normal(0, 1, 1000)
    x[500] = 15.0
    assert characterize_interval(x, 500, 500) == "spike"
    x2 = rng.normal(0, 1, 1000)
    x2[300:340] = 2.5  # constant stretch
    assert characterize_interval(x2, 300, 339) == "flat"
    x3 = rng.normal(0, 1, 1000)
    x3[600:700] += 8.0
    assert characterize_interval(x3, 600, 699) == "shift"
    x4 = rng.normal(0, 1, 1000)
    x4[100:150] += rng.normal(0, 15, 50)
    assert characterize_interval(x4, 100, 149) == "noisy"


def test_narrative_plasma_domain():
    sf = demo_signals()["plasma: synthetic Langmuir probe"]
    scores, _ = compute_scores(sf, "zscore", {})
    thr = float(np.percentile(scores, 99.5))  # guarantee some detections
    labels = (scores > thr).astype(int)
    paragraphs = build_narrative(sf, "n_e", "zscore", scores, thr, labels)
    text = " ".join(paragraphs)
    assert "electron density" in text          # channel explained
    assert "langmuir probe" in text            # instrument named
    assert "alarm bar" in text                 # threshold explained
    assert "answer key" in text                # ground-truth scoreboard
    # domain-specific meaning for whatever shape the strongest event took
    assert any(w in text for w in ("disruption", "saturat", "probe", "RF"))


def test_narrative_ocean_domain_and_cross_domain():
    sf = demo_signals()["ocean: synthetic buoy temperature"]
    scores, _ = compute_scores(sf, "stl_residual", {"period": 24})
    labels = (scores > 4.0).astype(int)
    assert labels.sum() > 0  # the demo heatwave must be detectable
    paragraphs = build_narrative(sf, "temperature", "stl_residual", scores, 4.0, labels)
    text = " ".join(paragraphs)
    assert "water temperature" in text
    assert "rhythm" in text                    # detector explainer
    # cross-domain sentence references at least one other field
    assert any(w in text for w in ("tokamak", "satellite", "grid", "plasma"))


def test_narrative_no_detections(simple_signal):
    simple_signal.anomaly_labels = None
    scores = np.zeros(len(simple_signal))
    labels = np.zeros(len(simple_signal), dtype=int)
    paragraphs = build_narrative(simple_signal, "value", "zscore", scores, 3.0, labels)
    text = " ".join(paragraphs)
    assert "Nothing currently clears the alarm bar" in text
    assert "answer key" not in text            # no ground truth available


def test_narrative_unknown_channel_and_domain(rng):
    from sigmaflow import SignalFrame

    sf = SignalFrame(time=np.arange(500.0), values={"flux_capacitance": rng.normal(0, 1, 500)})
    scores = np.abs(rng.normal(0, 1, 500))
    labels = (scores > 2.5).astype(int)
    paragraphs = build_narrative(sf, "flux_capacitance", "lof", scores, 2.5, labels)
    text = " ".join(paragraphs)
    assert "flux_capacitance" in text
    assert "general instrumentation" in text or "Nothing currently" in text


def test_dashboard_import_path():
    from sigmaflow.dashboard import create_app, launch_dashboard  # noqa: F401


def test_cli_dashboard_help():
    result = CliRunner().invoke(cli, ["dashboard", "--help"])
    assert result.exit_code == 0
    assert "interactive dashboard" in result.output
