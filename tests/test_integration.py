"""End-to-end: file -> pipeline -> detection -> evaluation."""

import numpy as np
import pytest

from sigmaflow import Pipeline, SignalFrame
from sigmaflow.detectors import IsolationForestDetector, STLResidualDetector
from sigmaflow.evaluation import evaluate
from sigmaflow.preprocess import Detrend, GapHandler, Normalizer, Resampler
from sigmaflow.synthetic import generate_ocean_temperature, generate_plasma_signal


def test_plasma_full_pipeline(tmp_path):
    sig = generate_plasma_signal(duration=2.0, sample_rate=5000)
    path = str(tmp_path / "plasma.h5")
    sig.to_hdf5(path)
    loaded = SignalFrame.from_hdf5(path)

    pipe = Pipeline([
        GapHandler(max_gap="1s"),
        Detrend(method="moving_average", window=501),
        Normalizer(method="robust"),
        IsolationForestDetector(contamination=0.15, window_size=51),
    ])
    result = pipe.fit_detect(loaded)
    metrics = evaluate(result, loaded.anomaly_labels)
    # both injected anomalies (precursor + saturation) must be found
    assert metrics["event_recall"] == 1.0
    assert metrics["auc_roc"] > 0.8


def test_ocean_seasonal_pipeline():
    sig = generate_ocean_temperature(
        duration_days=90,
        anomalies=[{"type": "marine_heatwave", "start_day": 40, "end_day": 44, "magnitude": 5.0}],
    )
    # STL with the diurnal period removes the daily cycle; the heatwave remains
    result = STLResidualDetector(period=24, residual_threshold=4.0).fit_detect(sig)
    metrics = evaluate(result, sig.anomaly_labels)
    assert metrics["event_recall"] == 1.0
    assert metrics["fpr"] < 0.05


def test_pipeline_yaml_reproducibility(tmp_path):
    sig = generate_plasma_signal(duration=1.0, sample_rate=2000)
    pipe = Pipeline([
        Resampler(target_rate=500),
        Normalizer(method="z_score"),
        IsolationForestDetector(contamination=0.1, random_state=3),
    ])
    path = str(tmp_path / "pipe.yaml")
    pipe.save(path)
    r1 = pipe.fit_detect(sig)
    r2 = Pipeline.load(path).fit_detect(sig)
    np.testing.assert_array_equal(r1.labels, r2.labels)
    assert len(r1.labels) == pytest.approx(500, abs=2)
