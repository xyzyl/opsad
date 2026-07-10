import numpy as np
import pytest

from sigmaflow import SignalFrame
from sigmaflow.detectors import IsolationForestDetector, LOFDetector
from sigmaflow.detectors.ml._features import FEATURE_NAMES, window_features

ML_DETECTORS = [IsolationForestDetector, LOFDetector]


def test_window_features_shape(rng):
    x = rng.normal(0, 1, 200)
    feats = window_features(x, 25)
    assert feats.shape == (200, len(FEATURE_NAMES))
    assert np.isfinite(feats).all()


def test_window_features_values():
    x = np.arange(10, dtype=float)
    feats = window_features(x, 3)
    # center rows: mean of [i-1, i, i+1] is i, slope is 1
    assert feats[5, 0] == 5.0            # value
    assert feats[5, 1] == pytest.approx(5.0)   # mean
    assert feats[5, 3] == 4.0            # min
    assert feats[5, 4] == 6.0            # max
    assert feats[5, 5] == pytest.approx(1.0)   # slope


def test_window_features_handles_nan():
    x = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
    feats = window_features(x, 3)
    assert np.isfinite(feats).all()


@pytest.mark.parametrize("cls", ML_DETECTORS)
def test_output_shape_and_validity(cls, simple_signal):
    result = cls().fit_detect(simple_signal)
    assert len(result.labels) == len(simple_signal)
    assert np.isfinite(result.scores).all()
    assert set(np.unique(result.labels)).issubset({0, 1})


@pytest.mark.parametrize("cls", ML_DETECTORS)
def test_spike_detected(cls, simple_signal):
    result = cls(contamination=0.01).fit_detect(simple_signal)
    assert result.labels[500] == 1


def test_isolation_forest_contamination_controls_rate(simple_signal):
    lo = IsolationForestDetector(contamination=0.01).fit_detect(simple_signal)
    hi = IsolationForestDetector(contamination=0.10).fit_detect(simple_signal)
    assert hi.labels.sum() > lo.labels.sum()


def test_isolation_forest_reproducible(simple_signal):
    r1 = IsolationForestDetector(random_state=7).fit_detect(simple_signal)
    r2 = IsolationForestDetector(random_state=7).fit_detect(simple_signal)
    np.testing.assert_allclose(r1.scores, r2.scores)


def test_lof_local_anomaly(rng):
    # value normal globally but unusual for its local context
    t = np.arange(1000.0)
    y = np.where(t < 500, rng.normal(0, 0.3, 1000), rng.normal(10, 0.3, 1000))
    y[250] = 5.0  # between the two regimes: globally in-range, locally alien
    sf = SignalFrame(time=t, values=y)
    result = LOFDetector(contamination=0.01).fit_detect(sf)
    assert result.labels[250] == 1


def test_lof_small_signal():
    sf = SignalFrame(time=np.arange(10.0), values=np.random.default_rng(0).normal(0, 1, 10))
    result = LOFDetector(n_neighbors=20).fit_detect(sf)  # n_neighbors clamped
    assert len(result.labels) == 10


def test_detect_after_separate_fit(rng, simple_signal):
    train = SignalFrame(time=np.arange(500.0), values=rng.normal(0, 1, 500))
    det = IsolationForestDetector(contamination=0.01).fit(train)
    result = det.detect(simple_signal)
    assert result.labels[500] == 1


def test_result_metadata(simple_signal):
    result = IsolationForestDetector(n_estimators=50).fit_detect(simple_signal)
    assert result.detector_name == "isolation_forest"
    assert result.parameters["n_estimators"] == 50
    assert result.signal_name == "simple"
    assert result.computation_time > 0
