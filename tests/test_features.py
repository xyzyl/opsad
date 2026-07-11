import numpy as np
import pytest

from sigmaflow import SignalFrame
from sigmaflow.detectors import IsolationForestDetector
from sigmaflow.features import AVAILABLE_FEATURES, DEFAULT_FEATURES, FeatureExtractor


def test_default_features_shape(rng):
    x = rng.normal(0, 1, 300)
    X = FeatureExtractor(window_size=25).transform(x)
    assert X.shape == (300, len(DEFAULT_FEATURES))
    assert np.isfinite(X).all()


def test_all_features_computable(rng):
    x = rng.normal(0, 1, 200)
    extractor = FeatureExtractor(features=list(AVAILABLE_FEATURES), window_size=21)
    X = extractor.transform(x)
    assert X.shape == (200, len(AVAILABLE_FEATURES))
    assert np.isfinite(X).all()


def test_hand_computed_values():
    x = np.arange(10, dtype=float)
    X = FeatureExtractor(features=["value", "mean", "min", "max", "slope"],
                         window_size=3).transform(x)
    assert X[5, 0] == 5.0                      # value
    assert X[5, 1] == pytest.approx(5.0)       # mean of [4,5,6]
    assert X[5, 2] == 4.0 and X[5, 3] == 6.0   # min/max
    assert X[5, 4] == pytest.approx(1.0)       # slope


def test_spectral_features_distinguish_frequencies():
    t = np.arange(600)
    slow = np.sin(2 * np.pi * t / 60)
    fast = np.sin(2 * np.pi * t / 6)
    ex = FeatureExtractor(features=["dominant_frequency"], window_size=61)
    f_slow = np.median(ex.transform(slow))
    f_fast = np.median(ex.transform(fast))
    assert f_fast > f_slow


def test_autocorrelation_feature(rng):
    smooth = np.cumsum(rng.normal(0, 1, 500))   # strongly autocorrelated
    noise = rng.normal(0, 1, 500)
    ex = FeatureExtractor(features=["autocorrelation_lag1"], window_size=51)
    assert np.median(ex.transform(smooth)) > np.median(ex.transform(noise))


def test_unknown_feature_raises():
    with pytest.raises(ValueError, match="unknown features"):
        FeatureExtractor(features=["telepathy"])


def test_nan_input_stays_finite():
    x = np.array([1.0, np.nan, 3.0] * 20)
    X = FeatureExtractor(window_size=5).transform(x)
    assert np.isfinite(X).all()


def test_detector_with_custom_features(simple_signal):
    det = IsolationForestDetector(
        contamination=0.01,
        features=["mean", "std", "range", "spectral_entropy"],
    )
    result = det.fit_detect(simple_signal)
    # window features (no raw value) flag the spike's neighborhood
    assert result.labels[490:515].sum() >= 5
    assert result.parameters["features"] == ["mean", "std", "range", "spectral_entropy"]


def test_multivariate_mode_catches_relationship_break(rng):
    # two channels normally in lockstep; the relationship breaks at 400-420
    t = np.arange(1000.0)
    a = np.sin(t / 20) + rng.normal(0, 0.05, 1000)
    b = a + rng.normal(0, 0.05, 1000)
    b[400:420] = -a[400:420]  # anti-correlated stretch; each channel alone looks normal
    sf = SignalFrame(time=t, values={"a": a, "b": b})
    det = IsolationForestDetector(contamination=0.02, multivariate=True)
    result = det.fit_detect(sf)
    assert result.labels[400:420].sum() >= 10
