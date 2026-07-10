import numpy as np
import pytest

from sigmaflow import SignalFrame
from sigmaflow.detectors import (
    CUSUMDetector,
    ModifiedZScoreDetector,
    STLResidualDetector,
    ZScoreDetector,
)
from sigmaflow.detectors.statistical.stl_residual import detect_period, seasonal_decompose
from sigmaflow.detectors.threshold import compute_threshold

ALL_STATISTICAL = [ZScoreDetector, ModifiedZScoreDetector, CUSUMDetector, STLResidualDetector]


@pytest.mark.parametrize("cls", ALL_STATISTICAL)
def test_output_shape_and_validity(cls, simple_signal):
    result = cls().fit_detect(simple_signal)
    assert len(result.labels) == len(simple_signal)
    assert len(result.scores) == len(simple_signal)
    assert np.isfinite(result.scores).all()
    assert set(np.unique(result.labels)).issubset({0, 1})


@pytest.mark.parametrize("cls", [ZScoreDetector, ModifiedZScoreDetector])
def test_spike_detected(cls, simple_signal):
    result = cls().fit_detect(simple_signal)
    assert result.labels[500] == 1
    assert result.labels.sum() < 20  # not everything flagged


def test_zscore_global_stats_from_fit(rng):
    train = SignalFrame(time=np.arange(500.0), values=rng.normal(0, 1, 500))
    test_y = rng.normal(0, 1, 100)
    test_y[50] = 10.0
    test = SignalFrame(time=np.arange(100.0), values=test_y)
    det = ZScoreDetector().fit(train)
    result = det.detect(test)
    assert result.labels[50] == 1


def test_zscore_rolling_window(rng):
    # slow drift: global z-score misses local spike, rolling catches it
    t = np.arange(2000.0)
    y = 0.02 * t + rng.normal(0, 0.5, 2000)
    y[1000] += 5.0
    sf = SignalFrame(time=t, values=y)
    result = ZScoreDetector(window_size=100).fit_detect(sf)
    assert result.labels[1000] == 1


def test_zscore_constant_signal():
    sf = SignalFrame(time=np.arange(100.0), values=np.ones(100))
    result = ZScoreDetector().fit_detect(sf)
    assert result.labels.sum() == 0


def test_modified_zscore_robust_to_contamination(rng):
    # 5% massive outliers: plain std inflates, MAD does not
    y = rng.normal(0, 1, 1000)
    idx = rng.choice(1000, 50, replace=False)
    y[idx] += 30.0
    sf = SignalFrame(time=np.arange(1000.0), values=y)
    result = ModifiedZScoreDetector().fit_detect(sf)
    assert result.labels[idx].mean() > 0.95


def test_modified_zscore_rolling(simple_signal):
    result = ModifiedZScoreDetector(window_size=100).fit_detect(simple_signal)
    assert result.labels[500] == 1


def test_cusum_detects_mean_shift(shifted_signal):
    # proper CUSUM usage: learn the target from a known-normal interval
    det = CUSUMDetector(threshold=8.0, drift=0.5).fit(shifted_signal.slice(0.0, 499.0))
    result = det.detect(shifted_signal)
    flagged = np.nonzero(result.labels)[0]
    assert len(flagged) > 0
    assert 600 <= flagged[0] <= 640  # alarm shortly after onset
    assert result.labels[:600].sum() == 0  # no alarms before the shift


def test_cusum_ignores_small_spike(rng):
    # a single 3-sigma point can't accumulate past the alarm level
    y = rng.normal(0, 1, 1000)
    y[500] += 3.0
    sf = SignalFrame(time=np.arange(1000.0), values=y)
    result = CUSUMDetector(threshold=8.0, drift=0.5).fit_detect(sf)
    assert result.labels[495:505].sum() == 0


def test_cusum_explicit_target(rng):
    y = rng.normal(10.0, 1.0, 500)
    sf = SignalFrame(time=np.arange(500.0), values=y)
    result = CUSUMDetector(target=10.0, threshold=8.0).fit_detect(sf)
    assert result.labels.sum() == 0


def test_stl_detects_offcycle_spike(seasonal_signal):
    result = STLResidualDetector(period=50).fit_detect(seasonal_signal)
    assert result.labels[375] == 1
    # seasonal peaks themselves must not alarm
    assert result.labels.sum() < 20


def test_stl_autodetects_period(seasonal_signal):
    result = STLResidualDetector().fit_detect(seasonal_signal)
    assert result.labels[375] == 1


def test_stl_no_seasonality_fallback(simple_signal):
    result = STLResidualDetector().fit_detect(simple_signal)
    assert len(result.labels) == len(simple_signal)
    assert result.labels[500] == 1


def test_stl_rejects_unknown_method():
    with pytest.raises(ValueError):
        STLResidualDetector(method="loess")


def test_detect_period_finds_cycle():
    t = np.arange(500)
    x = np.sin(2 * np.pi * t / 25)
    assert detect_period(x) == pytest.approx(25, abs=1)


def test_detect_period_degenerate_inputs():
    assert detect_period(np.zeros(100)) is None  # zero variance
    assert detect_period(np.ones(4)) is None     # too short


def test_seasonal_decompose_reconstructs():
    t = np.arange(300)
    x = 0.01 * t + 5 * np.sin(2 * np.pi * t / 30)
    trend, seasonal, residual = seasonal_decompose(x, 30)
    np.testing.assert_allclose(trend + seasonal + residual, x, atol=1e-9)
    assert np.std(residual) < np.std(x)


# ------------------------------------------------------------ thresholding

def test_compute_threshold_methods():
    scores = np.linspace(0, 1, 101)
    assert compute_threshold(scores, "percentile", 90) == pytest.approx(0.9)
    assert compute_threshold(scores, "fixed", 0.7) == 0.7
    sigma_thr = compute_threshold(scores, "sigma", 2)
    assert sigma_thr == pytest.approx(scores.mean() + 2 * scores.std())


def test_compute_threshold_errors():
    scores = np.zeros(10)
    with pytest.raises(ValueError):
        compute_threshold(scores, "fixed")
    with pytest.raises(ValueError):
        compute_threshold(scores, "percentile", 150)
    with pytest.raises(ValueError):
        compute_threshold(scores, "nonsense")


def test_set_threshold_on_detector(simple_signal):
    det = ZScoreDetector().set_threshold("percentile", 99.5)
    result = det.fit_detect(simple_signal)
    assert result.threshold == pytest.approx(np.percentile(result.scores, 99.5))
    with pytest.raises(ValueError):
        det.set_threshold("bogus")


def test_multichannel_max_aggregation(multichannel_signal):
    result = ZScoreDetector().fit_detect(multichannel_signal)
    assert result.labels[250] == 1  # anomaly lives in channel "b"
