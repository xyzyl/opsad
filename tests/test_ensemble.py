import numpy as np
import pytest

from sigmaflow import SignalFrame
from sigmaflow.detectors import (
    CUSUMDetector,
    EnsembleDetector,
    IsolationForestDetector,
    ModifiedZScoreDetector,
    ZScoreDetector,
)
from sigmaflow.evaluation import evaluate


@pytest.fixture
def mixed_anomaly_signal(rng):
    """A spike (point anomaly) AND a mean shift (contextual) in one signal."""
    t = np.arange(2000.0)
    y = rng.normal(0, 1, 2000)
    y[500] = 14.0
    y[1400:] += 2.5
    labels = np.zeros(2000, dtype=int)
    labels[500] = 1
    labels[1400:] = 1
    return SignalFrame(time=t, values=y).add_labels(labels)


def test_mean_ensemble_catches_both_anomaly_types(mixed_anomaly_signal):
    sig = mixed_anomaly_signal
    clean = sig.slice(0.0, 1399.0)  # everything before the mean shift
    # members' scores are standardized to robust sigmas, so a fixed
    # threshold of 3 means "3 sigmas above typical" for the combination
    ensemble = EnsembleDetector(
        detectors=[ZScoreDetector(threshold=4.0), CUSUMDetector(threshold=8.0)],
        aggregation="mean",
    ).set_threshold("fixed", 3.0)
    result = ensemble.fit(clean).detect(sig)
    m = evaluate(result, sig.anomaly_labels)
    assert m["event_recall"] == 1.0  # spike AND shift both caught
    assert m["fpr"] < 0.05


def test_max_aggregation(mixed_anomaly_signal):
    ensemble = EnsembleDetector(
        detectors=[ZScoreDetector(), ModifiedZScoreDetector()],
        aggregation="max",
    )
    result = ensemble.fit_detect(mixed_anomaly_signal)
    assert result.labels[500] == 1


def test_voting(mixed_anomaly_signal):
    ensemble = EnsembleDetector(
        detectors=[ZScoreDetector(threshold=3.0),
                   ModifiedZScoreDetector(threshold=3.5),
                   IsolationForestDetector(contamination=0.01)],
        aggregation="voting",
    )
    result = ensemble.fit_detect(mixed_anomaly_signal)
    # spike is obvious to all three -> unanimous vote
    assert result.scores[500] == 1.0
    assert result.labels[500] == 1
    assert result.threshold == 0.5


def test_weighted(mixed_anomaly_signal):
    ensemble = EnsembleDetector(
        detectors=[ZScoreDetector(), ModifiedZScoreDetector()],
        aggregation="weighted", weights=[2.0, 1.0],
    )
    result = ensemble.fit_detect(mixed_anomaly_signal)
    assert len(result.scores) == len(mixed_anomaly_signal)
    assert np.isfinite(result.scores).all()


def test_validation_errors():
    with pytest.raises(ValueError, match="at least one"):
        EnsembleDetector(detectors=[])
    with pytest.raises(ValueError, match="unknown aggregation"):
        EnsembleDetector(detectors=[ZScoreDetector()], aggregation="quantum")
    with pytest.raises(ValueError, match="one weight per detector"):
        EnsembleDetector(detectors=[ZScoreDetector()], aggregation="weighted")


def test_params_serializable(mixed_anomaly_signal):
    ensemble = EnsembleDetector(detectors=[ZScoreDetector()], aggregation="mean")
    params = ensemble.get_params()
    assert params["detectors"] == ["ZScoreDetector"]
    assert params["aggregation"] == "mean"
