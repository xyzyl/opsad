import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from sigmaflow import detect
from sigmaflow.detectors import ModifiedZScoreDetector
from sigmaflow.synthetic import (
    generate_generic_signal,
    generate_ocean_temperature,
    generate_plasma_signal,
)
from sigmaflow.viz import plot_signal


def test_plasma_signal_defaults():
    sf = generate_plasma_signal(duration=1.0, sample_rate=5000)
    assert sf.domain == "plasma"
    assert sf.channels == ["n_e"]
    assert sf.units["n_e"] == "m^-3"
    assert sf.anomaly_labels is not None
    assert 0 < sf.anomaly_labels.sum() < len(sf)


def test_plasma_signal_custom_anomalies():
    sf = generate_plasma_signal(
        duration=1.0,
        sample_rate=5000,
        anomalies=[{"type": "spike", "time": 0.5, "magnitude": 1.0}],
    )
    idx = int(0.5 * 5000)
    assert sf.anomaly_labels[idx] == 1
    assert sf.anomaly_labels.sum() == 1


def test_plasma_saturation_is_flat():
    sf = generate_plasma_signal(
        duration=1.0, sample_rate=5000,
        anomalies=[{"type": "sensor_saturation", "start": 0.4, "end": 0.6}],
    )
    x = sf["n_e"].to_numpy()
    region = x[int(0.45 * 5000): int(0.55 * 5000)]
    assert np.ptp(region) == 0.0


def test_plasma_unknown_anomaly():
    with pytest.raises(ValueError):
        generate_plasma_signal(anomalies=[{"type": "gremlin"}])


def test_ocean_signal_defaults():
    sf = generate_ocean_temperature(duration_days=30)
    assert sf.domain == "ocean"
    assert isinstance(sf.time, pd.DatetimeIndex)
    assert sf.units["temperature"] == "°C"
    assert sf.anomaly_labels is not None


def test_ocean_heatwave_raises_temperature():
    quiet = generate_ocean_temperature(duration_days=30, anomalies=[])
    hot = generate_ocean_temperature(
        duration_days=30,
        anomalies=[{"type": "marine_heatwave", "start_day": 10, "end_day": 15, "magnitude": 4.0}],
    )
    mask = hot.anomaly_labels.astype(bool)
    assert mask.sum() > 0
    diff = hot["temperature"].to_numpy()[mask] - quiet["temperature"].to_numpy()[mask]
    assert diff.mean() > 1.0


def test_generic_signal_reproducible():
    a = generate_generic_signal(n=500, seed=1)
    b = generate_generic_signal(n=500, seed=1)
    np.testing.assert_array_equal(a.to_numpy(), b.to_numpy())


def test_generic_signal_anomaly_types():
    sf = generate_generic_signal(
        n=1000,
        anomalies=[
            {"type": "spike", "index": 100},
            {"type": "level_shift", "start": 400, "end": 450},
            {"type": "noise_burst", "start": 700, "end": 720},
        ],
    )
    assert sf.anomaly_labels[100] == 1
    assert sf.anomaly_labels[400:451].all()
    assert sf.anomaly_labels[700:721].all()


def test_detectors_find_injected_anomalies_end_to_end():
    sf = generate_generic_signal(n=2000, anomalies=[{"type": "spike", "index": 1000, "magnitude": 12.0}])
    result = detect(sf, method="modified_zscore")
    assert result.labels[1000] == 1


def test_convenience_detect_unknown_method(simple_signal):
    with pytest.raises(ValueError, match="unknown detection method"):
        detect(simple_signal, method="crystal_ball")


# ------------------------------------------------------------------ viz

def test_plot_signal(simple_signal):
    ax = plot_signal(simple_signal)
    assert ax.get_title() == "simple"
    plt.close("all")


def test_result_plots(simple_signal):
    result = ModifiedZScoreDetector().fit_detect(simple_signal)
    ax1 = result.plot()
    assert "anomalies" in ax1.get_title()
    ax2 = result.plot_scores()
    assert ax2.get_ylabel() == "anomaly score"
    ax3 = result.plot_distribution()
    assert ax3.get_xlabel() == "anomaly score"
    plt.close("all")


def test_result_summary_and_dataframe(simple_signal):
    result = ModifiedZScoreDetector().fit_detect(simple_signal)
    text = result.summary()
    assert "anomalies" in text
    df = result.to_dataframe()
    assert set(df.columns) >= {"value", "score", "label"}
    assert len(df) == len(simple_signal)
    assert "AnomalyResult" in repr(result)
