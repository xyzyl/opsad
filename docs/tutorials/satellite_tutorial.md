# Satellite tutorial: GOES magnetometer telemetry

The GOES weather satellites carry magnetometers measuring Earth's magnetic
field at geostationary orbit — real spacecraft telemetry, published live by
NOAA. Geomagnetic storms, substorms, and spacecraft events all leave marks
in it.

## 1. Fetch live telemetry

```python
from sigmaflow.data import fetch_goes_magnetometer

signal = fetch_goes_magnetometer()
signal.summary()
# channels: Hp, He, Hn, total [nT] — last 24 hours at 1-minute cadence
```

## 2. Adapter defaults

```python
from sigmaflow.domains import SatelliteAdapter

adapter = SatelliteAdapter()
result = adapter.default_pipeline().fit_detect(signal)
result = adapter.classify_anomaly(result, signal)
result.summary()
```

The satellite preprocessing removes slow orbital variation with a moving
average before detecting, so the daily magnetic swing doesn't alarm.

## 3. Multivariate detection

Magnetometer components move together; a disturbance can be invisible in each
channel but obvious in their *relationship*:

```python
from sigmaflow.detectors import IsolationForestDetector

joint = IsolationForestDetector(contamination=0.01, multivariate=True)
result = joint.fit_detect(signal)
```

## 4. Compare detectors honestly

Attach labels (yours, or from a benchmark) and let the numbers decide:

```python
from sigmaflow.evaluation import compare_detectors
from sigmaflow.detectors import ZScoreDetector, LOFDetector

comparison = compare_detectors(
    labeled_signal,
    detectors=[ZScoreDetector(), LOFDetector(), IsolationForestDetector()],
)
print(comparison.summary_table())
print("best:", comparison.best("range_f1"))
comparison.plot_pr()
```
