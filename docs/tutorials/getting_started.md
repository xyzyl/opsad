# Getting Started with sigmaflow

This tutorial walks from a bare array to an evaluated, reproducible detection pipeline.

## 1. Wrap your data in a SignalFrame

A `SignalFrame` is a time-series plus optional scientific metadata. Only `time` and `values` are required:

```python
import numpy as np
import sigmaflow as sf

t = np.arange(0, 600, 0.5)                 # seconds
y = np.sin(t / 30) + np.random.normal(0, 0.1, len(t))
y[700] += 3.0                              # an anomaly

signal = sf.SignalFrame(time=t, values=y)
signal.summary()
```

Add metadata when you have it — it never gets in the way and improves reporting:

```python
signal = sf.SignalFrame(
    time=t,
    values={"temperature": y},
    name="buoy_42",
    units={"temperature": "°C"},
    instrument="moored_buoy",
    domain="ocean",
    metadata={"expected_range": {"temperature": (-2, 35)}},
)
```

`SignalFrame` supports time slicing (`signal.slice(0, 300)`), gap inspection
(`signal.gap_report()`), interpolation, resampling, and HDF5 round-trips
(`signal.to_hdf5(path)` / `sf.SignalFrame.from_hdf5(path)`).

## 2. Detect in one line

```python
result = sf.detect(signal, method="modified_zscore")
result.summary()
result.plot()          # signal with anomaly intervals shaded
result.plot_scores()   # scores with the threshold line
```

Every detector returns an `AnomalyResult` with per-timestep `scores` (higher =
more anomalous), binary `labels`, merged `(start, end, severity)` `intervals`,
and the `threshold` used.

## 3. Choose a detector

| Detector | Best for | Watch out |
|---|---|---|
| `zscore` | Fast baseline, stationary Gaussian noise | Fails on trends/seasonality |
| `modified_zscore` | Same, but robust when outliers contaminate the stats | — |
| `cusum` | Persistent mean shifts, drift, degradation | Fit it on known-normal data |
| `stl_residual` | Seasonal signals (diurnal load, tides) | Needs ≥ 2 full cycles |
| `isolation_forest` | General purpose, no distribution assumptions | Tune `contamination` |
| `lof` | Locally unusual values that are globally in range | Slower on long signals |

All detectors share one interface:

```python
from sigmaflow.detectors import CUSUMDetector

detector = CUSUMDetector(threshold=8.0, drift=0.5)
detector.fit(signal.slice(0, 200))     # learn "normal" from a clean interval
result = detector.detect(signal)
```

Thresholding is controllable on any detector:

```python
detector.set_threshold("percentile", 99.5)   # or "sigma", "fixed", "auto"
```

## 4. Compose a pipeline

```python
from sigmaflow import Pipeline
from sigmaflow.preprocess import GapHandler, Detrend, Normalizer
from sigmaflow.detectors import IsolationForestDetector

pipeline = Pipeline([
    GapHandler(max_gap="5s", fill_method="interpolate"),
    Detrend(method="linear"),
    Normalizer(method="robust"),
    IsolationForestDetector(contamination=0.01, n_estimators=200),
])
result = pipeline.fit_detect(signal)
```

Pipelines serialize to YAML so an analysis can be shared and reproduced exactly:

```python
pipeline.save("my_pipeline.yaml")
same = Pipeline.load("my_pipeline.yaml")
```

## 5. Evaluate honestly

If you have ground-truth labels (or use the synthetic generators, which attach
them), measure quality — don't guess:

```python
from sigmaflow.evaluation import evaluate
from sigmaflow.synthetic import generate_ocean_temperature

signal = generate_ocean_temperature(duration_days=90)
result = pipeline.fit_detect(signal)
metrics = evaluate(result, signal.anomaly_labels)

print(metrics["f1"], metrics["event_recall"], metrics["detection_latency"])
```

Point metrics (precision/recall/F1/MCC) score every timestep; event metrics
score whole anomaly episodes — `event_recall` tells you how many real events
you caught, `detection_latency` how fast, `over_detection_ratio` how fragmented
the detections are. `auc_roc`/`auc_pr`/`best_f1` are threshold-independent.

## 6. Use the CLI

```bash
sigmaflow detect data.csv --detector isolation_forest -p contamination=0.01 -o out.csv
sigmaflow evaluate data.csv --labels labels.csv --detector stl_residual
```

CSV input needs a `time` column (or pass `--time-column`); HDF5 files written
by `to_hdf5` load with all metadata intact.

## 7. Explore interactively

With the `dashboard` extra installed, a web dashboard gives you the whole loop —
browse signals, switch detectors, tune parameters with live re-scoring, drag the
threshold, and click anomalies to zoom:

```bash
pip install sigmaflow[dashboard]
sigmaflow dashboard                # demo signals built in
sigmaflow dashboard my_data.csv    # or bring your own
```

```python
from sigmaflow.dashboard import launch_dashboard
launch_dashboard(signal=my_signal, port=8050)
```

## Next steps

- Run [examples/plasma_disruption_detection.py](../../examples/plasma_disruption_detection.py)
  and [examples/ocean_sensor_qc.py](../../examples/ocean_sensor_qc.py).
- v0.2.0 will add domain adapters (plasma/ocean/satellite/energy), deep
  detectors, and ensembles.
