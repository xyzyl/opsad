# sigmaflow

**Cross-domain anomaly detection for scientific time-series.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Status](https://img.shields.io/badge/status-alpha-orange)

## Why sigmaflow?

Generic anomaly detection libraries treat a plasma density trace, an ocean buoy temperature record, and a satellite battery voltage as interchangeable arrays of numbers. sigmaflow is built for scientific instrument data: it carries physical units, sample rates, and instrument metadata through the whole pipeline, understands instrument failure modes (sensor saturation, telemetry dropouts, calibration jumps), and gives a fusion researcher and an oceanographer the same API, mental model, and evaluation tools. Metadata always enriches the analysis but is never required — a bare NumPy array works everywhere.

## Install

```bash
pip install sigmaflow
```

## 30-second example

```python
import sigmaflow as sf

signal = sf.SignalFrame(time=t, values=y)          # just arrays — no ceremony
result = sf.detect(signal, method="isolation_forest")
result.plot()
```

## Features

| Category | Included in v0.1.0 |
|---|---|
| Data model | `SignalFrame` (units, sample rate, metadata, HDF5 persistence), `AnomalyResult` |
| Statistical detectors | `ZScoreDetector`, `ModifiedZScoreDetector`, `CUSUMDetector`, `STLResidualDetector` |
| ML detectors | `IsolationForestDetector`, `LOFDetector` (windowed feature extraction) |
| Preprocessing | `Resampler` (anti-aliased), `GapHandler`, `Detrend`, `Normalizer` |
| Pipelines | Composable, YAML-serializable, reproducible |
| Evaluation | Point metrics (F1, MCC), event metrics (latency, fragmentation), AUC-ROC/PR |
| Synthetic benchmarks | Plasma diagnostic and ocean buoy generators with ground-truth labels |
| Dashboard | Interactive web dashboard: signal browser, live detector tuning, threshold control, click-to-zoom anomaly table |
| CLI | `sigmaflow detect`, `sigmaflow evaluate`, `sigmaflow dashboard` |

## A complete pipeline

```python
from sigmaflow import Pipeline
from sigmaflow.preprocess import GapHandler, Detrend, Normalizer
from sigmaflow.detectors import IsolationForestDetector
from sigmaflow.evaluation import evaluate
from sigmaflow.synthetic import generate_plasma_signal

signal = generate_plasma_signal(duration=5.0, sample_rate=10_000)

pipeline = Pipeline([
    GapHandler(max_gap="5s", fill_method="interpolate"),
    Detrend(method="moving_average", window=501),
    Normalizer(method="robust"),
    IsolationForestDetector(contamination=0.15),
])

result = pipeline.fit_detect(signal)
result.summary()
print(evaluate(result, signal.anomaly_labels))

pipeline.save("plasma_pipeline.yaml")          # share and reproduce
```

## Command line

```bash
sigmaflow detect data.csv --detector isolation_forest -p contamination=0.01
sigmaflow detect shot_184520.h5 --pipeline plasma_pipeline.yaml -o results.csv
sigmaflow evaluate data.csv --labels labels.csv --detector zscore
sigmaflow dashboard data.csv          # interactive web dashboard
```

## Interactive dashboard

```bash
pip install sigmaflow[dashboard]
sigmaflow dashboard                    # opens on http://127.0.0.1:8050
sigmaflow dashboard my_signal.h5       # add your own signal to the browser
```

Or from Python:

```python
from sigmaflow.dashboard import launch_dashboard
launch_dashboard(signal=my_signal_frame, detector="isolation_forest", port=8050)
```

The dashboard ships with the synthetic demo signals, so it works out of the box:
browse signals and channels, switch detectors and tune their parameters with live
re-scoring, move the threshold without recomputing, compare detections against
ground-truth bands, and click any row of the anomaly table to zoom to that event.

### Publish it as a website

`sigmaflow dashboard --export site` writes a static version (~1 MB, precomputed
scores; thresholding, metrics, and interpretation run in the browser) that
deploys to Cloudflare Pages in one command:

```bash
npx wrangler pages deploy site --project-name sigmaflow
```

See [docs/deploying.md](docs/deploying.md) for details, including how to host
the fully live app on a Python platform instead.

## Documentation

- [Getting started tutorial](docs/tutorials/getting_started.md)
- Runnable examples: [plasma disruption detection](examples/plasma_disruption_detection.py), [ocean sensor QC](examples/ocean_sensor_qc.py)

## Installation options

```bash
pip install sigmaflow             # core (numpy, scipy, scikit-learn, pandas)
pip install sigmaflow[dashboard]  # + interactive web dashboard
pip install sigmaflow[deep]       # + deep learning detectors (coming in v0.2)
pip install sigmaflow[dev]        # + test/lint tooling
```

## Roadmap

v0.2.0 adds deep learning detectors, ensembles, and domain adapters for plasma, ocean, satellite, and energy-grid instruments; v0.3.0 adds range-based metrics and streaming detection. See [CHANGELOG.md](CHANGELOG.md).

## Citation

```bibtex
@software{sigmaflow,
  title  = {sigmaflow: cross-domain anomaly detection for scientific time-series},
  author = {Iskander},
  year   = {2026},
  url    = {https://github.com/iskander/sigmaflow}
}
```

## License

MIT — see [LICENSE](LICENSE).
