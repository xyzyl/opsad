"""Detect a disruption precursor and sensor saturation in a synthetic
Langmuir probe signal, evaluate against ground truth, and save a figure.

Run:  python examples/plasma_disruption_detection.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sigmaflow import Pipeline
from sigmaflow.detectors import IsolationForestDetector
from sigmaflow.evaluation import evaluate
from sigmaflow.preprocess import Detrend, GapHandler, Normalizer
from sigmaflow.synthetic import generate_plasma_signal


def main():
    # Synthetic 1 MHz-class diagnostic downscaled for a quick demo:
    # a growing MHD-like oscillation from t=3.2s and saturation at 4.5-4.7s.
    signal = generate_plasma_signal(
        duration=5.0,
        sample_rate=10_000,
        anomalies=[
            {"type": "disruption_precursor", "onset": 3.2, "growth_rate": 1.6},
            {"type": "sensor_saturation", "start": 4.5, "end": 4.7},
        ],
    )
    signal.summary()

    pipeline = Pipeline([
        GapHandler(max_gap="0.1s", fill_method="interpolate"),
        Detrend(method="moving_average", window=2001),
        Normalizer(method="robust"),
        IsolationForestDetector(contamination=0.2, window_size=101),
    ])

    result = pipeline.fit_detect(signal)
    result.summary()

    metrics = evaluate(result, signal.anomaly_labels)
    print("\nevaluation against ground truth:")
    for key in ("f1", "event_recall", "event_precision", "detection_latency", "auc_pr"):
        print(f"  {key:18s} {metrics[key]:.4f}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    result.plot(ax=ax1)
    result.plot_scores(ax=ax2)
    fig.tight_layout()
    fig.savefig("plasma_disruption_detection.png", dpi=120)
    print("\nfigure saved to plasma_disruption_detection.png")


if __name__ == "__main__":
    main()
