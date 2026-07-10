"""Quality-control a synthetic moored-buoy record.

Two anomaly types, two detectors, matched to the physics:
  - a short marine heatwave in temperature -> STLResidualDetector removes
    the diurnal cycle so the excursion stands out instead of every warm
    afternoon;
  - biofouling-style sensor drift in salinity -> CUSUMDetector accumulates
    the slow systematic deviation that point detectors never see. (Salinity
    is the right channel for drift QC: unlike temperature it has no strong
    seasonal cycle to confuse a mean-shift detector.)

Run:  python examples/ocean_sensor_qc.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sigmaflow import SignalFrame
from sigmaflow.detectors import CUSUMDetector, STLResidualDetector
from sigmaflow.evaluation import evaluate
from sigmaflow.synthetic import generate_ocean_temperature


def main():
    # --- Temperature record with a 1.5-day marine heatwave at day 40 ---
    temperature = generate_ocean_temperature(
        duration_days=180,
        samples_per_day=24,
        anomalies=[
            {"type": "marine_heatwave", "start_day": 40, "end_day": 41.5, "magnitude": 4.0},
        ],
    )
    temperature.summary()

    stl = STLResidualDetector(period=24, residual_threshold=4.0)
    stl_result = stl.fit_detect(temperature)
    print("\n-- STL residual on temperature (short events) --")
    stl_result.summary()

    # --- Salinity record: stationary at ~35 PSU until biofouling sets in ---
    rng = np.random.default_rng(1)
    n = len(temperature)
    days = np.arange(n) / 24.0
    sal = 35.0 + rng.normal(0.0, 0.02, n)
    drift_mask = days >= 120.0
    sal[drift_mask] -= 0.005 * (days[drift_mask] - 120.0)  # -0.3 PSU by day 180
    salinity = SignalFrame(
        time=temperature.time,
        values={"salinity": sal},
        name="synthetic_ocean_buoy_salinity",
        units={"salinity": "PSU"},
        instrument="moored_buoy",
        domain="ocean",
    )
    salinity.add_labels(drift_mask.astype(int))

    # Fit the expected mean on the clean first 100 days, then watch the
    # cumulative sum grow as the conductivity cell fouls.
    cusum = CUSUMDetector(threshold=25.0, drift=1.0)
    cusum.fit(salinity.slice(salinity.time[0], salinity.time[24 * 100 - 1]))
    cusum_result = cusum.detect(salinity)
    print("\n-- CUSUM on salinity (drift) --")
    cusum_result.summary()

    print("\nSTL vs temperature ground truth:")
    m1 = evaluate(stl_result, temperature.anomaly_labels)
    for key in ("event_recall", "event_precision", "fpr", "auc_roc"):
        print(f"  {key:16s} {m1[key]:.4f}")

    print("CUSUM vs salinity ground truth:")
    m2 = evaluate(cusum_result, salinity.anomaly_labels)
    for key in ("event_recall", "fpr", "detection_latency"):
        print(f"  {key:16s} {m2[key]:.4f}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    stl_result.plot(ax=ax1)
    ax1.set_title("STL residual: heatwave in temperature")
    cusum_result.plot(ax=ax2)
    ax2.set_title("CUSUM: biofouling drift in salinity")
    fig.tight_layout()
    fig.savefig("ocean_sensor_qc.png", dpi=120)
    print("\nfigure saved to ocean_sensor_qc.png")


if __name__ == "__main__":
    main()
