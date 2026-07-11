"""Benchmark dataset loaders (NAB, SMD; Yahoo S5 by instruction).

Datasets download on first use and are cached under
``~/.sigmaflow/benchmarks`` (override with ``data_dir``).
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.signal_frame import SignalFrame

__all__ = ["load_nab", "load_smd", "load_yahoo_s5", "NAB_SERIES"]

_NAB_BASE = "https://raw.githubusercontent.com/numenta/NAB/master"

#: a representative subset of NAB's 58 series (path within NAB/data/)
NAB_SERIES = [
    "realTweets/Twitter_volume_AAPL.csv",
    "realTweets/Twitter_volume_GOOG.csv",
    "realKnownCause/machine_temperature_system_failure.csv",
    "realKnownCause/ambient_temperature_system_failure.csv",
    "realKnownCause/nyc_taxi.csv",
    "realAWSCloudwatch/ec2_cpu_utilization_825cc2.csv",
    "realAWSCloudwatch/rds_cpu_utilization_e47b3b.csv",
    "artificialWithAnomaly/art_daily_jumpsup.csv",
]


def _cache_dir(data_dir: str | None) -> Path:
    path = Path(data_dir) if data_dir else Path.home() / ".sigmaflow" / "benchmarks"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _download(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    req = urllib.request.Request(url, headers={"User-Agent": "sigmaflow"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        dest.write_bytes(resp.read())
    return dest


def load_nab(series: str = "realKnownCause/nyc_taxi.csv",
             data_dir: str | None = None) -> SignalFrame:
    """Load one series of the Numenta Anomaly Benchmark with its labels.

    ``series`` is the path within NAB's data directory — see
    :data:`NAB_SERIES` for a curated subset. Ground-truth anomaly windows
    come from NAB's ``combined_windows.json``.
    """
    cache = _cache_dir(data_dir)
    csv_path = _download(f"{_NAB_BASE}/data/{series}",
                         cache / series.replace("/", "__"))
    windows_path = _download(f"{_NAB_BASE}/labels/combined_windows.json",
                             cache / "nab_combined_windows.json")

    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    sf = SignalFrame(
        time=df["timestamp"].to_numpy(),
        values={"value": df["value"].to_numpy(dtype=float)},
        name=f"NAB {series}",
        metadata={
            "source": "Numenta Anomaly Benchmark",
            "series": series,
            "attribution": "NAB, AGPL-3.0, https://github.com/numenta/NAB",
        },
    )
    windows = json.loads(windows_path.read_text())[series]
    labels = np.zeros(len(sf), dtype=int)
    times = pd.DatetimeIndex(sf.time)
    for start, end in windows:
        mask = (times >= pd.Timestamp(start)) & (times <= pd.Timestamp(end))
        labels[np.asarray(mask)] = 1
    return sf.add_labels(labels)


_SMD_BASE = ("https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/"
             "master/ServerMachineDataset")


def load_smd(machine: str = "machine-1-1",
             data_dir: str | None = None) -> tuple[SignalFrame, SignalFrame]:
    """Load one machine of the Server Machine Dataset.

    Returns ``(train, test)`` SignalFrames; the test frame carries
    ground-truth labels. 38 telemetry channels per machine.
    """
    cache = _cache_dir(data_dir)
    train_path = _download(f"{_SMD_BASE}/train/{machine}.txt",
                           cache / f"smd_train_{machine}.txt")
    test_path = _download(f"{_SMD_BASE}/test/{machine}.txt",
                          cache / f"smd_test_{machine}.txt")
    label_path = _download(f"{_SMD_BASE}/test_label/{machine}.txt",
                           cache / f"smd_label_{machine}.txt")

    def to_frame(path: Path, name: str) -> SignalFrame:
        arr = np.loadtxt(path, delimiter=",")
        return SignalFrame(
            time=np.arange(len(arr), dtype=float),
            values={f"metric_{i}": arr[:, i] for i in range(arr.shape[1])},
            name=name,
            domain="satellite",  # closest built-in profile: machine telemetry
            metadata={
                "source": "Server Machine Dataset (OmniAnomaly)",
                "machine": machine,
                "attribution": "SMD, https://github.com/NetManAIOps/OmniAnomaly",
            },
        )

    train = to_frame(train_path, f"SMD {machine} train")
    test = to_frame(test_path, f"SMD {machine} test")
    test.add_labels(np.loadtxt(label_path).astype(int))
    return train, test


def load_yahoo_s5(*_args, **_kwargs):
    """Yahoo S5 requires accepting Yahoo's research license, so it can't
    be auto-downloaded."""
    raise RuntimeError(
        "Yahoo S5 must be requested manually (license): "
        "https://webscope.sandbox.yahoo.com/catalog.php?datatype=s . "
        "After download, load each CSV with pandas and wrap it in a "
        "SignalFrame with add_labels()."
    )
