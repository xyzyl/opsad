import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from sigmaflow import Pipeline
from sigmaflow.cli.main import cli
from sigmaflow.detectors import ZScoreDetector
from sigmaflow.preprocess import Detrend
from sigmaflow.synthetic import generate_generic_signal


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def csv_files(tmp_path):
    sig = generate_generic_signal(
        n=800, anomalies=[{"type": "spike", "index": 300, "magnitude": 12.0}]
    )
    data = str(tmp_path / "data.csv")
    labels = str(tmp_path / "labels.csv")
    df = sig.to_dataframe()
    df.index.name = "time"
    df.to_csv(data)
    pd.DataFrame({"label": sig.anomaly_labels}).to_csv(labels, index=False)
    return data, labels


def test_version(runner):
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_detect_csv(runner, csv_files):
    data, _ = csv_files
    result = runner.invoke(cli, ["detect", data, "--detector", "modified_zscore"])
    assert result.exit_code == 0, result.output
    assert "Detected" in result.output


def test_detect_with_params_and_output(runner, csv_files, tmp_path):
    data, _ = csv_files
    out = str(tmp_path / "out.csv")
    result = runner.invoke(cli, [
        "detect", data, "-d", "isolation_forest",
        "-p", "contamination=0.02", "-p", "n_estimators=50",
        "-o", out,
    ])
    assert result.exit_code == 0, result.output
    df = pd.read_csv(out)
    assert {"score", "label"} <= set(df.columns)


def test_detect_hdf5(runner, tmp_path):
    sig = generate_generic_signal(n=300, anomalies=[{"type": "spike", "index": 150}])
    path = str(tmp_path / "sig.h5")
    sig.to_hdf5(path)
    result = runner.invoke(cli, ["detect", path, "-d", "zscore"])
    assert result.exit_code == 0, result.output


def test_detect_with_pipeline(runner, csv_files, tmp_path):
    data, _ = csv_files
    pipe_path = str(tmp_path / "pipe.yaml")
    Pipeline([Detrend(method="linear"), ZScoreDetector()]).save(pipe_path)
    result = runner.invoke(cli, ["detect", data, "--pipeline", pipe_path])
    assert result.exit_code == 0, result.output


def test_detect_unknown_detector(runner, csv_files):
    data, _ = csv_files
    result = runner.invoke(cli, ["detect", data, "-d", "psychic"])
    assert result.exit_code != 0
    assert "unknown detector" in result.output


def test_detect_bad_param(runner, csv_files):
    data, _ = csv_files
    result = runner.invoke(cli, ["detect", data, "-p", "no_equals_sign"])
    assert result.exit_code != 0


def test_detect_missing_file(runner):
    result = runner.invoke(cli, ["detect", "no_such_file.csv"])
    assert result.exit_code != 0


def test_evaluate(runner, csv_files):
    data, labels = csv_files
    result = runner.invoke(cli, [
        "evaluate", data, "--labels", labels, "-d", "modified_zscore",
    ])
    assert result.exit_code == 0, result.output
    assert "f1" in result.output
    assert "auc_roc" in result.output


def test_evaluate_length_mismatch(runner, csv_files, tmp_path):
    data, _ = csv_files
    bad_labels = str(tmp_path / "bad.csv")
    pd.DataFrame({"label": np.zeros(10, dtype=int)}).to_csv(bad_labels, index=False)
    result = runner.invoke(cli, ["evaluate", data, "--labels", bad_labels])
    assert result.exit_code != 0
    assert "labels have" in result.output


def test_csv_with_datetime_time_column(runner, tmp_path):
    time = pd.date_range("2025-01-01", periods=200, freq="1h")
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 200)
    y[100] = 12.0
    df = pd.DataFrame({"time": time, "value": y})
    path = str(tmp_path / "dt.csv")
    df.to_csv(path, index=False)
    result = runner.invoke(cli, ["detect", path, "-d", "zscore"])
    assert result.exit_code == 0, result.output


def test_csv_no_numeric_columns(runner, tmp_path):
    path = str(tmp_path / "text.csv")
    pd.DataFrame({"time": [1, 2], "note": ["a", "b"]}).to_csv(path, index=False)
    result = runner.invoke(cli, ["detect", path])
    assert result.exit_code != 0
    assert "no numeric data columns" in result.output
