import json

import numpy as np
import pytest
from click.testing import CliRunner

from sigmaflow.cli.main import cli
from sigmaflow.data import SourceUnavailable
from sigmaflow.viz.dashboard import demo_signals
from sigmaflow.viz.static_export import (
    DEFAULT_EXPORT_DETECTORS,
    _signal_payload,
    _slugify,
    export_static_site,
)

CLASSIC = ["zscore", "modified_zscore", "cusum", "stl_residual",
           "isolation_forest", "lof"]


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    """Offline export of the demo signals with the fast classic detectors."""
    out = tmp_path_factory.mktemp("site")
    return export_static_site(str(out), signals=demo_signals(), live=False,
                              detectors=CLASSIC)


def test_slugify():
    assert _slugify("plasma: synthetic Langmuir probe") == "plasma-synthetic-langmuir-probe"


def test_export_writes_index_and_data(site):
    assert (site / "index.html").exists()
    data_files = list((site / "data").glob("*.json"))
    assert len(data_files) == 3  # exactly the signals passed — nothing added


def test_export_offline_without_signals_raises(tmp_path):
    with pytest.raises(SourceUnavailable, match="nothing to export"):
        export_static_site(str(tmp_path / "s"), live=False)


def test_index_contains_manifest_and_controls(site):
    html = (site / "index.html").read_text(encoding="utf-8")
    assert "__MANIFEST__" not in html  # placeholder replaced
    for component in ("signal-select", "detector-select", "threshold-method",
                      "narrative", "main-graph", "hist-graph", "anomaly-rows",
                      "source-note"):
        assert component in html
    assert "real instrument data" in html
    # narrative vocabularies made it into the page
    assert "disruption precursor" in html
    assert "marine heatwave" in html


def test_signal_payload_structure(site):
    payload = json.loads(
        (site / "data" / "plasma-synthetic-langmuir-probe.json").read_text(encoding="utf-8")
    )
    n = payload["n_samples"]
    assert len(payload["time"]) == n
    assert payload["domain"] == "plasma"
    ch = payload["channels"]["n_e"]
    assert len(ch["values"]) == n
    assert ch["unit"] == "m^-3"
    assert "This chart shows" in ch["para1"]
    assert ch["scale"] > 0
    assert payload["truth_intervals"]  # ground truth included
    assert set(payload["detectors"]) == set(CLASSIC)
    for det in payload["detectors"].values():
        assert len(det["scores"]) == n
        assert np.isfinite(det["auto_threshold"])


def test_default_export_detectors_include_autoencoder():
    assert "autoencoder" in DEFAULT_EXPORT_DETECTORS


def test_datetime_signal_exports_epoch_ms(site):
    payload = json.loads(
        (site / "data" / "ocean-synthetic-buoy-temperature.json").read_text(encoding="utf-8")
    )
    assert payload["time_kind"] == "datetime"
    assert payload["time"][0] > 1e12  # epoch milliseconds


def test_source_note_from_metadata(simple_signal):
    simple_signal.metadata.update({
        "source": "NOAA NDBC", "fetched_at": "2026-07-10 21:00 UTC",
        "attribution": "Data: NOAA (public domain)",
    })
    payload = _signal_payload("x", simple_signal, CLASSIC[:1])
    assert "NOAA NDBC" in payload["source_note"]
    assert "fetched 2026-07-10" in payload["source_note"]
    assert "live data from NOAA NDBC" in payload["channels"]["value"]["para1"]


def test_payload_without_truth(simple_signal):
    simple_signal.anomaly_labels = None
    payload = _signal_payload("x", simple_signal, CLASSIC[:1])
    assert payload["truth_intervals"] == []
    assert payload["source_note"] == ""


def test_nan_values_export_as_valid_json(rng):
    """Regression: live feeds contain NaN samples; Python's json module
    would emit literal `NaN`, which browsers reject — the ocean signal's
    channel dropdown silently stopped updating because of this."""
    from sigmaflow import SignalFrame

    y = rng.normal(15.0, 1.0, 500)
    y[100:120] = np.nan  # telemetry dropout, as in real NDBC data
    sf = SignalFrame(time=np.arange(500.0), values={"water_temperature": y})
    payload = _signal_payload("buoy", sf, CLASSIC[:2])

    text = json.dumps(payload, allow_nan=False)  # must not raise
    back = json.loads(text)
    vals = back["channels"]["water_temperature"]["values"]
    assert vals[110] is None                      # missing -> null
    assert vals[0] is not None
    # nan-aware stats stay finite despite the gap
    assert np.isfinite(back["channels"]["water_temperature"]["median"])
    assert back["channels"]["water_temperature"]["scale"] > 0


def test_why_section_present(site):
    html = (site / "index.html").read_text(encoding="utf-8")
    assert "why this, and not any anomaly detector?" in html
    assert "one detection engine with" in html


def test_cli_export_no_live(tmp_path):
    """--no-live with a user file exports that file without touching the net."""

    from sigmaflow.synthetic import generate_generic_signal

    sig = generate_generic_signal(n=400, anomalies=[{"type": "spike", "index": 200}])
    data = str(tmp_path / "data.csv")
    df = sig.to_dataframe()
    df.index.name = "time"
    df.to_csv(data)

    out_dir = str(tmp_path / "site")
    result = CliRunner().invoke(cli, ["dashboard", data, "--export", out_dir, "--no-live"])
    assert result.exit_code == 0, result.output
    assert "wrangler pages deploy" in result.output
    assert (tmp_path / "site" / "index.html").exists()
    files = list((tmp_path / "site" / "data").glob("*.json"))
    assert len(files) == 1


def test_cli_export_offline_no_data_errors(tmp_path, monkeypatch):
    result = CliRunner().invoke(
        cli, ["dashboard", "--export", str(tmp_path / "s"), "--no-live"])
    assert result.exit_code != 0
    assert "nothing to export" in result.output
