import json

import numpy as np
import pytest
from click.testing import CliRunner

from sigmaflow.cli.main import cli
from sigmaflow.viz.static_export import _signal_payload, _slugify, export_static_site


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    out = tmp_path_factory.mktemp("site")
    return export_static_site(str(out))


def test_slugify():
    assert _slugify("plasma: synthetic Langmuir probe") == "plasma-synthetic-langmuir-probe"


def test_export_writes_index_and_data(site):
    assert (site / "index.html").exists()
    data_files = list((site / "data").glob("*.json"))
    assert len(data_files) == 3  # the three demo signals


def test_index_contains_manifest_and_controls(site):
    html = (site / "index.html").read_text(encoding="utf-8")
    assert "__MANIFEST__" not in html  # placeholder replaced
    for component in ("signal-select", "detector-select", "threshold-method",
                      "narrative", "main-graph", "hist-graph", "anomaly-rows"):
        assert component in html
    assert "static export" in html  # the honest note about re-fitting
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
    # every detector precomputed, scores aligned, thresholds finite
    from sigmaflow.detectors import DETECTOR_REGISTRY

    assert set(payload["detectors"]) == set(DETECTOR_REGISTRY)
    for det in payload["detectors"].values():
        assert len(det["scores"]) == n
        assert np.isfinite(det["auto_threshold"])
        assert all(np.isfinite(s) for s in det["scores"][:100])


def test_datetime_signal_exports_epoch_ms(site):
    payload = json.loads(
        (site / "data" / "ocean-synthetic-buoy-temperature.json").read_text(encoding="utf-8")
    )
    assert payload["time_kind"] == "datetime"
    assert payload["time"][0] > 1e12  # epoch milliseconds


def test_custom_signal_included(tmp_path, simple_signal):
    out = export_static_site(str(tmp_path / "s"), {"my custom signal": simple_signal})
    slugs = {p.stem for p in (out / "data").glob("*.json")}
    assert "my-custom-signal" in slugs
    assert len(slugs) == 4  # custom + 3 demos


def test_payload_without_truth(simple_signal):
    simple_signal.anomaly_labels = None
    payload = _signal_payload("x", simple_signal)
    assert payload["truth_intervals"] == []


def test_cli_export(tmp_path):
    out_dir = str(tmp_path / "site")
    result = CliRunner().invoke(cli, ["dashboard", "--export", out_dir])
    assert result.exit_code == 0, result.output
    assert "wrangler pages deploy" in result.output
    assert (tmp_path / "site" / "index.html").exists()
