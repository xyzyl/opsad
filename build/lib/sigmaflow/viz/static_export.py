"""Export the dashboard as a static website (Cloudflare Pages, GitHub
Pages, any static host).

Scores for every signal x detector pair are precomputed into JSON;
thresholding, metrics, the anomaly table, and the plain-language
narrative run client-side, so the published page stays interactive.
Live detector *re-fitting* (parameter tuning) is the one thing that
needs the local Python app.

Usage:
    sigmaflow dashboard --export site/
    # or
    from sigmaflow.viz.static_export import export_static_site
    export_static_site("site/")
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.anomaly_result import labels_to_intervals
from ..core.signal_frame import SignalFrame, time_to_seconds
from .dashboard import (
    CHANNEL_DESCRIPTIONS,
    CROSS_DOMAIN,
    DETECTOR_EXPLAINERS,
    DETECTOR_PARAMS,
    DOMAIN_MEANING,
    _CHARACTER_PHRASES,
    _fmt_duration,
    compute_scores,
)

__all__ = ["export_static_site", "DEFAULT_EXPORT_DETECTORS"]

#: detectors precomputed into the static site. The LSTM and transformer
#: models are omitted by default purely for export-build time — pass
#: ``detectors=[...]`` to include them.
DEFAULT_EXPORT_DETECTORS = [
    "zscore", "modified_zscore", "cusum", "stl_residual",
    "isolation_forest", "lof", "autoencoder",
]


def _sig(x: float, digits: int = 5) -> float:
    """Round to significant digits so the JSON stays small."""
    return float(f"{float(x):.{digits}g}")


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _first_paragraph(sf: SignalFrame, channel: str) -> str:
    """Threshold-independent opening paragraph (mirrors build_narrative)."""
    what = CHANNEL_DESCRIPTIONS.get(channel, f"the '{channel}' channel")
    unit = sf.units.get(channel)
    src = f" by a {sf.instrument.replace('_', ' ')}" if sf.instrument else ""
    rate = sf.sample_rate
    rate_str = (f", one reading every {_fmt_duration(1 / rate)}" if rate and rate < 1
                else f" at {rate:,.0f} readings per second" if rate else "")
    para = (f"This chart shows {_fmt_duration(sf.duration)} of {what}"
            f"{f', in {unit},' if unit else ''} recorded{src}"
            f" — {len(sf):,} measurements{rate_str}.")
    source = sf.metadata.get("source")
    if source:
        fetched = sf.metadata.get("fetched_at")
        para += (f" This is real, live data from {source}"
                 + (f", fetched {fetched}." if fetched else "."))
    return para


def _signal_payload(name: str, sf: SignalFrame,
                    detectors: list[str] | None = None) -> dict:
    is_dt = isinstance(sf.time, pd.DatetimeIndex)
    secs = time_to_seconds(sf.time)
    if is_dt:
        time_out = [int(s * 1000) for s in secs]  # epoch ms for plotly date axes
    else:
        time_out = [_sig(s, 6) for s in secs]

    channels = {}
    for c in sf.channels:
        x = sf[c].to_numpy()
        median = float(np.median(x))
        mad = float(np.median(np.abs(x - median))) or float(np.std(x)) or 1.0
        channels[c] = {
            "values": [_sig(v, 6) for v in x],
            "unit": sf.units.get(c, ""),
            "median": _sig(median, 6),
            "scale": _sig(1.4826 * mad, 6),
            "para1": _first_paragraph(sf, c),
        }

    truth_intervals = []
    if sf.anomaly_labels is not None:
        for start, end, _ in labels_to_intervals(sf.anomaly_labels, sf.time):
            i0 = int(sf.time.get_indexer([start])[0])
            i1 = int(sf.time.get_indexer([end])[0])
            truth_intervals.append([i0, i1])

    detector_names = [d for d in (detectors or DEFAULT_EXPORT_DETECTORS)
                      if d in DETECTOR_PARAMS]
    detector_payload = {}
    for det_name in detector_names:
        defaults = {p: d for p, _k, d, _h in DETECTOR_PARAMS[det_name] if d is not None}
        scores, auto_thr = compute_scores(sf, det_name, defaults)
        detector_payload[det_name] = {
            "scores": [_sig(s, 4) for s in scores],
            "auto_threshold": _sig(auto_thr, 5),
            "params": defaults,
        }

    source_note = ""
    if sf.metadata.get("source"):
        bits = [sf.metadata["source"]]
        if sf.metadata.get("fetched_at"):
            bits.append(f"fetched {sf.metadata['fetched_at']}")
        if sf.metadata.get("attribution"):
            bits.append(sf.metadata["attribution"])
        source_note = " — ".join(bits)

    return {
        "name": name,
        "domain": sf.domain or "",
        "instrument": sf.instrument or "",
        "source_note": source_note,
        "time_kind": "datetime" if is_dt else "float",
        "time": time_out,
        "channels": channels,
        "truth_intervals": truth_intervals,
        "n_samples": len(sf),
        "detectors": detector_payload,
    }


def export_static_site(output_dir: str,
                       signals: dict[str, SignalFrame] | None = None,
                       live: bool = True,
                       detectors: list[str] | None = None) -> Path:
    """Write a deployable static dashboard into ``output_dir``.

    With ``live=True`` (default) the four live open-data sources are
    fetched at export time, so the published site shows real instrument
    data with a build timestamp. Raises if no signals are available —
    the website never falls back to synthetic data.

    Returns the output path. The directory is safe to publish as-is:
    ``index.html`` plus one JSON per signal under ``data/``.
    """
    from ._static_template import HTML_TEMPLATE

    out = Path(output_dir)
    (out / "data").mkdir(parents=True, exist_ok=True)

    registry = dict(signals or {})
    if live:
        from ..data import fetch_all_live

        registry.update(fetch_all_live())
    if not registry:
        from ..data import SourceUnavailable

        raise SourceUnavailable(
            "nothing to export: live sources are unreachable and no signals "
            "were passed. Check your connection, or pass signals= explicitly."
        )

    detector_names = [d for d in (detectors or DEFAULT_EXPORT_DETECTORS)
                      if d in DETECTOR_PARAMS]
    manifest_signals = []
    for name, sf in registry.items():
        slug = _slugify(name)
        payload = _signal_payload(name, sf, detector_names)
        payload["slug"] = slug
        with open(out / "data" / f"{slug}.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, separators=(",", ":"))
        manifest_signals.append({"slug": slug, "name": name})

    # domain-meaning table with a JSON-safe key for the fallback entry
    domain_meaning = {(k or "generic"): v for k, v in DOMAIN_MEANING.items()}
    manifest = {
        "signals": manifest_signals,
        "detectors": detector_names,
        "explainers": DETECTOR_EXPLAINERS,
        "domainMeaning": domain_meaning,
        "crossDomain": CROSS_DOMAIN,
        "characterPhrases": _CHARACTER_PHRASES,
    }

    html = HTML_TEMPLATE.replace("__MANIFEST__", json.dumps(manifest))
    with open(out / "index.html", "w", encoding="utf-8") as f:
        f.write(html)
    return out
