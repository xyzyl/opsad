"""Interactive Dash dashboard: signal browser, live detector tuning,
threshold control, anomaly table with click-to-zoom.

Requires the ``dashboard`` extra:  pip install sigmaflow[dashboard]
"""

from __future__ import annotations

import numpy as np

from ..core.anomaly_result import labels_to_intervals
from ..core.signal_frame import SignalFrame, time_to_seconds
from ..detectors import DETECTOR_REGISTRY
from ..detectors.threshold import compute_threshold
from ..evaluation import evaluate

__all__ = ["launch_dashboard", "create_app"]

# ---------------------------------------------------------------------- #
# Palette (dataviz reference instance, light mode)
# ---------------------------------------------------------------------- #

C = {
    "surface": "#fcfcfb",
    "page": "#f9f9f7",
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "baseline": "#c3c2b7",
    "series": "#2a78d6",     # categorical slot 1 (blue)
    "critical": "#d03b3b",   # status: detected anomaly
    "truth": "#9ec5f4",      # sequential blue 200: ground-truth reference band
    "border": "rgba(11,11,11,0.10)",
}
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

# Editable parameters per detector: (name, input kind, default, help)
DETECTOR_PARAMS: dict[str, list[tuple]] = {
    "zscore": [
        ("window_size", "int", None, "rolling window (blank = global stats)"),
        ("threshold", "float", 3.0, "|z| alarm level"),
    ],
    "modified_zscore": [
        ("window_size", "int", None, "rolling window (blank = global stats)"),
        ("threshold", "float", 3.5, "score alarm level"),
    ],
    "cusum": [
        ("target", "float", None, "expected mean (blank = estimate)"),
        ("threshold", "float", 5.0, "alarm level in std units"),
        ("drift", "float", 0.5, "slack before accumulation"),
    ],
    "stl_residual": [
        ("period", "int", None, "seasonal period in samples (blank = auto)"),
        ("residual_threshold", "float", 3.0, "residual |z| alarm level"),
    ],
    "isolation_forest": [
        ("n_estimators", "int", 200, "number of trees"),
        ("contamination", "float", 0.02, "expected anomaly fraction (blank = auto)"),
        ("window_size", "int", 25, "feature window in samples"),
    ],
    "lof": [
        ("n_neighbors", "int", 20, "neighborhood size"),
        ("contamination", "float", 0.02, "expected anomaly fraction (blank = auto)"),
        ("window_size", "int", 25, "feature window in samples"),
    ],
}

MAX_PLOT_POINTS = 20_000  # decimate longer signals for display only

# ---------------------------------------------------------------------- #
# Plain-language interpretation
# ---------------------------------------------------------------------- #

DETECTOR_EXPLAINERS = {
    "zscore": "measures how far each reading sits from the signal's average, "
              "in units of its usual wobble — points far outside that wobble get flagged",
    "modified_zscore": "measures how far each reading sits from the signal's typical value, "
                       "using the median so a few wild readings can't skew what counts as normal",
    "cusum": "adds up small persistent deviations over time, so it notices slow shifts "
             "(drift, degradation) that no single reading would reveal",
    "stl_residual": "first strips away the signal's repeating rhythm (daily, seasonal, "
                    "orbital…), then flags only what departs from that rhythm",
    "isolation_forest": "asks, for every moment, how few yes/no questions it takes to single "
                        "it out from the rest — moments that are easy to isolate are unusual",
    "lof": "compares each moment with its nearest look-alikes and flags those sitting in "
           "sparse neighborhoods — normal globally, but odd for their context",
}

CHANNEL_DESCRIPTIONS = {
    "n_e": "electron density (how densely packed the plasma's charged particles are)",
    "T_e": "electron temperature",
    "temperature": "water temperature",
    "salinity": "salinity (dissolved salt content)",
    "value": "a sensor reading",
}

# what each anomaly *shape* usually means, per domain and across domains
DOMAIN_MEANING = {
    "plasma": {
        "spike": "electrical pickup or a momentary probe glitch",
        "flat": "the probe saturating — it hit the ceiling of what it can measure, "
                "so this is an instrument artifact, not a real plasma value",
        "shift": "a growing instability in the plasma — the kind of pattern monitored "
                 "as a disruption precursor in fusion experiments",
        "noisy": "interference, e.g. RF pickup from the machine's heating systems",
    },
    "ocean": {
        "spike": "a single bad reading — debris, a passing vessel, or a seabird on the buoy",
        "flat": "a stuck or iced-over sensor",
        "shift": "either a real ocean event (a marine heatwave) or slow sensor drift from "
                 "biofouling — the tell is whether neighboring instruments agree",
        "noisy": "storm or wave action contaminating the record",
    },
    None: {
        "spike": "an isolated glitch in the measurement chain",
        "flat": "a saturated or stuck sensor",
        "shift": "a genuine change in the system being measured — or a calibration jump",
        "noisy": "an interference burst",
    },
}

CROSS_DOMAIN = {
    "spike": "A one-sample spike like this is the universal instrument glitch: it shows up "
             "as cosmic-ray hits in satellite telemetry, voltage transients on grid sensors, "
             "and dropouts in ocean records.",
    "flat": "A flat-lined stretch is the cross-domain signature of a saturated sensor — a "
            "plasma probe at its ceiling, a frozen buoy, a clipped telemetry channel all "
            "look exactly like this.",
    "shift": "A sustained shift is the same signature a tokamak team watches for as a "
             "disruption precursor, an oceanographer reads as a marine heatwave, and a grid "
             "operator sees when a generator trips offline.",
    "noisy": "A burst of noise looks the same everywhere: RF pickup in a physics lab, storm "
             "chop at sea, radio interference on a satellite downlink.",
}


def _fmt_time(t, datetime_index: bool) -> str:
    if datetime_index:
        import pandas as pd

        return pd.Timestamp(t).strftime("%b %d, %H:%M")
    return f"t = {float(t):.4g} s"


def _fmt_duration(seconds: float) -> str:
    if seconds >= 172800:
        return f"{seconds / 86400:.0f} days"
    if seconds >= 5400:
        return f"{seconds / 3600:.3g} hours"
    if seconds >= 120:
        return f"{seconds / 60:.3g} minutes"
    return f"{seconds:.4g} seconds"


def characterize_interval(values: np.ndarray, i0: int, i1: int) -> str:
    """Classify what the signal did inside a detected interval:
    'spike', 'flat', 'shift', or 'noisy'."""
    seg = values[i0 : i1 + 1]
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med))) or float(np.std(values)) or 1.0
    scale = 1.4826 * mad
    if len(seg) >= 4 and np.ptp(seg) < 1e-12 * max(1.0, abs(float(seg[0]))):
        return "flat"
    if len(seg) <= 3:
        return "spike"
    if float(np.std(seg)) > 3.0 * scale:
        return "noisy"
    if abs(float(np.mean(seg)) - med) > 1.5 * scale:
        return "shift"
    return "shift"


_CHARACTER_PHRASES = {
    "spike": "jumped for a single instant, then returned to normal",
    "flat": "flat-lined at a constant value",
    "shift": "moved away from its typical level and stayed there",
    "noisy": "became far noisier than usual",
}


def build_narrative(sf: SignalFrame, channel: str, detector_name: str,
                    scores: np.ndarray, threshold: float,
                    labels: np.ndarray) -> list[str]:
    """Plain-language paragraphs explaining what the dashboard currently shows."""
    import pandas as pd

    values = sf[channel].to_numpy()
    secs = time_to_seconds(sf.time)
    is_dt = isinstance(sf.time, pd.DatetimeIndex)
    domain = sf.domain if sf.domain in DOMAIN_MEANING else None
    paragraphs = []

    # 1 — what the data is
    what = CHANNEL_DESCRIPTIONS.get(channel, f"the '{channel}' channel")
    unit = sf.units.get(channel)
    src = f" by a {sf.instrument.replace('_', ' ')}" if sf.instrument else ""
    rate = sf.sample_rate
    rate_str = (f", one reading every {_fmt_duration(1 / rate)}" if rate and rate < 1
                else f" at {rate:,.0f} readings per second" if rate else "")
    paragraphs.append(
        f"This chart shows {_fmt_duration(sf.duration)} of {what}"
        f"{f', in {unit},' if unit else ''} recorded{src}"
        f" — {len(sf):,} measurements{rate_str}."
    )

    # 2 — what the detector and threshold are doing
    explainer = DETECTOR_EXPLAINERS.get(detector_name, "scores each moment by how unusual it is")
    n_flagged = int(labels.sum())
    pct = 100.0 * n_flagged / len(labels) if len(labels) else 0.0
    events = labels_to_intervals(labels, sf.time, scores)
    paragraphs.append(
        f"The {detector_name.replace('_', ' ')} detector {explainer}. "
        f"The dashed red line in the lower chart is the alarm bar: every moment scoring "
        f"above it is declared an anomaly. At the current setting, {pct:.2f}% of all "
        f"readings clear it, grouped into {len(events)} distinct event"
        f"{'s' if len(events) != 1 else ''}."
    )

    # 3 — what was found and what it implies (strongest event)
    if events:
        start, end, severity = max(events, key=lambda e: e[2])
        i0 = sf.time.get_indexer([start])[0]
        i1 = sf.time.get_indexer([end])[0]
        character = characterize_interval(values, i0, i1)
        when = _fmt_time(start, is_dt)
        span = secs[i1] - secs[i0]
        span_str = f" for about {_fmt_duration(span)}" if span > 0 else ""
        meaning = DOMAIN_MEANING[domain][character]
        domain_name = domain or "general instrumentation"
        paragraphs.append(
            f"The strongest event is around {when}, where the signal "
            f"{_CHARACTER_PHRASES[character]}{span_str}. In {domain_name} terms, that "
            f"pattern most often means {meaning}. {CROSS_DOMAIN[character]}"
        )
    else:
        paragraphs.append(
            "Nothing currently clears the alarm bar — either this stretch of data is "
            "genuinely unremarkable, or the bar is set too high. Try the percentile "
            "threshold method with a value like 99 to surface the most unusual 1%."
        )

    # 4 — scoreboard against ground truth, when available
    if sf.anomaly_labels is not None:
        truth_events = labels_to_intervals(sf.anomaly_labels, sf.time)
        m = evaluate(labels, sf.anomaly_labels)
        caught = round(m["event_recall"] * len(truth_events))
        sentence = (
            f"This signal comes with an answer key: {len(truth_events)} genuine "
            f"anomal{'ies were' if len(truth_events) != 1 else 'y was'} planted in it, and "
            f"the detector currently catches {caught} of {len(truth_events)} "
            f"(the pale blue bands mark where they really are)."
        )
        if m["event_recall"] < 1.0:
            sentence += (" Lowering the alarm bar would catch more — at the price of more "
                         "false alarms on ordinary wiggles.")
        elif m["fpr"] > 0.05:
            sentence += (" It also flags a fair number of ordinary moments, though — "
                         "raising the alarm bar would cut those false alarms.")
        paragraphs.append(sentence)

    return paragraphs


# ---------------------------------------------------------------------- #
# Pure helpers (unit-testable without a running app)
# ---------------------------------------------------------------------- #

def demo_signals() -> dict[str, SignalFrame]:
    """The built-in signal browser entries (all carry ground-truth labels)."""
    from ..synthetic import (
        generate_generic_signal,
        generate_ocean_temperature,
        generate_plasma_signal,
    )

    return {
        "plasma: synthetic Langmuir probe": generate_plasma_signal(
            duration=5.0, sample_rate=2000
        ),
        "ocean: synthetic buoy temperature": generate_ocean_temperature(
            duration_days=180, samples_per_day=24,
            anomalies=[
                {"type": "marine_heatwave", "start_day": 40, "end_day": 41.5,
                 "magnitude": 4.0},
                {"type": "sensor_drift", "start_day": 120, "rate_per_day": 0.05},
            ],
        ),
        "generic: noise + spike + level shift": generate_generic_signal(
            n=2000,
            anomalies=[
                {"type": "spike", "index": 500},
                {"type": "level_shift", "start": 1200, "end": 1300},
            ],
        ),
    }


def build_detector(detector_name: str, params: dict):
    """Instantiate a detector from dashboard form values (None = default)."""
    if detector_name not in DETECTOR_REGISTRY:
        raise ValueError(f"unknown detector {detector_name!r}")
    kwargs = {}
    for name, _kind, default, _help in DETECTOR_PARAMS[detector_name]:
        value = params.get(name)
        if value is None or value == "":
            # blank contamination means sklearn's "auto", others use ctor default
            if name == "contamination":
                kwargs[name] = "auto"
            elif default is not None:
                kwargs[name] = default
        else:
            kwargs[name] = value
    return DETECTOR_REGISTRY[detector_name](**kwargs)


def compute_scores(sf: SignalFrame, detector_name: str, params: dict):
    """Fit + score; returns (scores, auto_threshold)."""
    detector = build_detector(detector_name, params)
    detector.fit(sf)
    scores = detector.score(sf)
    return scores, float(detector._auto_threshold(scores))


def resolve_threshold(scores: np.ndarray, method: str, value, auto_value: float) -> float:
    if method == "auto":
        return auto_value
    if method == "fixed" and value is None:
        return auto_value  # fixed without a value: fall back to the detector default
    return float(compute_threshold(scores, method, None if value is None else float(value)))


def intervals_records(sf: SignalFrame, labels: np.ndarray, scores: np.ndarray) -> list[dict]:
    """Anomaly intervals as table rows."""
    secs = time_to_seconds(sf.time)
    records = []
    for k, (start, end, severity) in enumerate(labels_to_intervals(labels, sf.time, scores), 1):
        i0 = sf.time.get_indexer([start])[0]
        i1 = sf.time.get_indexer([end])[0]
        records.append({
            "#": k,
            "start": str(start),
            "end": str(end),
            "duration (s)": round(float(secs[i1] - secs[i0]), 6),
            "peak score": round(float(severity), 4),
        })
    return records


def _decimate(*arrays, n: int):
    length = len(arrays[0])
    if length <= n:
        return arrays
    step = int(np.ceil(length / n))
    return tuple(a[::step] for a in arrays)


def _base_layout(fig):
    fig.update_layout(
        template="none",
        autosize=True,
        paper_bgcolor=C["surface"],
        plot_bgcolor=C["surface"],
        font=dict(family=FONT, color=C["ink2"], size=12),
        margin=dict(l=56, r=16, t=36, b=40),
        hovermode="x unified",
        dragmode="zoom",
    )
    fig.update_xaxes(gridcolor=C["grid"], linecolor=C["baseline"],
                     tickfont=dict(color=C["muted"]), zeroline=False)
    fig.update_yaxes(gridcolor=C["grid"], linecolor=C["baseline"],
                     tickfont=dict(color=C["muted"]), zeroline=False)
    return fig


def signal_score_figure(sf: SignalFrame, channel: str, scores: np.ndarray,
                        threshold: float, labels: np.ndarray,
                        show_truth: bool = True, x_range=None):
    """Stacked shared-x subplots: signal with anomaly bands, scores with threshold."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    time = np.asarray(sf.time)
    values = sf[channel].to_numpy()
    secs = time_to_seconds(sf.time)
    t_plot, v_plot, s_plot = _decimate(time, values, scores, n=MAX_PLOT_POINTS)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.62, 0.38], vertical_spacing=0.06)
    unit = sf.units.get(channel, "")

    fig.add_trace(
        go.Scattergl(x=t_plot, y=v_plot, mode="lines", name=channel,
                     line=dict(color=C["series"], width=1.6),
                     hovertemplate="%{y:.6g}<extra></extra>"),
        row=1, col=1,
    )
    # detected anomalies as markers on the signal — visible at any interval width
    flagged = labels.astype(bool)
    if flagged.any():
        (tf, vf) = _decimate(time[flagged], values[flagged], n=MAX_PLOT_POINTS // 4)
        fig.add_trace(
            go.Scattergl(x=tf, y=vf, mode="markers", name="detected anomaly",
                         marker=dict(color=C["critical"], size=5,
                                     line=dict(color=C["surface"], width=1)),
                         hovertemplate="anomaly %{y:.6g}<extra></extra>"),
            row=1, col=1,
        )
    fig.add_trace(
        go.Scattergl(x=t_plot, y=s_plot, mode="lines", name="anomaly score",
                     showlegend=False,
                     line=dict(color=C["series"], width=1.6),
                     hovertemplate="%{y:.4g}<extra></extra>"),
        row=2, col=1,
    )
    fig.add_hline(y=threshold, row=2, col=1, line_color=C["critical"],
                  line_dash="dash", line_width=1.5,
                  annotation_text=f"threshold {threshold:.3g}",
                  annotation_font_color=C["critical"], annotation_font_size=11)

    # ground-truth reference bands, padded so short intervals stay visible.
    # NB: plotly only registers row/col vrects on subplots that already
    # have traces, so this must come after add_trace.
    if show_truth and sf.anomaly_labels is not None:
        pad = (secs[-1] - secs[0]) / 500 if len(secs) > 1 else 0
        for start, end, _ in labels_to_intervals(sf.anomaly_labels, sf.time):
            i0 = max(0, sf.time.get_indexer([start])[0])
            i1 = sf.time.get_indexer([end])[0]
            x0, x1 = secs[i0] - pad / 2, secs[i1] + pad / 2
            if isinstance(time[0], np.datetime64):
                x0 = np.datetime64(int(x0 * 1e9), "ns")
                x1 = np.datetime64(int(x1 * 1e9), "ns")
            fig.add_vrect(x0=x0, x1=x1, fillcolor=C["truth"], opacity=0.4,
                          line_width=0, layer="below", row=1, col=1)

    fig.update_yaxes(title_text=f"{channel} [{unit}]" if unit else channel,
                     title_font=dict(size=12, color=C["ink2"]), row=1, col=1)
    fig.update_yaxes(title_text="score", title_font=dict(size=12, color=C["ink2"]),
                     row=2, col=1)
    _base_layout(fig)
    fig.update_layout(
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    font=dict(size=11, color=C["ink2"])),
    )
    if x_range:
        fig.update_xaxes(range=x_range)
    return fig


def histogram_figure(scores: np.ndarray, threshold: float):
    import plotly.graph_objects as go

    fig = go.Figure(
        go.Histogram(x=scores, nbinsx=60, marker_color=C["series"],
                     marker_line=dict(color=C["surface"], width=1),
                     hovertemplate="score %{x}<br>count %{y}<extra></extra>")
    )
    fig.add_vline(x=threshold, line_color=C["critical"], line_dash="dash",
                  line_width=1.5)
    _base_layout(fig)
    fig.update_layout(height=240, showlegend=False,
                      title=dict(text="score distribution (log count)",
                                 font=dict(size=13, color=C["ink2"])))
    fig.update_yaxes(type="log")
    return fig


def metric_tiles(sf: SignalFrame, labels: np.ndarray, scores: np.ndarray) -> list[tuple[str, str]]:
    """(label, value) pairs for the stat-tile row."""
    n_events = len(labels_to_intervals(labels, sf.time))
    tiles = [
        ("anomalies", f"{n_events}"),
        ("flagged samples", f"{100 * labels.mean():.2f}%"),
        ("max score", f"{float(np.max(scores)):.3g}" if len(scores) else "—"),
    ]
    if sf.anomaly_labels is not None:
        m = evaluate(labels, sf.anomaly_labels)
        tiles += [
            ("F1 vs truth", f"{m['f1']:.3f}"),
            ("event recall", f"{m['event_recall']:.2f}"),
            ("false pos. rate", f"{m['fpr']:.4f}"),
        ]
    return tiles


# ---------------------------------------------------------------------- #
# App factory
# ---------------------------------------------------------------------- #

def create_app(signals: dict[str, SignalFrame] | None = None,
               default_detector: str = "isolation_forest"):
    """Build the Dash app. ``signals`` maps browser names to SignalFrames;
    the synthetic demos are appended so the browser is never empty."""
    try:
        import dash_bootstrap_components as dbc
        from dash import ALL, Dash, Input, Output, State, dash_table, dcc, html
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "the dashboard needs optional dependencies: pip install sigmaflow[dashboard]"
        ) from exc

    registry: dict[str, SignalFrame] = dict(signals or {})
    for name, sf in demo_signals().items():
        registry.setdefault(name, sf)
    first_signal = next(iter(registry))

    app = Dash(
        __name__,
        title="sigmaflow dashboard",
        external_stylesheets=[dbc.themes.BOOTSTRAP],
    )

    label_style = {"fontSize": "0.72rem", "textTransform": "uppercase",
                   "letterSpacing": "0.06em", "color": C["muted"],
                   "marginBottom": "2px", "marginTop": "12px"}
    card_style = {"background": C["surface"], "border": f"1px solid {C['border']}",
                  "borderRadius": "8px", "padding": "16px"}

    def param_inputs(detector_name: str):
        rows = []
        for name, kind, default, help_text in DETECTOR_PARAMS[detector_name]:
            rows.append(html.Div(name.replace("_", " "), style=label_style))
            rows.append(dbc.Input(
                id={"type": "param", "name": name},
                type="number",
                value=default,
                placeholder=help_text,
                step="any" if kind == "float" else 1,
                size="sm",
            ))
            rows.append(html.Div(help_text, style={"fontSize": "0.7rem",
                                                   "color": C["muted"]}))
        return rows

    sidebar = html.Div(style={**card_style, "width": "300px", "flexShrink": "0"}, children=[
        html.Div("signal", style=label_style),
        dcc.Dropdown(id="signal-select", options=list(registry),
                     value=first_signal, clearable=False),
        html.Div("channel", style=label_style),
        dcc.Dropdown(id="channel-select", clearable=False),
        html.Div("detector", style=label_style),
        dcc.Dropdown(id="detector-select", options=list(DETECTOR_PARAMS),
                     value=default_detector, clearable=False),
        html.Div(id="param-container", children=param_inputs(default_detector)),
        html.Hr(style={"borderColor": C["grid"]}),
        html.Div("threshold method", style=label_style),
        dcc.Dropdown(id="threshold-method",
                     options=["auto", "percentile", "sigma", "fixed"],
                     value="auto", clearable=False),
        html.Div("threshold value", style=label_style),
        dbc.Input(id="threshold-value", type="number", step="any", size="sm",
                  placeholder="used by percentile / sigma / fixed"),
        dbc.Checkbox(id="show-truth", value=True,
                     label="show ground-truth bands",
                     style={"marginTop": "14px", "fontSize": "0.85rem",
                            "color": C["ink2"]}),
        dbc.Button("run detection", id="run-button", color="primary",
                   className="w-100", style={"marginTop": "14px",
                                             "background": C["series"]}),
        html.Div(id="run-status", style={"fontSize": "0.75rem",
                                         "color": C["muted"],
                                         "marginTop": "8px"}),
    ])

    main = html.Div(style={"flex": "1 1 480px", "minWidth": "0",
                           "display": "flex", "flexDirection": "column",
                           "gap": "16px"}, children=[
        html.Div(id="tiles", style={"display": "flex", "gap": "12px",
                                    "flexWrap": "wrap"}),
        html.Div(style=card_style, children=[
            html.Div("what is this showing?", style={
                "fontSize": "0.72rem", "textTransform": "uppercase",
                "letterSpacing": "0.06em", "color": C["muted"],
                "marginBottom": "6px"}),
            html.Div(id="narrative", style={"fontSize": "0.88rem",
                                            "lineHeight": "1.55",
                                            "color": C["ink2"],
                                            "maxWidth": "72ch"}),
        ]),
        html.Div(style=card_style, children=[
            dcc.Graph(id="main-graph", style={"width": "100%"},
                      config={"displaylogo": False, "responsive": True}),
        ]),
        html.Div(style={"display": "flex", "gap": "16px", "alignItems": "stretch",
                        "flexWrap": "wrap"},
                 children=[
            html.Div(style={**card_style, "flex": "1 1 320px", "minWidth": "0"},
                     children=[dcc.Graph(id="hist-graph", style={"width": "100%"},
                                         config={"displaylogo": False,
                                                 "responsive": True})]),
            html.Div(style={**card_style, "flex": "1 1 420px", "minWidth": "0",
                            "overflow": "auto"},
                     children=[
                html.Div("detected anomalies — click a row to zoom",
                         style={**label_style, "marginTop": "0"}),
                dash_table.DataTable(
                    id="anomaly-table",
                    columns=[{"name": c, "id": c} for c in
                             ("#", "start", "end", "duration (s)", "peak score")],
                    page_size=8,
                    style_as_list_view=True,
                    style_cell={"fontFamily": FONT, "fontSize": "0.8rem",
                                "color": C["ink2"], "background": C["surface"],
                                "textAlign": "left", "padding": "6px 10px"},
                    style_header={"color": C["muted"], "fontWeight": "600",
                                  "textTransform": "uppercase",
                                  "fontSize": "0.7rem",
                                  "borderBottom": f"1px solid {C['grid']}"},
                    style_data={"borderBottom": f"1px solid {C['grid']}"},
                ),
            ]),
        ]),
    ])

    app.layout = html.Div(
        style={"background": C["page"], "minHeight": "100vh",
               "fontFamily": FONT, "padding": "20px 24px"},
        children=[
            html.Div(style={"display": "flex", "alignItems": "baseline",
                            "gap": "12px", "marginBottom": "16px"}, children=[
                html.Span("σ", style={"fontSize": "1.6rem", "color": C["series"],
                                      "fontWeight": "700"}),
                html.H1("sigmaflow dashboard",
                        style={"fontSize": "1.15rem", "fontWeight": "650",
                               "color": C["ink"], "margin": "0"}),
                html.Span("interactive anomaly detection",
                          style={"fontSize": "0.85rem", "color": C["muted"]}),
            ]),
            dcc.Store(id="score-store"),
            html.Div(style={"display": "flex", "gap": "16px",
                            "alignItems": "flex-start", "flexWrap": "wrap"},
                     children=[sidebar, main]),
        ],
    )

    # ------------------------------------------------------------------ #
    # Callbacks
    # ------------------------------------------------------------------ #

    @app.callback(Output("channel-select", "options"),
                  Output("channel-select", "value"),
                  Input("signal-select", "value"))
    def _update_channels(signal_name):
        channels = registry[signal_name].channels
        return channels, channels[0]

    @app.callback(Output("param-container", "children"),
                  Input("detector-select", "value"))
    def _update_params(detector_name):
        return param_inputs(detector_name)

    @app.callback(
        Output("score-store", "data"),
        Output("run-status", "children"),
        Input("run-button", "n_clicks"),
        Input("signal-select", "value"),
        Input("detector-select", "value"),
        State({"type": "param", "name": ALL}, "value"),
        State({"type": "param", "name": ALL}, "id"),
        prevent_initial_call=False,
    )
    def _run_detection(_clicks, signal_name, detector_name, param_values, param_ids):
        import time as _time

        sf = registry[signal_name]
        params = {pid["name"]: val for pid, val in zip(param_ids, param_values)}
        t0 = _time.perf_counter()
        try:
            scores, auto_thr = compute_scores(sf, detector_name, params)
        except Exception as exc:
            return None, f"error: {exc}"
        elapsed = _time.perf_counter() - t0
        data = {"signal": signal_name, "detector": detector_name,
                "scores": scores.tolist(), "auto_threshold": auto_thr}
        return data, f"{detector_name} scored {len(scores)} samples in {elapsed:.2f}s"

    @app.callback(
        Output("main-graph", "figure"),
        Output("hist-graph", "figure"),
        Output("anomaly-table", "data"),
        Output("tiles", "children"),
        Output("narrative", "children"),
        Input("score-store", "data"),
        Input("threshold-method", "value"),
        Input("threshold-value", "value"),
        Input("show-truth", "value"),
        Input("channel-select", "value"),
        Input("anomaly-table", "active_cell"),
        State("anomaly-table", "data"),
    )
    def _render(data, method, value, show_truth, channel, active_cell, table_rows):
        import plotly.graph_objects as go
        from dash import ctx

        if not data:
            empty = _base_layout(go.Figure())
            return empty, empty, [], [], []
        sf = registry[data["signal"]]
        if channel not in sf.channels:
            channel = sf.channels[0]
        scores = np.asarray(data["scores"])
        threshold = resolve_threshold(scores, method or "auto", value,
                                      data["auto_threshold"])
        labels = (scores > threshold).astype(int)
        records = intervals_records(sf, labels, scores)

        x_range = None
        if (ctx.triggered_id == "anomaly-table" and active_cell
                and table_rows and active_cell["row"] < len(table_rows)):
            row = table_rows[active_cell["row"]]
            secs = time_to_seconds(sf.time)
            span = max(row["duration (s)"], (secs[-1] - secs[0]) / 100)
            if hasattr(sf.time, "tz") or str(sf.time.dtype).startswith("datetime"):
                import pandas as pd
                start = pd.Timestamp(row["start"]) - pd.Timedelta(seconds=2 * span)
                end = pd.Timestamp(row["end"]) + pd.Timedelta(seconds=2 * span)
            else:
                start = float(row["start"]) - 2 * span
                end = float(row["end"]) + 2 * span
            x_range = [start, end]

        main_fig = signal_score_figure(sf, channel, scores, threshold, labels,
                                       show_truth=bool(show_truth), x_range=x_range)
        hist_fig = histogram_figure(scores, threshold)

        tile_divs = [
            html.Div(style={"background": C["surface"],
                            "border": f"1px solid {C['border']}",
                            "borderRadius": "8px", "padding": "10px 18px",
                            "minWidth": "120px"},
                     children=[
                html.Div(label, style={"fontSize": "0.68rem",
                                       "textTransform": "uppercase",
                                       "letterSpacing": "0.06em",
                                       "color": C["muted"]}),
                html.Div(val, style={"fontSize": "1.35rem", "fontWeight": "650",
                                     "color": C["ink"]}),
            ])
            for label, val in metric_tiles(sf, labels, scores)
        ]

        paragraphs = build_narrative(sf, channel, data["detector"], scores,
                                     threshold, labels)
        narrative = [html.P(p, style={"margin": "0 0 8px 0"}) for p in paragraphs]

        return main_fig, hist_fig, records, tile_divs, narrative

    return app


def launch_dashboard(signal: SignalFrame | None = None,
                     detector: str = "isolation_forest",
                     port: int = 8050, debug: bool = False) -> None:
    """Launch the interactive dashboard in a local web server.

    ``signal`` (optional) is added to the signal browser alongside the
    built-in synthetic demos and selected by default.
    """
    signals = {}
    if signal is not None:
        signals[signal.name or "user signal"] = signal
    app = create_app(signals, default_detector=detector)
    app.run(port=port, debug=debug)
