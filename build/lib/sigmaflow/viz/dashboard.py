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
            duration_days=180, samples_per_day=24
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
            return empty, empty, [], []
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
        return main_fig, hist_fig, records, tile_divs

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
