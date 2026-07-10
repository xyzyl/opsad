"""sigmaflow command-line interface."""

from __future__ import annotations

import sys

import click
import numpy as np
import pandas as pd

from .. import __version__
from ..core.pipeline import Pipeline
from ..core.signal_frame import SignalFrame
from ..detectors import DETECTOR_REGISTRY


def _load_signal(path: str, time_column: str | None = None) -> SignalFrame:
    """Load a SignalFrame from CSV or HDF5."""
    lower = path.lower()
    if lower.endswith((".h5", ".hdf5")):
        return SignalFrame.from_hdf5(path)
    df = pd.read_csv(path)
    if time_column is None:
        time_column = "time" if "time" in df.columns else df.columns[0]
    if time_column not in df.columns:
        raise click.ClickException(f"time column {time_column!r} not found in {path}")
    time = df[time_column]
    if not pd.api.types.is_numeric_dtype(time):
        time = pd.to_datetime(time)
    channels = {
        c: df[c].to_numpy(dtype=float)
        for c in df.columns
        if c != time_column and pd.api.types.is_numeric_dtype(df[c])
    }
    if not channels:
        raise click.ClickException(f"no numeric data columns found in {path}")
    return SignalFrame(time=time.to_numpy(), values=channels, name=path)


def _parse_params(params: tuple[str, ...]) -> dict:
    """Parse repeated ``-p key=value`` options with best-effort typing."""
    out = {}
    for item in params:
        if "=" not in item:
            raise click.ClickException(f"parameter {item!r} must be key=value")
        key, raw = item.split("=", 1)
        value: object = raw
        for cast in (int, float):
            try:
                value = cast(raw)
                break
            except ValueError:
                continue
        if raw.lower() in ("true", "false"):
            value = raw.lower() == "true"
        out[key.strip()] = value
    return out


def _build_runner(detector: str, pipeline: str | None, params: tuple[str, ...]):
    """Return an object with fit_detect(sf) from CLI options."""
    if pipeline:
        return Pipeline.load(pipeline)
    if detector not in DETECTOR_REGISTRY:
        raise click.ClickException(
            f"unknown detector {detector!r}; choose from {sorted(DETECTOR_REGISTRY)}"
        )
    return DETECTOR_REGISTRY[detector](**_parse_params(params))


@click.group()
@click.version_option(version=__version__, prog_name="sigmaflow")
def cli():
    """sigmaflow: anomaly detection for scientific time-series."""


@cli.command()
@click.argument("data_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--detector", "-d", default="isolation_forest", show_default=True,
              help="Detector name (ignored when --pipeline is given).")
@click.option("--pipeline", type=click.Path(exists=True, dir_okay=False),
              help="Saved pipeline YAML to run instead of a single detector.")
@click.option("--param", "-p", "params", multiple=True,
              help="Detector parameter as key=value (repeatable), e.g. -p contamination=0.01")
@click.option("--time-column", help="Name of the time column for CSV input.")
@click.option("--output", "-o", type=click.Path(dir_okay=False),
              help="Write per-timestep results (value, score, label) to this CSV.")
def detect(data_file, detector, pipeline, params, time_column, output):
    """Detect anomalies in DATA_FILE (CSV or HDF5)."""
    signal = _load_signal(data_file, time_column)
    runner = _build_runner(detector, pipeline, params)
    result = runner.fit_detect(signal)
    result.summary()
    if output:
        result.to_dataframe().to_csv(output)
        click.echo(f"results written to {output}")


@cli.command()
@click.argument("data_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--labels", required=True, type=click.Path(exists=True, dir_okay=False),
              help="CSV with ground-truth 0/1 labels (column 'label', or single column).")
@click.option("--detector", "-d", default="isolation_forest", show_default=True)
@click.option("--pipeline", type=click.Path(exists=True, dir_okay=False))
@click.option("--param", "-p", "params", multiple=True)
@click.option("--time-column", help="Name of the time column for CSV input.")
def evaluate(data_file, labels, detector, pipeline, params, time_column):
    """Run detection on DATA_FILE and score it against labeled anomalies."""
    from ..evaluation import evaluate as evaluate_result

    signal = _load_signal(data_file, time_column)

    labels_df = pd.read_csv(labels)
    if "label" in labels_df.columns:
        truth = labels_df["label"].to_numpy()
    else:
        truth = labels_df.iloc[:, -1].to_numpy()
    truth = np.asarray(truth).astype(int)
    if len(truth) != len(signal):
        raise click.ClickException(
            f"labels have {len(truth)} rows but signal has {len(signal)} samples"
        )

    runner = _build_runner(detector, pipeline, params)
    result = runner.fit_detect(signal)
    if len(result.labels) != len(truth):
        raise click.ClickException(
            "pipeline changed the signal length (e.g. resampling); "
            "evaluate needs labels aligned with the detector input"
        )
    metrics = evaluate_result(result, truth)

    click.echo(f"detector: {result.detector_name}")
    for key in ("precision", "recall", "f1", "mcc", "fpr",
                "event_precision", "event_recall", "detection_latency",
                "over_detection_ratio", "auc_roc", "auc_pr", "best_f1"):
        if key in metrics:
            click.echo(f"  {key:22s} {metrics[key]:.4f}")


@cli.command()
@click.argument("data_file", required=False,
                type=click.Path(exists=True, dir_okay=False))
@click.option("--detector", "-d", default="isolation_forest", show_default=True,
              help="Detector selected when the dashboard opens.")
@click.option("--port", default=8050, show_default=True, help="HTTP port.")
@click.option("--time-column", help="Name of the time column for CSV input.")
@click.option("--export", "export_dir", type=click.Path(file_okay=False),
              help="Instead of serving, write a deployable static site "
                   "(Cloudflare Pages, GitHub Pages, ...) to this directory.")
def dashboard(data_file, detector, port, time_column, export_dir):
    """Launch the interactive dashboard (needs sigmaflow[dashboard]).

    DATA_FILE (CSV or HDF5) is optional; the built-in synthetic demo
    signals are always available in the signal browser. With --export,
    a static version with precomputed scores is written instead of
    starting a server.
    """
    signal = _load_signal(data_file, time_column) if data_file else None

    if export_dir:
        from ..viz.static_export import export_static_site

        signals = {signal.name or "user signal": signal} if signal is not None else None
        out = export_static_site(export_dir, signals)
        click.echo(f"static site written to {out}")
        click.echo("deploy it with: npx wrangler pages deploy "
                   f"{out} --project-name <your-project>")
        return

    try:
        from ..viz.dashboard import launch_dashboard
    except ImportError:
        raise click.ClickException(
            "the dashboard needs optional dependencies: pip install sigmaflow[dashboard]"
        )
    click.echo(f"sigmaflow dashboard on http://127.0.0.1:{port}")
    launch_dashboard(signal=signal, detector=detector, port=port)


def main():  # pragma: no cover
    cli(sys.argv[1:])


if __name__ == "__main__":  # pragma: no cover
    main()
