"""SignalFrame: the core time-series + metadata data structure."""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

__all__ = ["SignalFrame", "time_to_seconds"]


def time_to_seconds(index: pd.Index) -> np.ndarray:
    """Convert a time index (datetime or numeric) to float seconds."""
    if isinstance(index, pd.DatetimeIndex):
        # normalize to ns explicitly — pandas indexes may carry other units
        return np.asarray(index, dtype="datetime64[ns]").astype(np.int64) / 1e9
    return np.asarray(index, dtype=np.float64)


def _build_time_index(time: Iterable) -> pd.Index:
    """Normalize array-like time input into a DatetimeIndex or Float64 index."""
    if isinstance(time, pd.DatetimeIndex):
        return time
    arr = np.asarray(time)
    if arr.ndim != 1:
        raise ValueError("time must be one-dimensional")
    if arr.size == 0:
        raise ValueError("time must not be empty")
    if np.issubdtype(arr.dtype, np.datetime64) or arr.dtype == object or arr.dtype.kind == "U":
        return pd.DatetimeIndex(pd.to_datetime(arr))
    return pd.Index(arr.astype(np.float64))


class SignalFrame:
    """A time-series with channels, physical units, and instrument metadata.

    Metadata is always optional: a bare ``SignalFrame(time=t, values=y)``
    works everywhere in the library. When present, units, expected ranges,
    and domain information enrich validation and reporting.
    """

    def __init__(
        self,
        time: Iterable,
        values: Iterable | Mapping[str, Iterable],
        name: str | None = None,
        units: Mapping[str, str] | None = None,
        sample_rate: float | None = None,
        instrument: str | None = None,
        domain: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ):
        index = _build_time_index(time)

        if isinstance(values, Mapping):
            data = {str(k): np.asarray(v, dtype=np.float64) for k, v in values.items()}
        else:
            arr = np.asarray(values, dtype=np.float64)
            if arr.ndim == 2:
                data = {f"channel_{i}": arr[:, i] for i in range(arr.shape[1])}
            else:
                data = {"value": arr}

        for channel, arr in data.items():
            if arr.ndim != 1:
                raise ValueError(f"channel {channel!r} must be one-dimensional")
            if len(arr) != len(index):
                raise ValueError(
                    f"channel {channel!r} has {len(arr)} samples but time has {len(index)}"
                )

        self._df = pd.DataFrame(data, index=index)
        self.name = name
        self.units = dict(units) if units else {}
        self.instrument = instrument
        self.domain = domain
        self.metadata = dict(metadata) if metadata else {}
        self._sample_rate = float(sample_rate) if sample_rate else None
        self.anomaly_labels: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def time(self) -> pd.Index:
        return self._df.index

    @property
    def channels(self) -> list[str]:
        return list(self._df.columns)

    @property
    def values(self) -> pd.DataFrame:
        return self._df

    @property
    def sample_rate(self) -> float | None:
        """Sample rate in Hz — specified at creation, or inferred from timestamps."""
        if self._sample_rate is not None:
            return self._sample_rate
        secs = time_to_seconds(self._df.index)
        if len(secs) < 2:
            return None
        dt = float(np.median(np.diff(secs)))
        return 1.0 / dt if dt > 0 else None

    @property
    def duration(self) -> float:
        """Total time span in seconds."""
        secs = time_to_seconds(self._df.index)
        return float(secs[-1] - secs[0]) if len(secs) > 1 else 0.0

    def __getitem__(self, channel: str) -> pd.Series:
        return self._df[channel]

    def __len__(self) -> int:
        return len(self._df)

    def __repr__(self) -> str:
        name = self.name or "unnamed"
        return (
            f"SignalFrame({name!r}, channels={self.channels}, "
            f"n={len(self)}, sample_rate={self.sample_rate})"
        )

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #

    def _with(self, df: pd.DataFrame, **overrides: Any) -> "SignalFrame":
        """Copy this frame's metadata onto a new DataFrame (used by transforms)."""
        sf = SignalFrame(
            time=df.index,
            values={c: df[c].to_numpy() for c in df.columns},
            name=overrides.get("name", self.name),
            units=overrides.get("units", self.units),
            sample_rate=overrides.get("sample_rate", self._sample_rate),
            instrument=self.instrument,
            domain=self.domain,
            metadata=overrides.get("metadata", self.metadata),
        )
        labels = overrides.get("anomaly_labels", self.anomaly_labels)
        if labels is not None and len(labels) == len(sf):
            sf.anomaly_labels = np.asarray(labels)
        return sf

    # ------------------------------------------------------------------ #
    # Operations
    # ------------------------------------------------------------------ #

    def resample(self, target_rate: float) -> "SignalFrame":
        """Resample to a uniform grid at ``target_rate`` Hz via linear interpolation.

        For downsampling with anti-alias filtering use
        :class:`sigmaflow.preprocess.Resampler`.
        """
        if target_rate <= 0:
            raise ValueError("target_rate must be positive")
        secs = time_to_seconds(self._df.index)
        new_secs = np.arange(secs[0], secs[-1] + 1e-12, 1.0 / target_rate)
        data = {
            c: np.interp(new_secs, secs, self._df[c].to_numpy()) for c in self._df.columns
        }
        if isinstance(self._df.index, pd.DatetimeIndex):
            new_index = pd.DatetimeIndex((new_secs * 1e9).astype("int64").view("datetime64[ns]"))
        else:
            new_index = pd.Index(new_secs)
        df = pd.DataFrame(data, index=new_index)
        return self._with(df, sample_rate=float(target_rate), anomaly_labels=None)

    def slice(self, start, end) -> "SignalFrame":
        """Time-based slice, inclusive of both endpoints."""
        if isinstance(self._df.index, pd.DatetimeIndex):
            start, end = pd.to_datetime(start), pd.to_datetime(end)
        mask = (self._df.index >= start) & (self._df.index <= end)
        df = self._df.loc[mask]
        labels = self.anomaly_labels[np.asarray(mask)] if self.anomaly_labels is not None else None
        return self._with(df, anomaly_labels=labels)

    def dropna(self) -> "SignalFrame":
        """Remove timesteps where any channel is NaN."""
        mask = ~self._df.isna().any(axis=1)
        df = self._df.loc[mask]
        labels = self.anomaly_labels[np.asarray(mask)] if self.anomaly_labels is not None else None
        return self._with(df, anomaly_labels=labels)

    def interpolate(self, method: str = "linear") -> "SignalFrame":
        """Fill NaN gaps. Methods: 'linear', 'cubic', 'nearest'."""
        if method not in ("linear", "cubic", "nearest"):
            raise ValueError(f"unknown interpolation method {method!r}")
        if isinstance(self._df.index, pd.DatetimeIndex):
            # pandas requires method="time" for value-aware interpolation on datetimes
            df = self._df.interpolate(method="time" if method == "linear" else method)
        else:
            df = self._df.interpolate(method="index" if method == "linear" else method)
        df = df.ffill().bfill()
        return self._with(df)

    def add_labels(self, anomaly_labels: Iterable) -> "SignalFrame":
        """Attach ground-truth anomaly labels (0/1 per timestep) for evaluation."""
        labels = np.asarray(anomaly_labels).astype(int)
        if len(labels) != len(self):
            raise ValueError(
                f"labels have length {len(labels)} but signal has {len(self)} samples"
            )
        self.anomaly_labels = labels
        return self

    # ------------------------------------------------------------------ #
    # Export / import
    # ------------------------------------------------------------------ #

    def to_numpy(self) -> np.ndarray:
        """Raw values as (n_samples, n_channels) array."""
        return self._df.to_numpy()

    def to_dataframe(self) -> pd.DataFrame:
        return self._df.copy()

    def to_hdf5(self, path: str) -> None:
        """Persist signal and metadata to an HDF5 file."""
        import h5py

        with h5py.File(path, "w") as f:
            is_datetime = isinstance(self._df.index, pd.DatetimeIndex)
            f.create_dataset("time", data=time_to_seconds(self._df.index))
            grp = f.create_group("values")
            for c in self._df.columns:
                grp.create_dataset(c, data=self._df[c].to_numpy())
            if self.anomaly_labels is not None:
                f.create_dataset("anomaly_labels", data=self.anomaly_labels)
            f.attrs["time_kind"] = "datetime" if is_datetime else "float"
            f.attrs["name"] = self.name or ""
            f.attrs["instrument"] = self.instrument or ""
            f.attrs["domain"] = self.domain or ""
            f.attrs["units"] = json.dumps(self.units)
            f.attrs["metadata"] = json.dumps(self.metadata, default=str)
            if self._sample_rate is not None:
                f.attrs["sample_rate"] = self._sample_rate

    @classmethod
    def from_hdf5(cls, path: str) -> "SignalFrame":
        """Load a signal previously saved with :meth:`to_hdf5`."""
        import h5py

        with h5py.File(path, "r") as f:
            secs = f["time"][:]
            if f.attrs.get("time_kind") == "datetime":
                time = pd.DatetimeIndex((secs * 1e9).astype("int64").view("datetime64[ns]"))
            else:
                time = secs
            values = {c: f["values"][c][:] for c in f["values"]}
            sf = cls(
                time=time,
                values=values,
                name=f.attrs.get("name") or None,
                units=json.loads(f.attrs.get("units", "{}")),
                sample_rate=float(f.attrs["sample_rate"]) if "sample_rate" in f.attrs else None,
                instrument=f.attrs.get("instrument") or None,
                domain=f.attrs.get("domain") or None,
                metadata=json.loads(f.attrs.get("metadata", "{}")),
            )
            if "anomaly_labels" in f:
                sf.add_labels(f["anomaly_labels"][:])
        return sf

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #

    def gap_report(self, factor: float = 3.0) -> list[tuple]:
        """Return (start, end, duration_seconds) for gaps larger than
        ``factor`` times the median sampling interval."""
        secs = time_to_seconds(self._df.index)
        if len(secs) < 3:
            return []
        diffs = np.diff(secs)
        median_dt = float(np.median(diffs))
        gaps = []
        for i in np.nonzero(diffs > factor * median_dt)[0]:
            gaps.append((self._df.index[i], self._df.index[i + 1], float(diffs[i])))
        return gaps

    def summary(self) -> str:
        """Human-readable summary: channels, duration, sample rate, gaps, stats."""
        lines = [f"SignalFrame: {self.name or 'unnamed'}"]
        if self.domain or self.instrument:
            lines.append(f"  domain={self.domain or '-'}  instrument={self.instrument or '-'}")
        lines.append(f"  samples: {len(self)}   duration: {self.duration:.6g} s")
        rate = self.sample_rate
        lines.append(f"  sample rate: {rate:.6g} Hz" if rate else "  sample rate: unknown")
        gaps = self.gap_report()
        lines.append(f"  gaps: {len(gaps)}")
        for c in self.channels:
            s = self._df[c]
            unit = self.units.get(c, "")
            lines.append(
                f"  {c}{f' [{unit}]' if unit else ''}: "
                f"min={s.min():.6g} max={s.max():.6g} mean={s.mean():.6g} "
                f"std={s.std():.6g} nan={int(s.isna().sum())}"
            )
        text = "\n".join(lines)
        print(text)
        return text
