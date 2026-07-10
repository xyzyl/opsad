"""GapHandler: detect and fill gaps in a time-series."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.base import BasePreprocessor
from ..core.signal_frame import SignalFrame, time_to_seconds

__all__ = ["GapHandler"]

_FILL_METHODS = ("interpolate", "forward_fill", "nan", "zero")


def _parse_gap(max_gap) -> float:
    """Accept float seconds or a pandas-style string like '5s' / '2min'."""
    if isinstance(max_gap, str):
        return pd.Timedelta(max_gap).total_seconds()
    return float(max_gap)


class GapHandler(BasePreprocessor):
    """Detect missing stretches in the time base and fill them.

    Gaps no longer than ``max_gap`` are filled by inserting timesteps at
    the signal's median sampling interval, with values chosen by
    ``fill_method``. Larger gaps are left untouched but recorded in
    ``metadata['gaps']`` (with ``report_gaps=True``) so downstream steps
    and users can see them. Existing NaN values are filled the same way.
    """

    name = "gap_handler"

    def __init__(
        self,
        max_gap="5s",
        fill_method: str = "interpolate",
        report_gaps: bool = True,
    ):
        if fill_method not in _FILL_METHODS:
            raise ValueError(f"unknown fill_method {fill_method!r}; choose from {_FILL_METHODS}")
        self.max_gap = max_gap
        self.fill_method = fill_method
        self.report_gaps = report_gaps

    def transform(self, sf: SignalFrame) -> SignalFrame:
        secs = time_to_seconds(sf.time)
        if len(secs) < 3:
            return sf
        max_gap_s = _parse_gap(self.max_gap)
        diffs = np.diff(secs)
        median_dt = float(np.median(diffs))
        gap_idx = np.nonzero(diffs > 1.5 * median_dt)[0]

        new_secs = [secs]
        large_gaps = []
        for i in gap_idx:
            gap = float(diffs[i])
            if gap > max_gap_s:
                large_gaps.append((sf.time[i], sf.time[i + 1], gap))
                continue
            inserted = np.arange(secs[i] + median_dt, secs[i + 1] - median_dt / 2, median_dt)
            if len(inserted):
                new_secs.append(inserted)

        all_secs = np.sort(np.concatenate(new_secs))
        data = {}
        for c in sf.channels:
            x = sf[c].to_numpy()
            filled = np.full(len(all_secs), np.nan)
            pos = np.searchsorted(all_secs, secs)
            filled[pos] = x
            data[c] = self._fill(all_secs, filled)

        if isinstance(sf.time, pd.DatetimeIndex):
            index = pd.DatetimeIndex((all_secs * 1e9).astype("int64").view("datetime64[ns]"))
        else:
            index = pd.Index(all_secs)
        df = pd.DataFrame(data, index=index)

        metadata = dict(sf.metadata)
        if self.report_gaps:
            metadata["gaps"] = [
                {"start": str(s), "end": str(e), "duration_s": d} for s, e, d in large_gaps
            ]
        return sf._with(df, metadata=metadata, anomaly_labels=None)

    def _fill(self, secs: np.ndarray, x: np.ndarray) -> np.ndarray:
        missing = np.isnan(x)
        if not missing.any():
            return x
        if self.fill_method == "nan":
            return x
        if self.fill_method == "zero":
            out = x.copy()
            out[missing] = 0.0
            return out
        valid = ~missing
        if not valid.any():
            return x
        if self.fill_method == "interpolate":
            return np.interp(secs, secs[valid], x[valid])
        # forward_fill: carry the previous valid value forward
        out = x.copy()
        idx = np.where(valid, np.arange(len(x)), 0)
        np.maximum.accumulate(idx, out=idx)
        out = out[idx]
        # samples before the first valid value stay at the first valid value
        first = np.argmax(valid)
        out[:first] = x[first]
        return out
