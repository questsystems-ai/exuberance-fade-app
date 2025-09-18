"""
Auto-loaded patch: normalizes Polygon minute bars to the backtester schema.
- Creates a 'timestamp' column (UTC) from Polygon's 't' (ns/ms) or 'start_timestamp'
- Renames {o,h,l,c,v,vw} -> {open,high,low,close,volume,vwap}
- Ensures columns: timestamp, open, high, low, close, volume, vwap
Only affects data_ingest.load_polygon_minutes; Alpaca path untouched.
"""

from __future__ import annotations
import pandas as pd
import numpy as np

def _normalize_polygon_df(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None or len(df) == 0:
        return df

    # If Polygon returned nested/records, coerce to DataFrame
    if not isinstance(df, pd.DataFrame):
        try:
            df = pd.DataFrame(df)
        except Exception:
            return df

    # Timestamp
    if "timestamp" not in df.columns:
        ts = None
        if "t" in df.columns:
            # Polygon aggregates: 't' is epoch ns (sometimes ms depending on source)
            try:
                ts = pd.to_datetime(df["t"], unit="ns", utc=True)
            except Exception:
                ts = pd.to_datetime(df["t"], unit="ms", utc=True)
        elif "start_timestamp" in df.columns:
            ts = pd.to_datetime(df["start_timestamp"], utc=True)
        if ts is not None:
            df = df.assign(timestamp=ts)

    # Rename short keys -> standard
    rename = {"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "vw": "vwap"}
    df = df.rename(columns=rename)

    # Create vwap if missing
    if "vwap" not in df.columns:
        if {"open", "high", "low", "close"}.issubset(df.columns):
            df["vwap"] = df[["open", "high", "low", "close"]].mean(axis=1)
        else:
            df["vwap"] = np.nan

    # Keep/order known columns; drop rows without timestamp
    cols = [c for c in ["timestamp", "open", "high", "low", "close", "volume", "vwap"] if c in df.columns]
    if "timestamp" in df.columns:
        df = df.dropna(subset=["timestamp"])
        try:
            df = df.sort_values("timestamp")
        except Exception:
            pass
    return df[cols] if cols else df

# Monkey-patch data_ingest.load_polygon_minutes
try:
    import data_ingest as _di
    if hasattr(_di, "load_polygon_minutes"):
        _orig_load = _di.load_polygon_minutes

        def load_polygon_minutes(symbol: str, start: str, end: str, rth_only: bool = True):
            out = _orig_load(symbol, start, end, rth_only=rth_only)
            return _normalize_polygon_df(out)

        _di.load_polygon_minutes = load_polygon_minutes  # type: ignore[attr-defined]
except Exception as _e:
    # Fail open: if anything goes wrong, do not block program start
    pass
