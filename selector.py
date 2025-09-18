# selector.py
# Cross-sectional candidate selection for "exuberance fade".
# Ranks symbols by an exuberance_score each minute and marks top-K as tradeable.

from __future__ import annotations
from typing import Dict, Any
import pandas as pd
import numpy as np


def _safe(series, default=0.0):
    if series is None:
        return default
    s = pd.Series(series)
    return s.fillna(default)


def annotate_candidates(
    dfs_by_symbol: Dict[str, pd.DataFrame],
    params: Any,                             # backtest.Params or similar (uses attrs below)
    top_k_per_min: int | None = None,        # if None, use params.top_k_per_min if present, else 5
) -> Dict[str, pd.DataFrame]:
    """
    Build 'candidate_ok' per symbol by ranking an exuberance score across symbols each minute.
    Score uses normalized thresholds: gap, vwap_z, vol_mult, rsi. Liquidity gates apply first.

    Returns new dict of DataFrames (shallow copies) with 'candidate_ok' (bool) and 'exuberance_score' (float).
    """
    # defaults
    price_min = float(getattr(params, "price_min", 3.0))
    dollar_vol_min = float(getattr(params, "dollar_vol_min", 5e5))  # per-minute $
    gap_th = float(getattr(params, "gap_th", 0.03))
    vwap_z_th = float(getattr(params, "vwap_z", 3.0))
    rsi_th = float(getattr(params, "rsi_th", 80))
    vol_mult_th = getattr(params, "vol_mult_th", None)
    if vol_mult_th is None:
        vol_mult_th = 3.0
    vol_mult_th = float(vol_mult_th)

    top_k = int(top_k_per_min if top_k_per_min is not None else getattr(params, "top_k_per_min", 5))

    # 1) Build a compact cross-sectional frame
    frames = []
    for sym, df in dfs_by_symbol.items():
        if df.empty:
            continue
        x = df[["timestamp", "close", "volume"]].copy()
        x["symbol"] = sym

        # Features (if missing in df, fill zeros)
        x["gap_pct"] = df["gap_pct"] if "gap_pct" in df else 0.0
        x["vwap_z"]  = df["vwap_z"]  if "vwap_z"  in df else 0.0
        x["vol_mult"] = df["vol_mult"] if "vol_mult" in df else 0.0
        x["rsi"] = df["rsi"] if "rsi" in df else 50.0

        # Liquidity gates
        x["price_ok"] = x["close"] >= price_min
        x["dollar_vol"] = x["close"] * x["volume"]
        x["dv_ok"] = x["dollar_vol"] >= dollar_vol_min

        # Normalized contributions (positive part only)
        # Avoid divide-by-zero by max(threshold, tiny)
        tiny = 1e-9
        x["g_gap"] = (x["gap_pct"] / max(gap_th, tiny)).clip(lower=0.0)
        x["g_vwap"] = (x["vwap_z"] / max(vwap_z_th, tiny)).clip(lower=0.0)
        x["g_vol"] = (x["vol_mult"] / max(vol_mult_th, tiny)).clip(lower=0.0)
        x["g_rsi"] = (x["rsi"] / max(rsi_th, tiny)).clip(lower=0.0)

        # Simple equal-weight score (weights can be added later if desired)
        x["exuberance_score"] = x["g_gap"] + x["g_vwap"] + x["g_vol"] + x["g_rsi"]

        frames.append(x)

    if not frames:
        return {sym: df.assign(candidate_ok=False, exuberance_score=0.0) for sym, df in dfs_by_symbol.items()}

    all_rows = pd.concat(frames, ignore_index=True)
    all_rows.sort_values(["timestamp", "exuberance_score"], ascending=[True, False], inplace=True)

    # 2) Per-minute ranking across symbols
    def pick_top(g: pd.DataFrame):
        g2 = g.copy()
        # Apply gates first
        g2 = g2[(g2["price_ok"]) & (g2["dv_ok"]) & (g2["exuberance_score"] > 0)]
        if g2.empty:
            g["candidate_ok"] = False
            return g[["symbol", "timestamp", "candidate_ok", "exuberance_score"]]
        g2 = g2.head(top_k)
        g2["candidate_ok"] = True
        # Merge back so we keep False for non-picked rows
        out = g.merge(g2[["symbol", "candidate_ok"]], on="symbol", how="left")
        out["candidate_ok"] = out["candidate_ok"].astype("boolean").fillna(False).astype(bool)
        return out[["symbol", "timestamp", "candidate_ok", "exuberance_score"]]

    picks = all_rows.groupby("timestamp", group_keys=False).apply(pick_top)
    # 3) Split back per symbol and merge onto original frames
    out: Dict[str, pd.DataFrame] = {}
    for sym, df in dfs_by_symbol.items():
        if df.empty:
            out[sym] = df.assign(candidate_ok=False, exuberance_score=0.0)
            continue
        sel = picks[picks["symbol"] == sym][["timestamp", "candidate_ok", "exuberance_score"]]
        z = df.merge(sel, on="timestamp", how="left")
        z["candidate_ok"] = z["candidate_ok"].fillna(False)
        z["exuberance_score"] = z["exuberance_score"].fillna(0.0)
        out[sym] = z
    return out
