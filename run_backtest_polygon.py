#!/usr/bin/env python3
# Exuberance Fade Backtester — cleaned & guarded
# Implements RTH helper + shard slicing guard + window QC logging.
# Polygon path includes monthly-chunk prefetch + resilient monkeypatch.
# Also supports --use-optimized to apply best_params from summary.csv before reporting.

import argparse, os, json, warnings, time, zipfile
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytz
from tqdm import tqdm

# Project imports
from config import DATA_DIR, REPORTS_DIR, DEFAULT_SYMBOLS
from data_ingest import (
    load_local_parquet,
    generate_synthetic_minutes,
    load_alpaca_minutes,
)
# Polygon parallel + original single-symbol loader
from data_ingest import load_polygon_minutes_multi
from data_ingest import load_polygon_minutes as _orig_load_polygon_minutes

# Provide a GLOBAL default name that other helpers can call.
# We may overwrite this symbol later (monkeypatch) after args are parsed.
def load_polygon_minutes(symbol, *args, **kwargs):
    return _orig_load_polygon_minutes(symbol, *args, **kwargs)

from signals import (
    add_intraday_features,
    signal_gap_fade,
    signal_vwap_extreme,
    signal_late_blowoff,
    minute_volume_profile_flag,
)
from backtest import Params, simulate_symbol
from optimizer import walk_forward, save_results, month_splits  # month_splits used for window count
from selector import annotate_candidates
from monitoring import AlertManager, RateLimitGuard
from reporter import generate_quick_report

# --- warnings control ---------------------------------------------------------
warnings.filterwarnings("ignore", message="DataFrameGroupBy.apply")
warnings.filterwarnings("ignore", message="Converting to PeriodArray/Index representation will drop timezone information")
QC_MIN_SYMS_DEFAULT = int(os.environ.get("QC_MIN_SYMS", "1"))

# --- utils: timestamp normalization ------------------------------------------
def _resolve_ts_column(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """Find a timestamp column among common names or an index. Ensures tz-aware UTC dtype."""
    candidates = ["ts", "timestamp", "time", "datetime", "date", "t", "bar_time"]
    tscol = None
    for c in candidates:
        if c in df.columns:
            tscol = c
            break
    if tscol is None:
        # any datetime typed column?
        for c in df.columns:
            try:
                if pd.api.types.is_datetime64_any_dtype(df[c]):
                    tscol = c
                    break
            except Exception:
                pass
    if tscol is None and str(df.index.dtype).startswith("datetime64"):
        df = df.reset_index().rename(columns={"index": "ts"})
        tscol = "ts"
    if tscol is None:
        raise ValueError(f"Could not locate a timestamp column; cols={list(df.columns)[:12]}")
    # ensure tz-aware UTC
    df[tscol] = pd.to_datetime(df[tscol], utc=True, errors="coerce")
    return df, tscol


def _month_keys(df: pd.DataFrame, tscol: str) -> pd.Series:
    """UTC -> naive -> to_period('M') -> str to avoid tz warnings."""
    return (
        df[tscol]
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
        .dt.to_period("M")
        .astype(str)
    )

# --- RTH helper ---------------------------------------------------------------
def _apply_rth_et(df: pd.DataFrame, *, log_prefix: str) -> pd.DataFrame:
    """Keep Monday–Friday 09:30–16:00 ET; warn + expect fallback if empty."""
    df, ts = _resolve_ts_column(df)
    et = df[ts].dt.tz_convert("America/New_York")
    mins = et.dt.hour * 60 + et.dt.minute
    is_weekday = et.dt.dayofweek < 5
    rth_mask = is_weekday & (mins >= 570) & (mins <= 960)  # 09:30–16:00
    out = df.loc[rth_mask].copy()
    if out.empty:
        print(f"[WARN] {log_prefix}: empty after RTH; retrying without RTH")
    return out


# --- local/remote loaders (no RTH inside) ------------------------------------
def _load_symbol_data(symbol: str, start: str, end: str, source: str,
                      alp_guard: RateLimitGuard, pol_guard: RateLimitGuard,
                      alert: AlertManager) -> pd.DataFrame:
    try:
        if source == "local":
            return load_local_parquet(symbol, start, end, data_dir=DATA_DIR)
        elif source == "alpaca":
            df = load_alpaca_minutes(symbol, start, end, rth_only=False)
            alp_guard.record(1)
            return df
        elif source == "polygon":
            df = load_polygon_minutes(symbol, start, end, rth_only=False)  # monkeypatched name
            pol_guard.record(1)
            return df
        else:
            raise ValueError(f"Unknown source '{source}'. Use one of: local|alpaca|polygon")
    except Exception as e:
        msg = str(e).lower()
        if "429" in msg or "rate limit" in msg or "too many requests" in msg:
            alert.error(f"{source} rate-limit while loading {symbol} {start}->{end}: {e}")
        else:
            alert.error(f"{source} loader error for {symbol} {start}->{end}: {e}")
        raise


# --- shard slicing guard ------------------------------------------------------
def _normalize_symbols(sym_in) -> List[str]:
    """Support strings: 'AAPL MSFT,NVDA' or lists/tuples."""
    if isinstance(sym_in, str):
        s = sym_in.strip()
        return [t.strip() for t in (s.split(",") if "," in s else s.split()) if t.strip()]
    if isinstance(sym_in, (list, tuple)) and len(sym_in) == 1 and isinstance(sym_in[0], str):
        s = sym_in[0].strip()
        return [t.strip() for t in (s.split(",") if "," in s else s.split()) if t.strip()]
    if isinstance(sym_in, (list, tuple)):
        return [str(x).strip() for x in sym_in if str(x).strip()]
    try:
        return list(sym_in)
    except Exception:
        return [str(sym_in)]


def symbols_for_shard(all_symbols, shard_index: int | None, shard_count: int | None) -> List[str]:
    """Deterministically slice by position: idx % shard_count == shard_index"""
    full = _normalize_symbols(all_symbols)
    if not full:
        raise SystemExit("Error: parsed symbol list is empty.")
    if shard_index is None or shard_count is None or shard_count <= 1:
        out = full
    else:
        out = [s for idx, s in enumerate(full) if idx % shard_count == shard_index]
    print(f"[shard {shard_index}/{shard_count}] symbols: {out}")
    if len(out) == 0:
        raise SystemExit("Shard produced zero symbols. Check --shard-index/--shard-count and --symbols.")
    return out


# --- feature preparation ------------------------------------------------------
def _prepare_symbol_df(df: pd.DataFrame, symbol: str, base: Params) -> pd.DataFrame:
    df = df.copy()
    df["symbol"] = symbol
    tscol = "timestamp" if "timestamp" in df.columns else df.columns[0]
    df = df.sort_values(tscol)
    df = add_intraday_features(df)
    df = minute_volume_profile_flag(df, multiple=3.0)
    df = signal_gap_fade(df, gap_th=base.gap_th, hold_minutes=base.hold_minutes)
    df = signal_vwap_extreme(df, z_th=base.vwap_z, rsi_th=base.rsi_th)
    df = signal_late_blowoff(df, breakout_pct=base.breakout_pct)
    return df


# --- run dir/meta -------------------------------------------------------------
def _make_run_dir(base_reports_dir: str, run_tag: str | None = None):
def _make_run_dir_shared(base_reports_dir: str, args):
    """
    If args.run_root is set, use reports/<run_root>/<run_root>_s<index> (or <run_root> if non-sharded).
    Otherwise fall back to _make_run_dir().
    Returns (run_dir, run_id, now_utc, parent_dir).
    """
    now_utc = datetime.now(timezone.utc)
    if getattr(args, "run_root", None):
        root = os.path.join(base_reports_dir, args.run_root)
        os.makedirs(root, exist_ok=True)
        if args.shard_index is not None and args.shard_count and args.shard_count > 1:
            child = f"{args.run_root}_s{args.shard_index}"
        else:
            child = args.run_root
        run_dir = os.path.join(root, child)
        os.makedirs(run_dir, exist_ok=True)
        # run_id becomes the child folder name to keep per-run uniqueness stable
        return run_dir, child, now_utc, root
    # default behavior
    run_dir, run_id, now_utc = _make_run_dir(base_reports_dir, getattr(args, "run_tag", None))
    return run_dir, run_id, now_utc, os.path.dirname(run_dir)

    now_utc = datetime.now(timezone.utc)
    run_id = now_utc.strftime("%Y%m%d_%H%M%SZ")
    if run_tag:
        run_id = f"{run_id}_{run_tag}"
    run_dir = os.path.join(base_reports_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, run_id, now_utc


# --- window counting + runtime estimate --------------------------------------
def _count_windows(dfs: Dict[str, pd.DataFrame], train_m: int, test_m: int) -> int:
    months = sorted(set().union(*[set(month_splits(df)) for df in dfs.values()])) if dfs else []
    return max(0, len(months) - train_m - test_m + 1)


def _estimate_runtime_seconds(dfs: Dict[str, pd.DataFrame], base: Params, units: int) -> float:
    if not dfs:
        return 0.0
    sample = next((df for df in dfs.values() if len(df) > 0), None)
    if sample is None:
        return 0.0
    n = min(4000, len(sample))
    mini = sample.head(n)
    t0 = time.time()
    simulate_symbol(mini, base, initial_equity=100_000.0)
    dt = max(1e-6, time.time() - t0)
    sec_per_row = dt / n
    total_rows = sum(len(df) for df in dfs.values())
    return sec_per_row * total_rows * max(1, units)


# --- window QC logging --------------------------------------------------------
def _qc_train_test_counts(dfs: Dict[str, pd.DataFrame], train_m: int, test_m: int,
                          rth_used: Dict[str, bool], qc_mode: str = "common") -> Tuple[Dict[str, Tuple[int,int]], int, List[str]]:
    """Return per-symbol (train_bars, test_bars), window count, and test month using a quorum rule.
    If qc_mode == "common": require all symbols to have the test month; otherwise use a quorum threshold
    read from env QC_MIN_SYMS (default 1). This only affects QC logging + optional dropping, not the optimizer windows.
    """
    # per-symbol month sets
    per_months: Dict[str, set] = {}
    for sym, df in dfs.items():
        df, ts = _resolve_ts_column(df)
        months = _month_keys(df, ts)
        per_months[sym] = set(months.dropna().unique())

    # newest-first candidate months (union)
    candidates = sorted(set().union(*per_months.values()) if per_months else [], reverse=True)

    # threshold
    if qc_mode == "common":
        thresh = len(per_months) if per_months else 0
    else:
        thresh = max(1, QC_MIN_SYMS_DEFAULT)

    # pick newest month that >= thresh symbols actually have
    test_months: List[str] = []
    for m in candidates:
        have = sum(1 for s in per_months if m in per_months[s])
        if have >= thresh:
            test_months = [m] if test_m > 0 else []
            break

    # info only (not used to split windows)
    sorted_months = sorted(set().union(*per_months.values())) if per_months else []
    windows_count = max(0, len(sorted_months) - train_m - test_m + 1)

    # per-symbol counts
    per_counts: Dict[str, Tuple[int,int]] = {}
    for sym, df in dfs.items():
        if df.empty:
            per_counts[sym] = (0, 0)
            print(f"[QC] {sym}: train_bars=0 test_bars=0 (after {'RTH' if rth_used.get(sym, False) else 'ALL'})")
            continue
        df, ts = _resolve_ts_column(df)
        mser = _month_keys(df, ts)

        test_mask = mser.isin(test_months) if test_months else mser == "__none__"
        if test_months:
            edge = test_months[0]
            train_mask = mser < edge
        else:
            train_mask = mser.notna()

        train_bars = int(train_mask.sum())
        test_bars  = int(test_mask.sum())
        tag = "RTH" if rth_used.get(sym, False) else "ALL"
        print(f"[QC] {sym}: train_bars={train_bars} test_bars={test_bars} (after {tag}; test_month={test_months[0] if test_months else 'none'}; thresh={thresh})")
        per_counts[sym] = (train_bars, test_bars)

    return per_counts, windows_count, test_months

def main():
    ap = argparse.ArgumentParser(description="Exuberance Fade Backtester")
    ap.add_argument("--symbols", nargs="*", default=DEFAULT_SYMBOLS,
                    help="Space or comma separated; quotes OK, e.g. 'MSFT AVGO' or 'MSFT,AVGO'")
    ap.add_argument("--source", default="local", choices=["local", "alpaca", "polygon"])
    ap.add_argument("--start", required=False, default="2024-01-01")
    ap.add_argument("--end",   required=False, default="2025-12-31")
    ap.add_argument("--rth-only", action="store_true", default=False)

    ap.add_argument("--initial_equity", type=float, default=100000.0)
    ap.add_argument("--shard-index", type=int, default=None)
    ap.add_argument("--shard-count", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1337)

    ap.add_argument("--no-opt", action="store_true")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--max-combos", type=int, default=None)
    ap.add_argument("--train-months", type=int, default=None)
    ap.add_argument("--test-months",  type=int, default=None)

    ap.add_argument("--write-cache", action="store_true")
    ap.add_argument("--run-tag", type=str, default=None)

    ap.add_argument("--alpaca-limit", type=int, default=200)
    ap.add_argument("--polygon-limit", type=int, default=60)

    ap.add_argument("--warn-runtime-mins", type=float, default=15.0)
    ap.add_argument("--qc-mode", type=str, default="common", choices=["common","union","off"],
                    help="Month alignment for QC: common=intersection, union=union, off=don\'t drop symbols")
    ap.add_argument("--no-report", action="store_true")

    ap.add_argument("--use-optimized", action="store_true", default=False,
                    help="Re-sim with best params from summary.csv before writing report")
    ap.add_argument("--min-rth-bars-per-day", type=int, default=0,
                    help="Drop US/Eastern days with fewer than N RTH minute bars (0=off)")
    ap.add_argument("--polygon-max-concurrency", type=int, default=6,
                    help="Max concurrent Polygon requests (default 6)")
    ap.add_argument("--polygon-reqs-per-min", type=int, default=100,
                    help="Soft RPM clamp for Polygon (default 100)")
    ap.add_argument("--run-root", type=str, default=os.environ.get("RUN_ROOT_ID"),
                    help="Shared parent folder under reports/ for multi-shard runs; child = \"<run-root>_s<index>\"")
    ap.add_argument("--auto-zip", action="store_true",
                    help="When using --run-root with shards, zip the parent once all shard folders exist")

    args = ap.parse_args()
    # --- grid override via env (monkeypatch optimizer._full_grid) ---
    import os
    try:
        import optimizer as _opt
        if hasattr(_opt, "_full_grid"):
            _orig_full_grid = _opt._full_grid
            def _parse_env_list(name: str):
                v = os.environ.get(name)
                if not v: return None
                parts = [t.strip() for t in v.replace(";",",").split(",") if t.strip()]
                out = []
                for t in parts:
                    try: out.append(float(t))
                    except: out.append(t)
                return out
            def _patched_full_grid():
                g = _orig_full_grid()
                overrides = {
                    "take_profit_sigma": _parse_env_list("GRID_TAKE_PROFIT_SIGMA"),
                    "stop_loss_pct":     _parse_env_list("GRID_STOP_LOSS_PCT"),
                    "vol_mult_th":       _parse_env_list("GRID_VOL_MULT_TH"),
                    "vwap_z":            _parse_env_list("GRID_VWAP_Z"),
                    "stake_pct":         _parse_env_list("GRID_STAKE_PCT"),
                }
                overrides = {k:v for k,v in overrides.items() if v is not None}
                if overrides:
                    g.update(overrides)
                    print("[grid override]", overrides)
                return g
            _opt._full_grid = _patched_full_grid
    except Exception as e:
        print(f"[WARN] grid override patch failed: {e}")
    # --- /grid override ---


    # --- polygon: global prefetch + monkeypatch ---
    try:
        _polygon_frames = None
        if getattr(args, "source", "local") == "polygon":
            # normalize symbols (commas & whitespace)
            _symbols_list = []
            _syms = getattr(args, "symbols", [])
            if isinstance(_syms, (list, tuple)):
                for x in _syms:
                    for tok in str(x).replace(",", " ").split():
                        if tok.strip():
                            _symbols_list.append(tok.strip().upper())
            elif isinstance(_syms, str):
                _symbols_list = [t.strip().upper() for t in _syms.replace(",", " ").split() if t.strip()]
            if not _symbols_list:
                raise RuntimeError("No symbols parsed for Polygon prefetch")

            from pandas.tseries.offsets import MonthEnd
            import pandas as _pd
            start_ts = _pd.Timestamp(args.start, tz="UTC").normalize()
            end_ts   = _pd.Timestamp(args.end,   tz="UTC").normalize()
            months = list(_pd.period_range(start_ts.to_period("M"), end_ts.to_period("M"), freq="M"))

            _acc = {s: [] for s in _symbols_list}
            for per in months:
                m_start = _pd.Timestamp(f"{per.year}-{per.month:02d}-01", tz="UTC")
                m_end   = (m_start + MonthEnd(1)).normalize() + _pd.Timedelta(hours=23, minutes=59, seconds=59)

                for sym in _symbols_list:
                    # first attempt: single-symbol fetch (avoid cross-symbol paging races)
                    part = load_polygon_minutes_multi(
                        [sym],
                        m_start.isoformat(),
                        m_end.isoformat(),
                        rth_only=getattr(args, "rth_only", True),
                        adjusted=False,   # raw to match Alpaca
                        max_concurrency=1,
                        reqs_per_min=int(getattr(args, "polygon_reqs_per_min", 60)),
                    ).get(sym)

                    # lightweight completeness check (forgiving threshold)
                    need_retry = False
                    if part is not None and not part.empty:
                        et = part["timestamp"].dt.tz_convert("US/Eastern")
                        n_days = max(1, et.dt.normalize().nunique())
                        if len(part) < max(300 * n_days, 3500):
                            need_retry = True
                    else:
                        need_retry = True

                    if need_retry:
                        # gentle retry with lower RPM
                        part = load_polygon_minutes_multi(
                            [sym],
                            m_start.isoformat(),
                            m_end.isoformat(),
                            rth_only=getattr(args, "rth_only", True),
                            adjusted=False,
                            max_concurrency=1,
                            reqs_per_min=min(40, int(getattr(args, "polygon_reqs_per_min", 60))),
                        ).get(sym)

                    if part is not None and not part.empty:
                        _acc[sym].append(part)

                print(f"[polygon] month {per} fetched (per-symbol); rows by sym:",
                      {k: sum(len(x) for x in v) for k, v in _acc.items()})

            # assemble final per-symbol frames
            _polygon_frames = {}
            for sym, parts in _acc.items():
                if parts:
                    df = _pd.concat(parts, ignore_index=True)
                    if "timestamp" in df.columns:
                        df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
                    _polygon_frames[sym] = df.reset_index(drop=True)
                else:
                    _polygon_frames[sym] = _pd.DataFrame()

        else:
            _polygon_frames = None

        # expose loader that returns prefetched frames when available
        global load_polygon_minutes
        def load_polygon_minutes(symbol, *args2, **kwargs2):
            if _polygon_frames is not None and isinstance(symbol, str):
                _sym = symbol.strip().upper()
                _df = _polygon_frames.get(_sym)
                if _df is not None and not _df.empty:
                    return _df
            return _orig_load_polygon_minutes(symbol, *args2, **kwargs2)

    except Exception as _e:
        print(f"[WARN] polygon prefetch/monkeypatch failed: {_e} (falling back to per-call loader)")
    # --- /polygon: global prefetch + monkeypatch ---

    # Make run folder & alert channel
    os.makedirs(REPORTS_DIR, exist_ok=True)
    run_dir, run_id, now_utc, _parent_dir = _make_run_dir_shared(REPORTS_DIR, args)
    alert = AlertManager(run_dir)
    alp_guard = RateLimitGuard("Alpaca", args.alpaca_limit if args.alpaca_limit > 0 else None, alert)
    pol_guard = RateLimitGuard("Polygon", args.polygon_limit if args.polygon_limit > 0 else None, alert)

    # --- shard slicing guard ---
    shard_syms = symbols_for_shard(args.symbols, args.shard_index, args.shard_count)

    base = Params()

    # 1) Load & prepare
    dfs: Dict[str, pd.DataFrame] = {}
    rth_used: Dict[str, bool] = {}

    for sym in tqdm(shard_syms, desc="Loading & preparing"):
        try:
            raw = generate_synthetic_minutes(sym, args.start, args.end) if False else \
                  _load_symbol_data(sym, args.start, args.end, args.source, alp_guard, pol_guard, alert)

            if args.rth_only:
                df_rth = _apply_rth_et(raw, log_prefix=sym)
                if df_rth.empty:
                    df = raw
                    rth_used[sym] = False
                else:
                    df = df_rth
                    rth_used[sym] = True
            else:
                df = raw
                rth_used[sym] = False

            if df.empty:
                print(f"[WARN] {sym}: empty after load; skipping")
                continue

            dfs[sym] = _prepare_symbol_df(df, sym, base)

        except Exception as e:
            alert.error(f"Loader/prep error for {sym}: {e}")
            continue

    if not dfs:
        print("[SKIP] All symbols empty after load/RTH; nothing to run for this shard/window.")
        run_meta = {
            "run_id": run_id,
            "time_utc": now_utc.isoformat(),
            "time_local_Pacific": now_utc.astimezone(pytz.timezone("America/Los_Angeles")).isoformat(),
            "args": vars(args),
            "params": asdict(base),
            "symbols": shard_syms,
            "start": args.start,
            "end": args.end,
            "source": args.source,
            "rth_only": args.rth_only,
            "initial_equity": args.initial_equity,
            "env": {"pandas": pd.__version__},
            "reason": "all_empty_post_load",
        }
        with open(os.path.join(run_dir, "run_meta.json"), "w") as f:
            json.dump(run_meta, f, indent=2)
        print(f"Done. Run folder: {run_dir}")
        return

    # 2) Window QC -------------------------------------------------------------
    train_m = 1 if args.fast else (args.train_months or 2)
    test_m  = args.test_months or 1

    per_counts, windows_count, test_months = _qc_train_test_counts(dfs, train_m, test_m, rth_used, qc_mode=getattr(args, "qc_mode", "common"))

    drop = [s for s,(trb, teb) in per_counts.items() if teb == 0]
    if args.qc_mode in ("off","union"):
        drop = []
    if drop:
        print(f"[INFO] Dropping symbols with zero test bars this window: {drop}")
        for s in drop:
            dfs.pop(s, None)

    if not dfs:
        print("[SKIP] No symbols with test bars after filters; skipping optimizer & sim for this shard/window.")
        run_meta = {
            "run_id": run_id,
            "time_utc": now_utc.isoformat(),
            "time_local_Pacific": now_utc.astimezone(pytz.timezone("America/Los_Angeles")).isoformat(),
            "args": vars(args),
            "params": asdict(base),
            "symbols": shard_syms,
            "kept_symbols": [],
            "start": args.start,
            "end": args.end,
            "source": args.source,
            "rth_only": args.rth_only,
            "initial_equity": args.initial_equity,
            "env": {"pandas": pd.__version__},
            "reason": "no_test_bars",
        }
        with open(os.path.join(run_dir, "run_meta.json"), "w") as f:
            json.dump(run_meta, f, indent=2)
        print(f"Done. Run folder: {run_dir}")
        return

    kept_syms = list(dfs.keys())

    # 3) Runtime estimate ------------------------------------------------------
    if args.no_opt:
        units = 1
    else:
        try:
            from optimizer import _full_grid
            combos_total = 1
            for _, vals in _full_grid().items():
                combos_total *= max(1, len(vals))
        except Exception:
            combos_total = 1000
        units = (args.max_combos or combos_total) * max(1, windows_count)

    pred_sec = _estimate_runtime_seconds(dfs, base, units=units)
    if pred_sec > args.warn_runtime_mins * 60:
        alert.warn(f"Predicted runtime ~ {pred_sec/60:.1f} min "
                   f"(units={units}, windows={windows_count}, symbols={len(dfs)}). Consider scaling out.")

    # 4) Optimizer -------------------------------------------------------------
    if not args.no_opt:
        wf = walk_forward(
            dfs_by_symbol=dfs,
            base_params=base,
            train_months=train_m,
            test_months=test_m,
            grid=None,
            max_combos=args.max_combos,
            reports_dir=run_dir,
            show_progress=True,
            progress_log=True,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            seed=args.seed,
        )
        save_results(wf, os.path.join(run_dir, "summary.csv"))

        # --- use-optimized(best_params) ---
        if getattr(args, "use_optimized", False):
            try:
                import ast, csv
                import pandas as _pd
                from dataclasses import fields as _fields
                _summ = _pd.read_csv(os.path.join(run_dir, "summary.csv"))
                _keys = ["test_total_pnl","test_pnl","objective","score","val_score","val_pnl","mean_test_pnl"]
                for _k in _keys:
                    if _k in _summ.columns:
                        _row = _summ.sort_values(_k, ascending=False).iloc[0]
                        break
                else:
                    _row = _summ.iloc[0]
                _bp_raw = _row.get("best_params", None)
                _bp_dict = {}
                if isinstance(_bp_raw, dict):
                    _bp_dict = _bp_raw
                elif isinstance(_bp_raw, str) and _bp_raw.strip():
                    try:
                        _bp_dict = json.loads(_bp_raw)
                    except Exception:
                        try:
                            _bp_dict = ast.literal_eval(_bp_raw)
                        except Exception:
                            _bp_dict = {}
                try:
                    with open(os.path.join(run_dir, "best_params.json"), "w") as _f:
                        json.dump(_bp_dict, _f, indent=2)
                except Exception:
                    pass

                _syn = {"z_th":"vwap_z","hold":"hold_minutes","hold_mins":"hold_minutes","timeout":"hold_minutes"}
                from dataclasses import fields as _fields2
                _field_types = {f.name: f.type for f in _fields2(Params)}
                _kw = {}
                for k,v in list(_bp_dict.items()):
                    name = _syn.get(k, k)
                    if name in _field_types:
                        t = _field_types[name]
                        try:
                            if t is bool:
                                v = bool(int(v)) if isinstance(v, str) else bool(v)
                            elif t in (int, float):
                                v = t(v)
                        except Exception:
                            pass
                        _kw[name] = v

                _before = asdict(base)
                if _kw:
                    base = Params(**{**_before, **_kw})
                    _changed = {k: (_before.get(k), _kw[k]) for k in _kw if _before.get(k) != _kw[k]}
                    print("[use-optimized] Applied best_params:", _changed if _changed else "(no material deltas)")
                    try:
                        with open(os.path.join(run_dir, "best_params_applied.json"), "w") as _f:
                            json.dump(_kw, _f, indent=2)
                    except Exception:
                        pass
                    try:
                        m = _pd.read_csv(os.path.join(run_dir, "metrics_summary.csv")).set_index("metric")["value"].to_dict() if os.path.exists(os.path.join(run_dir, "metrics_summary.csv")) else {}
                        row = {**{k:_kw.get(k) for k in sorted(_kw)}, **{'total_pnl': m.get('total_pnl'), 'hit_rate': m.get('hit_rate'), 'win_loss_ratio': m.get('win_loss_ratio')}}
                        with open(os.path.join(run_dir, "best_params_summary.csv"), "w", newline="") as f:
                            w = csv.DictWriter(f, fieldnames=list(row.keys()))
                            w.writeheader(); w.writerow(row)
                    except Exception:
                        pass
                else:
                    print("[use-optimized] best_params present but no fields matched Params; using base.")
            except Exception as _e:
                print(f"[WARN] use-optimized(best_params) failed: {_e}; continuing with base.")
        # --- /use-optimized(best_params) ---

    # 5) Baseline cross-sectional sim -----------------------------------------
    dfs_selected = annotate_candidates(dfs, base, top_k_per_min=getattr(base, "top_k_per_min", 5))

    combined_trades = []
    for sym, df_sym in dfs_selected.items():
        out = simulate_symbol(df_sym, base, initial_equity=args.initial_equity)
        t = out["trades"].copy()
        if not t.empty:
            t["symbol"] = sym
            combined_trades.append(t)

    if combined_trades:
        all_trades = pd.concat(combined_trades, ignore_index=True)
        all_trades.to_csv(os.path.join(run_dir, "trades.csv"), index=False)

    # 6) Save run metadata + report -------------------------------------------
    run_meta = {
        "run_id": run_id,
        "time_utc": now_utc.isoformat(),
        "time_local_Pacific": now_utc.astimezone(pytz.timezone("America/Los_Angeles")).isoformat(),
        "args": vars(args),
        "params": asdict(base),
        "symbols": shard_syms,
        "kept_symbols": kept_syms,
        "start": args.start,
        "end": args.end,
        "source": args.source,
        "rth_only": args.rth_only,
        "initial_equity": args.initial_equity,
        "env": {"pandas": pd.__version__},
        "windows_count": windows_count,
        "test_months": test_months,
    }
    with open(os.path.join(run_dir, "run_meta.json"), "w") as f:
        json.dump(run_meta, f, indent=2)

    if not args.no_report:
        outs = generate_quick_report(run_dir)
        print("Report files created:")
        for k, v in outs.items():
            print(f"  - {k}: {v}")

    print(f"Done. Run folder: {run_dir}")
    print("  - summary.csv (if optimizer ran)")
    print("  - trades.csv (if any)")
    print("  - run_meta.json")
    print("  - quick_report.pdf (unless --no-report)")
    # --- auto-zip parent when all shard folders exist ---
    try:
        if getattr(args, "auto-zip", False) or getattr(args, "auto_zip", False):
            if getattr(args, "run_root", None) and args.shard_index is not None and args.shard_count:
                expected = [os.path.join(_parent_dir, f"{args.run_root}_s{i}") for i in range(int(args.shard_count))]
                if all(os.path.isdir(d) for d in expected):
                    lock_path = os.path.join(_parent_dir, ".zip.lock")
                    try:
                        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                        os.close(fd)
                        zip_path = os.path.join(REPORTS_DIR, f"{args.run_root}.zip")
                        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                            for root, _, files in os.walk(_parent_dir):
                                for fn in files:
                                    # skip lock file and nested zips if any
                                    if fn.endswith(".lock") or (fn.endswith(".zip") and root == REPORTS_DIR):
                                        continue
                                    fp = os.path.join(root, fn)
                                    arc = os.path.relpath(fp, REPORTS_DIR)
                                    zf.write(fp, arcname=arc)
                        print(f"[bundle] Created parent zip: {zip_path}")
                    finally:
                        try: os.unlink(lock_path)
                        except: pass
    except Exception as _zip_e:
        print(f"[WARN] auto-zip failed: {_zip_e}")
    # --- /auto-zip ---



if __name__ == "__main__":
    main()