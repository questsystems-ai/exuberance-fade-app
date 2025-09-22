
#!/usr/bin/env python3
# Exuberance Fade Backtester — cleaned, guarded, and defaults-overlayed
import argparse, os, json, warnings, time
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytz
from tqdm import tqdm

from config import DATA_DIR, REPORTS_DIR, DEFAULT_SYMBOLS
from data_ingest import (
    load_local_parquet,
    generate_synthetic_minutes,
    load_alpaca_minutes,
    load_polygon_minutes,
)
from signals import (
    add_intraday_features,
    signal_gap_fade,
    signal_vwap_extreme,
    signal_late_blowoff,
    minute_volume_profile_flag,
)
from backtest import Params, simulate_symbol
from optimizer import walk_forward, month_splits  # month_splits used for window count
from selector import annotate_candidates
from monitoring import AlertManager, RateLimitGuard
from reporter import generate_quick_report

warnings.filterwarnings("ignore", message="DataFrameGroupBy.apply")
warnings.filterwarnings("ignore", message="Converting to PeriodArray/Index representation will drop timezone information")

# -------------------- DEFAULTS_JSON overlay --------------------
def _load_defaults_json() -> dict:
    raw = os.getenv("DEFAULTS_JSON")
    if not raw:
        return {}
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return {k: v for k, v in obj.items() if k in Params.__annotations__}
    except Exception:
        pass
    return {}

def _apply_overlay(p: Params, overlay: dict) -> Params:
    base = asdict(p)
    for k, v in (overlay or {}).items():
        if k in base:
            base[k] = v
    return Params(**base)

# -------------------- timestamp helpers --------------------
def _resolve_ts_column(df: pd.DataFrame):
    candidates = ["ts", "timestamp", "time", "datetime", "date", "t", "bar_time"]
    tscol = next((c for c in candidates if c in df.columns), None)
    if tscol is None:
        for c in df.columns:
            try:
                if pd.api.types.is_datetime64_any_dtype(df[c]):
                    tscol = c; break
            except Exception:
                pass
    if tscol is None and str(df.index.dtype).startswith("datetime64"):
        df = df.reset_index().rename(columns={"index": "ts"}); tscol = "ts"
    if tscol is None:
        raise ValueError(f"Could not locate a timestamp column; cols={list(df.columns)[:12]}")
    df[tscol] = pd.to_datetime(df[tscol], utc=True, errors="coerce")
    return df, tscol

def _month_keys(df: pd.DataFrame, tscol: str) -> pd.Series:
    return (df[tscol].dt.tz_convert("UTC").dt.tz_localize(None).dt.to_period("M").astype(str))

# -------------------- RTH helper --------------------
def _apply_rth_et(df: pd.DataFrame, *, log_prefix: str) -> pd.DataFrame:
    df, ts = _resolve_ts_column(df)
    et = df[ts].dt.tz_convert("America/New_York")
    mins = et.dt.hour * 60 + et.dt.minute
    is_weekday = et.dt.dayofweek < 5
    rth_mask = is_weekday & (mins >= 570) & (mins <= 960)  # 09:30–16:00
    out = df.loc[rth_mask].copy()
    if out.empty:
        print(f"[WARN] {log_prefix}: empty after RTH; retrying without RTH")
    return out

# -------------------- loaders --------------------
def _load_symbol_data(symbol: str, start: str, end: str, source: str,
                      alp_guard: RateLimitGuard, pol_guard: RateLimitGuard,
                      alert: AlertManager) -> pd.DataFrame:
    try:
        if source == "local":
            return load_local_parquet(symbol, start, end, data_dir=DATA_DIR)
        elif source == "alpaca":
            df = load_alpaca_minutes(symbol, start, end, rth_only=False); alp_guard.record(1); return df
        elif source == "polygon":
            df = load_polygon_minutes(symbol, start, end, rth_only=False); pol_guard.record(1); return df
        else:
            raise ValueError(f"Unknown source '{source}'. Use one of: local|alpaca|polygon")
    except Exception as e:
        msg = str(e).lower()
        if "429" in msg or "rate limit" in msg or "too many requests" in msg:
            alert.error(f"{source} rate-limit while loading {symbol} {start}->{end}: {e}")
        else:
            alert.error(f"{source} loader error for {symbol} {start}->{end}: {e}")
        raise

# -------------------- shard slicing --------------------
def _normalize_symbols(sym_in) -> List[str]:
    if isinstance(sym_in, str):
        s = sym_in.strip(); return [t.strip() for t in (s.split(",") if "," in s else s.split()) if t.strip()]
    if isinstance(sym_in, (list, tuple)) and len(sym_in) == 1 and isinstance(sym_in[0], str):
        s = sym_in[0].strip(); return [t.strip() for t in (s.split(",") if "," in s else s.split()) if t.strip()]
    if isinstance(sym_in, (list, tuple)):
        return [str(x).strip() for x in sym_in if str(x).strip()]
    try: return list(sym_in)
    except Exception: return [str(sym_in)]

def symbols_for_shard(all_symbols, shard_index: int | None, shard_count: int | None) -> List[str]:
    full = _normalize_symbols(all_symbols)
    if not full: raise SystemExit("Error: parsed symbol list is empty.")
    if shard_index is None or shard_count is None or shard_count <= 1:
        out = full
    else:
        out = [s for idx, s in enumerate(full) if idx % shard_count == shard_index]
    print(f"[shard {shard_index}/{shard_count}] symbols: {out}")
    if len(out) == 0: raise SystemExit("Shard produced zero symbols. Check --shard-index/--shard-count and --symbols.")
    return out

# -------------------- feature prep --------------------
def _prepare_symbol_df(df: pd.DataFrame, symbol: str, base: Params) -> pd.DataFrame:
    df = df.copy(); df["symbol"] = symbol
    tscol = "timestamp" if "timestamp" in df.columns else df.columns[0]
    df = df.sort_values(tscol)
    df = add_intraday_features(df)
    df = minute_volume_profile_flag(df, multiple=3.0)
    # Apply signal thresholds from *base* (after overlay) so selection has flags to work with
    df = signal_gap_fade(df, gap_th=base.gap_th, hold_minutes=base.hold_minutes)
    df = signal_vwap_extreme(df, z_th=base.vwap_z, rsi_th=base.rsi_th)
    df = signal_late_blowoff(df, breakout_pct=base.breakout_pct)
    return df

# -------------------- run dir/meta --------------------
def _make_run_dir(base_reports_dir: str, run_tag: str | None = None):
    now_utc = datetime.now(timezone.utc)
    run_id = now_utc.strftime("%Y%m%d_%H%M%SZ")
    if run_tag: run_id = f"{run_id}_{run_tag}"
    run_dir = os.path.join(base_reports_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, run_id, now_utc

# -------------------- window utilities --------------------
def _count_windows(dfs: Dict[str, pd.DataFrame], train_m: int, test_m: int) -> int:
    months = sorted(set().union(*[set(month_splits(df)) for df in dfs.values()])) if dfs else []
    return max(0, len(months) - train_m - test_m + 1)

def _estimate_runtime_seconds(dfs: Dict[str, pd.DataFrame], base: Params, units: int) -> float:
    if not dfs: return 0.0
    sample = next((df for df in dfs.values() if len(df) > 0), None)
    if sample is None: return 0.0
    n = min(4000, len(sample)); mini = sample.head(n)
    t0 = time.time(); simulate_symbol(mini, base, initial_equity=100_000.0); dt = max(1e-6, time.time() - t0)
    sec_per_row = dt / n; total_rows = sum(len(df) for df in dfs.values())
    return sec_per_row * total_rows * max(1, units)

# -------------------- QC logging --------------------
def _qc_train_test_counts(dfs: Dict[str, pd.DataFrame], train_m: int, test_m: int,
                          rth_used: Dict[str, bool]) -> Tuple[Dict[str, Tuple[int,int]], int, List[str]]:
    all_months = set(); per_sym_month = {}
    for sym, df in dfs.items():
        df, ts = _resolve_ts_column(df); months = _month_keys(df, ts)
        per_sym_month[sym] = months; all_months |= set(months.unique())
    sorted_months = sorted(all_months)
    windows_count = max(0, len(sorted_months) - train_m - test_m + 1)
    test_months = sorted_months[-test_m:] if len(sorted_months) >= test_m else []
    per_counts = {}
    for sym, df in dfs.items():
        if df.empty:
            per_counts[sym] = (0, 0); print(f"[QC] {sym}: train_bars=0 test_bars=0 (after {'RTH' if rth_used.get(sym, False) else 'ALL'})"); continue
        df, ts = _resolve_ts_column(df); m = _month_keys(df, ts)
        test_mask = m.isin(test_months); train_mask = m.isin(sorted_months[:-test_m]) if test_m > 0 else m.notna()
        train_bars = int(train_mask.sum()); test_bars = int(test_mask.sum())
        tag = "RTH" if rth_used.get(sym, False) else "ALL"
        print(f"[QC] {sym}: train_bars={train_bars} test_bars={test_bars} (after {tag})")
        per_counts[sym] = (train_bars, test_bars)
    return per_counts, windows_count, test_months

# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser(description="Exuberance Fade Backtester")
    ap.add_argument("--symbols", nargs="*", default=DEFAULT_SYMBOLS)
    ap.add_argument("--source", default="local", choices=["local", "alpaca", "polygon"])
    ap.add_argument("--start", default="2024-01-01"); ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--rth-only", action="store_true", default=False)
    ap.add_argument("--initial_equity", type=float, default=100000.0)
    ap.add_argument("--shard-index", type=int, default=None); ap.add_argument("--shard-count", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--no-opt", action="store_true"); ap.add_argument("--fast", action="store_true")
    ap.add_argument("--max-combos", type=int, default=None)
    ap.add_argument("--train-months", type=int, default=None); ap.add_argument("--test-months", type=int, default=None)
    ap.add_argument("--write-cache", action="store_true"); ap.add_argument("--run-tag", type=str, default=None)
    ap.add_argument("--alpaca-limit", type=int, default=200); ap.add_argument("--polygon-limit", type=int, default=60)
    ap.add_argument("--warn-runtime-mins", type=float, default=15.0); ap.add_argument("--no-report", action="store_true")
    args = ap.parse_args()

    os.makedirs(REPORTS_DIR, exist_ok=True)
    run_dir, run_id, now_utc = _make_run_dir(REPORTS_DIR, args.run_tag)
    alert = AlertManager(run_dir)
    alp_guard = RateLimitGuard("Alpaca", args.alpaca_limit if args.alpaca_limit > 0 else None, alert)
    pol_guard = RateLimitGuard("Polygon", args.polygon_limit if args.polygon_limit > 0 else None, alert)

    shard_syms = symbols_for_shard(args.symbols, args.shard_index, args.shard_count)

    # --- Params with DEFAULTS_JSON overlay so signals aren't starved ---
    base = Params()
    overlay = _load_defaults_json()
    if overlay:
        base = _apply_overlay(base, overlay)
    # common safe fallback if someone sets a non-recognized entry window
    if getattr(base, "entry_window", None) not in ("am", "pm", "both"):
        try:
            base = _apply_overlay(base, {"entry_window": "both"})
        except Exception:
            pass

    # 1) Load & prepare
    dfs: Dict[str, pd.DataFrame] = {}
    rth_used: Dict[str, bool] = {}
    for sym in tqdm(shard_syms, desc="Loading & preparing"):
        try:
            raw = generate_synthetic_minutes(sym, args.start, args.end) if False else \
                  _load_symbol_data(sym, args.start, args.end, args.source, alp_guard, pol_guard, alert)
            if args.rth_only:
                df_rth = _apply_rth_et(raw, log_prefix=sym)
                df = raw if df_rth.empty else df_rth
                rth_used[sym] = not df_rth.empty
            else:
                df = raw; rth_used[sym] = False
            if df.empty:
                print(f"[WARN] {sym}: empty after load; skipping"); continue
            dfs[sym] = _prepare_symbol_df(df, sym, base)
        except Exception as e:
            alert.error(f"Loader/prep error for {sym}: {e}"); continue

    if not dfs:
        print("[SKIP] All symbols empty after load/RTH; nothing to run for this shard/window.")
        run_meta = {
            "run_id": run_id, "time_utc": now_utc.isoformat(),
            "time_local_Pacific": now_utc.astimezone(pytz.timezone("America/Los_Angeles")).isoformat(),
            "args": vars(args), "params": asdict(base), "symbols": shard_syms,
            "start": args.start, "end": args.end, "source": args.source, "rth_only": args.rth_only,
            "initial_equity": args.initial_equity, "env": {"pandas": pd.__version__},
            "reason": "all_empty_post_load",
        }
        with open(os.path.join(run_dir, "run_meta.json"), "w") as f: json.dump(run_meta, f, indent=2)
        print(f"Done. Run folder: {run_dir}"); return

    # 2) Window QC
    train_m = 1 if args.fast else (args.train_months or 2); test_m  = args.test_months or 1
    per_counts, windows_count, test_months = _qc_train_test_counts(dfs, train_m, test_m, rth_used)

    # Drop symbols with zero test bars
    drop = [s for s,(trb, teb) in per_counts.items() if teb == 0]
    if drop:
        print(f"[INFO] Dropping symbols with zero test bars this window: {drop}")
        for s in drop: dfs.pop(s, None)
    if not dfs:
        print("[SKIP] No symbols with test bars after filters; skipping optimizer & sim for this shard/window.")
        run_meta = {
            "run_id": run_id, "time_utc": now_utc.isoformat(),
            "time_local_Pacific": now_utc.astimezone(pytz.timezone("America/Los_Angeles")).isoformat(),
            "args": vars(args), "params": asdict(base),
            "symbols": shard_syms, "kept_symbols": [], "start": args.start, "end": args.end,
            "source": args.source, "rth_only": args.rth_only, "initial_equity": args.initial_equity,
            "env": {"pandas": pd.__version__}, "reason": "no_test_bars",
        }
        with open(os.path.join(run_dir, "run_meta.json"), "w") as f: json.dump(run_meta, f, indent=2)
        print(f"Done. Run folder: {run_dir}"); return

    kept_syms = list(dfs.keys())

    # 3) Runtime estimate
    try:
        from optimizer import _full_grid
        combos_total = 1
        for _, vals in _full_grid().items():
            combos_total *= max(1, len(vals))
    except Exception:
        combos_total = 1000
    units = ((args.max_combos or combos_total) * max(1, windows_count)) if not args.no_opt else 1
    pred_sec = _estimate_runtime_seconds(dfs, base, units=units)
    if pred_sec > args.warn_runtime_mins * 60:
        alert.warn(f"Predicted runtime ~ {pred_sec/60:.1f} min (units={units}, windows={windows_count}, symbols={len(dfs)}). Consider scaling out.")

    # 4) Optimizer
    if not args.no_opt:
        wf = walk_forward(
            dfs_by_symbol=dfs, base_params=base,
            train_months=train_m, test_months=test_m,
            grid=None, max_combos=args.max_combos,
            reports_dir=run_dir, show_progress=True, progress_log=True,
            shard_index=args.shard_index, shard_count=args.shard_count, seed=args.seed,
        )
        # Safe write guard
        if wf is None or getattr(wf, "empty", False):
            print("[SAFE-GUARD] walk_forward returned no rows; skipping summary.csv write.")
        else:
            wf.to_csv(os.path.join(run_dir, "summary.csv"), index=False)

    # 5) Baseline cross-sectional sim
    dfs_selected = annotate_candidates(dfs, base, top_k_per_min=getattr(base, "top_k_per_min", 5))
    combined_trades = []
    for sym, df_sym in dfs_selected.items():
        out = simulate_symbol(df_sym, base, initial_equity=args.initial_equity)
        t = out["trades"].copy()
        if not t.empty:
            t["symbol"] = sym
            entry_col = "entry_ts" if "entry_ts" in t.columns else ("entry" if "entry" in t.columns else None)
            if entry_col:
                t[entry_col] = pd.to_datetime(t[entry_col], utc=True, errors="coerce")
                entry_m = t[entry_col].dt.tz_convert("UTC").dt.to_period("M").astype(str)
                t = t.loc[entry_m.isin(test_months)].copy()
            if not t.empty: combined_trades.append(t)
    if combined_trades:
        all_trades = pd.concat(combined_trades, ignore_index=True)
        all_trades.to_csv(os.path.join(run_dir, "trades.csv"), index=False)

    # 6) Save run metadata + report
    run_meta = {
        "run_id": run_id, "time_utc": now_utc.isoformat(),
        "time_local_Pacific": now_utc.astimezone(pytz.timezone("America/Los_Angeles")).isoformat(),
        "args": vars(args), "params": asdict(base),
        "symbols": shard_syms, "kept_symbols": kept_syms,
        "start": args.start, "end": args.end, "source": args.source, "rth_only": args.rth_only,
        "initial_equity": args.initial_equity, "env": {"pandas": pd.__version__},
        "windows_count": windows_count, "test_months": test_months,
    }
    with open(os.path.join(run_dir, "run_meta.json"), "w") as f: json.dump(run_meta, f, indent=2)

    if not args.no_report:
        outs = generate_quick_report(run_dir)
        print("Report files created:")
        for k, v in outs.items(): print(f"  - {k}: {v}")

    print(f"Done. Run folder: {run_dir}")
    print("  - summary.csv (if optimizer ran)")
    print("  - trades.csv (if any)")
    print("  - run_meta.json")
    print("  - quick_report.pdf (unless --no-report)")

if __name__ == "__main__":
    main()
