import argparse
import os
import json
import warnings
import time
from dataclasses import asdict
from datetime import datetime, timezone
import pytz

import numpy as np
import pandas as pd
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
from optimizer import walk_forward, save_results, month_splits  # month_splits for window count
from selector import annotate_candidates
from monitoring import AlertManager, RateLimitGuard

from reporter import generate_quick_report

warnings.filterwarnings("ignore", message="Converting to PeriodArray")
warnings.filterwarnings("ignore", message="DataFrameGroupBy.apply")


def _prepare_symbol_df(df: pd.DataFrame, symbol: str, base: Params) -> pd.DataFrame:
    df = df.copy()
    df["symbol"] = symbol
    df = df.sort_values("timestamp")
    df = add_intraday_features(df)  # rsi, sigma, vwap_dev, vwap
    df = minute_volume_profile_flag(df, multiple=3.0)              # vol_mult
    df = signal_gap_fade(df, gap_th=base.gap_th, hold_minutes=base.hold_minutes)  # gap_pct, ORH, sig_gap_fade
    df = signal_vwap_extreme(df, z_th=base.vwap_z, rsi_th=base.rsi_th)            # vwap_z, sig_vwap_extreme
    df = signal_late_blowoff(df, breakout_pct=base.breakout_pct)                  # sig_late_blowoff
    return df


def _load_symbol_data(symbol: str, start: str, end: str, source: str, rth_only: bool,
                      alp_guard: RateLimitGuard, pol_guard: RateLimitGuard, alert: AlertManager) -> pd.DataFrame:
    try:
        if source == "local":
            return load_local_parquet(symbol, start, end, data_dir=DATA_DIR)
        elif source == "alpaca":
            df = load_alpaca_minutes(symbol, start, end, rth_only=rth_only)
            alp_guard.record(1)
            return df
        elif source == "polygon":
            df = load_polygon_minutes(symbol, start, end, rth_only=rth_only)
            pol_guard.record(1)  # treat as ~1 logical request per symbol-range
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


def _make_run_dir(base_reports_dir: str, run_tag: str | None = None):
    now_utc = datetime.now(timezone.utc)
    run_id = now_utc.strftime("%Y%m%d_%H%M%SZ")
    if run_tag:
        run_id = f"{run_id}_{run_tag}"
    run_dir = os.path.join(base_reports_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, run_id, now_utc


def _count_windows(dfs: dict[str, pd.DataFrame], train_m: int, test_m: int) -> int:
    months = sorted(set().union(*[set(month_splits(df)) for df in dfs.values()]))
    return max(0, len(months) - train_m - test_m + 1)


def _estimate_runtime_seconds(dfs: dict[str, pd.DataFrame], base: Params, units: int) -> float:
    """Micro-benchmark simulate_symbol on a small slice and scale to 'units' of work."""
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


def main():
    ap = argparse.ArgumentParser(description="Exuberance Fade Backtester (Polygon/Alpaca)")
    ap.add_argument("--symbols", nargs="*", default=DEFAULT_SYMBOLS)
    ap.add_argument("--start", default="2025-01-02")
    ap.add_argument("--end",   default="2025-09-15")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--source", default="local", choices=["local","alpaca","polygon"])
    ap.add_argument("--rth-only", action="store_true", default=True)
    ap.add_argument("--initial_equity", type=float, default=100000.0)
    ap.add_argument("--shard-index", type=int, default=None, help="This shard's index, 0-based")
    ap.add_argument("--shard-count", type=int, default=None, help="Total number of shards")
    ap.add_argument("--seed", type=int, default=1337, help="Random shuffle seed for combo ordering")


    ap.add_argument("--no-opt", action="store_true")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--max-combos", type=int, default=None)
    ap.add_argument("--train-months", type=int, default=None)
    ap.add_argument("--test-months",  type=int, default=None)

    ap.add_argument("--write-cache", action="store_true")
    ap.add_argument("--run-tag", type=str, default=None)

    # plan/runtime alert knobs
    ap.add_argument("--alpaca-limit", type=int, default=200, help="calls/min (0 disables)")
    ap.add_argument("--polygon-limit", type=int, default=60, help="calls/min (0 disables)")
    ap.add_argument("--warn-runtime-mins", type=float, default=15.0)

    ap.add_argument("--no-report", action="store_true", help="Skip generating quick_report.pdf and summaries")

    args = ap.parse_args()

    os.makedirs(REPORTS_DIR, exist_ok=True)
    run_dir, run_id, now_utc = _make_run_dir(REPORTS_DIR, args.run_tag)
    alert = AlertManager(run_dir)

    alp_guard = RateLimitGuard("Alpaca", args.alpaca_limit if args.alpaca_limit > 0 else None, alert)
    pol_guard = RateLimitGuard("Polygon", args.polygon_limit if args.polygon_limit > 0 else None, alert)

    base = Params()  # uses JIT engine by default if available

    # 1) Load & prepare data
    dfs = {}
    for sym in tqdm(args.symbols, desc="Loading & preparing"):
        if args.demo:
            df = generate_synthetic_minutes(sym, args.start, args.end)
        else:
            df = _load_symbol_data(sym, args.start, args.end, args.source, args.rth_only, alp_guard, pol_guard, alert)

        if args.write_cache and (not args.demo) and args.source in ("alpaca","polygon"):
            os.makedirs(os.path.join(DATA_DIR, sym), exist_ok=True)
            out_fp = os.path.join(DATA_DIR, sym, f"{args.start}_to_{args.end}_{args.source}.parquet")
            df.to_parquet(out_fp, index=False)

        dfs[sym] = _prepare_symbol_df(df, sym, base)

    # 2) Correct runtime prediction (no scary warnings on --no-opt)
    train_m = 1 if args.fast else (args.train_months or 3)
    test_m  = args.test_months or 1
    windows_count = _count_windows(dfs, train_m, test_m)

    # work units: baseline = 1; optimizer ≈ (combos * windows) units
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
                   f"(units={units}, windows={windows_count}, symbols={len(dfs)}). Consider Runpod.")

    # 3) Walk-forward optimizer (optional)
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

    # 4) Baseline (non-optimized) run with cross-sectional selection
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

    # 5) Save run metadata
    run_meta = {
        "run_id": run_id,
        "time_utc": now_utc.isoformat(),
        "time_local_Pacific": now_utc.astimezone(pytz.timezone("America/Los_Angeles")).isoformat(),
        "args": vars(args),
        "params": asdict(base),
        "symbols": args.symbols,
        "start": args.start,
        "end": args.end,
        "source": args.source,
        "rth_only": args.rth_only,
        "initial_equity": args.initial_equity,
        "env": { "pandas": pd.__version__ }
    }
    with open(os.path.join(run_dir, "run_meta.json"), "w") as f:
        json.dump(run_meta, f, indent=2)

    if not args.no_report:
        outs = generate_quick_report(run_dir)
        print("Report files created:")
        for k, v in outs.items():
            print(f"  - {k}: {v}")

    print(f"Done. Run folder: {run_dir}")
    print("  - alerts.log (rate-limit & runtime warnings)")
    print("  - summary.csv (if optimizer ran)")
    print("  - trades.csv")
    print("  - run_meta.json")
    print("  - top_params_by_window.csv / top_params_overall*.csv (if optimizer ran)")


if __name__ == "__main__":
    main()
