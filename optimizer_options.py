#!/usr/bin/env python3
import argparse, os, json
from dataclasses import asdict
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from tqdm import tqdm

from run_backtest_polygon_options import OptParams, simulate_covered_calls, _load_underlying, _load_earnings_calendar

def month_keys(ts: pd.Series) -> pd.Series:
    return ts.dt.tz_convert("UTC").dt.tz_localize(None).dt.to_period("M").astype(str)

def month_range_strs(start_month: str, n: int) -> List[str]:
    p = pd.Period(start_month)
    return [str(p + i) for i in range(n)]

def full_grid() -> Dict[str, List[Any]]:
    return {
        "strike_otm_pct": [0.03, 0.05, 0.07, 0.10],
        "min_premium_yield": [0.0075, 0.01, 0.015],
        "dte_target": [5],
        "allocation_pct": [0.25, 0.5, 0.75],
        "prefer_assignment": [True, False],
        "take_profit_prem_decay": [0.6, 0.8, 0.9],
        "iv_rank_min": [0.4, 0.6, 0.7],
    }

def _symbol_lifespan(df: pd.DataFrame) -> Tuple[pd.Timestamp, pd.Timestamp]:
    t0 = df["timestamp"].min().tz_convert("UTC")
    t1 = df["timestamp"].max().tz_convert("UTC")
    return t0, t1

def _window_bounds_from_months(months: List[str]) -> Tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Period(months[0]).to_timestamp(how="start").tz_localize("UTC")
    end = (pd.Period(months[-1]) + 1).to_timestamp(how="start").tz_localize("UTC")
    return start, end

def walk_forward(dfs: Dict[str, pd.DataFrame], base: OptParams, train_m: int, test_m: int,
                 grid: Optional[Dict[str, List[Any]]] = None, max_combos: Optional[int] = None,
                 earnings_map: Optional[Dict[str, List[pd.Timestamp]]] = None,
                 per_symbol_out: Optional[str] = None):
    import itertools
    grid = grid or full_grid()
    keys, vals = zip(*grid.items())
    combos = [dict(zip(keys, v)) for v in itertools.product(*vals)]
    if max_combos is not None:
        combos = combos[:max_combos]

    all_months = sorted(set().union(*[set(month_keys(df["timestamp"])) for df in dfs.values()]))
    n_windows = max(0, len(all_months) - train_m - test_m + 1)
    rng = np.random.default_rng(base.seed)

    lifespans = {}
    for sym, df in dfs.items():
        if df is None or df.empty:
            continue
        t0, t1 = _symbol_lifespan(df)
        lifespans[sym] = {"first_seen": t0, "last_seen": t1}

    per_symbol_rows: List[Dict[str, Any]] = []

    rows = []
    for wi in tqdm(range(n_windows), desc="WF windows"):
        train = all_months[wi:wi+train_m]
        test  = all_months[wi+train_m:wi+train_m+test_m]

        train_start, train_end = _window_bounds_from_months(train)
        test_start, test_end   = _window_bounds_from_months(test)

        def alive(sym: str, start: pd.Timestamp, end: pd.Timestamp) -> bool:
            ls = lifespans.get(sym)
            if not ls: return False
            return not (ls["last_seen"] <= start or ls["first_seen"] >= end)

        best = None; best_score = -1e18
        for combo in combos:
            p = OptParams(**{**asdict(base), **combo})
            pnl, dd = 0.0, 0.0
            for sym, df in dfs.items():
                if df is None or df.empty or not alive(sym, train_start, train_end):
                    continue
                sel = df[(df["timestamp"] >= train_start) & (df["timestamp"] < train_end)]
                if sel.empty:
                    continue
                out = simulate_covered_calls(sel, p, rng, symbol=sym, earnings_map=earnings_map)
                m = out["metrics"]; pnl += float(m.get("total_pnl",0)); dd = min(dd, float(m.get("max_dd",0)))
            score = pnl if dd > -0.2 else pnl - 1e6
            if score > best_score:
                best_score = score; best = combo

        p = OptParams(**{**asdict(base), **best})
        pnl_t, dd_t, trades_t = 0.0, 0.0, 0
        window_contrib: List[Dict[str, Any]] = []
        for sym, df in dfs.items():
            if df is None or df.empty or not alive(sym, test_start, test_end):
                continue
            sel = df[(df["timestamp"] >= test_start) & (df["timestamp"] < test_end)]
            if sel.empty: 
                continue
            out = simulate_covered_calls(sel, p, rng, symbol=sym, earnings_map=earnings_map)
            m = out["metrics"]
            pnl_t += float(m.get("total_pnl",0)); dd_t = min(dd_t, float(m.get("max_dd",0)))
            trades = len(out["trades"])
            trades_t += trades
            window_contrib.append({
                "window": f"{test[0]}->{test[-1]}",
                "symbol": sym,
                "test_total_pnl": float(m.get("total_pnl",0)),
                "test_max_dd": float(m.get("max_dd",0) or 0.0),
                "test_trades": trades,
                "test_cagr": float(m.get("cagr", float("nan"))),
                "test_sharpe": float(m.get("sharpe", float("nan")))
            })

        rows.append({
            "train_months": ",".join(train), "test_months": ",".join(test),
            "best_params": json.dumps(best), "train_score": best_score,
            "test_total_pnl": pnl_t, "test_max_dd": dd_t, "test_trades": trades_t
        })
        per_symbol_rows.extend(window_contrib)

    per_symbol_df = pd.DataFrame(per_symbol_rows)
    if not per_symbol_df.empty:
        agg = per_symbol_df.groupby("symbol").agg(
            total_test_pnl=("test_total_pnl","sum"),
            min_test_dd=("test_max_dd","min"),
            total_test_trades=("test_trades","sum"),
            windows_seen=("window","nunique"),
            mean_test_sharpe=("test_sharpe","mean"),
            median_test_sharpe=("test_sharpe","median"),
            mean_test_cagr=("test_cagr","mean"),
            median_test_cagr=("test_cagr","median")
        ).reset_index()
    else:
        agg = pd.DataFrame(columns=["symbol","total_test_pnl","min_test_dd","total_test_trades","windows_seen"])

    return pd.DataFrame(rows), agg, lifespans

def main():
    ap = argparse.ArgumentParser(description="Options WF Optimizer (IV-rank + earnings-aware + baskets + lifespans + per-symbol contrib)")
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--basket", type=str, default=None)
    ap.add_argument("--baskets_json", type=str, default="baskets.json")
    ap.add_argument("--source", default="polygon", choices=["local","alpaca","polygon"])
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--train-months", type=int, default=3)
    ap.add_argument("--test-months", type=int, default=1)
    ap.add_argument("--max-combos", type=int, default=None)
    ap.add_argument("--run-tag", type=str, default=None)
    ap.add_argument("--earnings_csv", type=str, default=None)
    args = ap.parse_args()

    os.makedirs("reports", exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    run_dir = os.path.join("reports", f"{now}_options_opt_{args.run_tag or 'run'}")
    os.makedirs(run_dir, exist_ok=True)

    # Resolve symbols
    symbols = args.symbols
    if args.basket:
        bpath = args.baskets_json or "baskets.json"
        if not os.path.exists(bpath):
            here = os.path.dirname(__file__)
            b2 = os.path.join(here, os.path.basename(bpath))
            bpath = b2 if os.path.exists(b2) else bpath
        with open(bpath, "r") as f:
            baskets = json.load(f)
        if args.basket not in baskets:
            raise KeyError(f"Basket '{args.basket}' not found in {bpath}")
        symbols = baskets[args.basket]
        print(f"Using basket '{args.basket}':", " ".join(symbols))
    if not symbols:
        raise SystemExit("No symbols specified. Use --symbols ... or --basket ...")

    # Load underlying + capture lifespans implicitly via dfs
    dfs = {}
    dummy_alert = type("A", (), {"error": print})
    dummy_guard = type("G", (), {"record": lambda *a, **k: None})
    for s in tqdm(symbols, desc="Load underlying"):
        try:
            dfs[s] = _load_underlying(s, args.start, args.end, args.source, dummy_guard, dummy_guard, dummy_alert)
        except Exception as e:
            print("load error", s, e)

    base = OptParams()
    earnings_map = _load_earnings_calendar(args.earnings_csv) if args.earnings_csv else {}

    wf_df, contrib_df, lifespans = walk_forward(
        dfs, base, args.train_months, args.test_months, max_combos=args.max_combos, earnings_map=earnings_map
    )

    # Write outputs
    out_summary = os.path.join(run_dir, "summary.csv")
    out_contrib = os.path.join(run_dir, "per_symbol_contrib.csv")
    out_life    = os.path.join(run_dir, "lifespans.csv")

    wf_df.to_csv(out_summary, index=False)
    contrib_df.to_csv(out_contrib, index=False)

    life_rows = [{"symbol": s, "first_seen": v["first_seen"].isoformat(), "last_seen": v["last_seen"].isoformat()} for s,v in lifespans.items()]
    pd.DataFrame(life_rows).to_csv(out_life, index=False)

    print("Done. Run folder:", run_dir)
    print("  -", out_summary)
    print("  -", out_contrib)
    print("  -", out_life)

if __name__ == "__main__":
    main()
