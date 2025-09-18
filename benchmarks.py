
"""
benchmarks.py
Compare your backtest equity to benchmarks (SPY/QQQ) and optional local Cboe indices (BXM/BXY/PUT).
Saves metrics tables and charts into a timestamped subfolder of reports/ (or a folder you pass).
"""

import os
import json
from dataclasses import dataclass
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import REPORTS_DIR, POLYGON_API_KEY, ALPACA_KEY_ID, ALPACA_SECRET_KEY
from data_ingest import load_polygon_minutes, load_alpaca_minutes


@dataclass
class BenchConfig:
    source: str = "alpaca"  # "alpaca" | "polygon" | "local"
    tickers: List[str] = None  # e.g., ["SPY","QQQ"]
    start: str = "2025-06-01"
    end: str = "2025-08-31"
    run_dir: Optional[str] = None  # if None, create under REPORTS_DIR using current time
    initial_equity: float = 100000.0
    trades_csv: str = "trades.csv"  # relative to run_dir
    # Optional local CSVs for Cboe indices (columns: date, total_return or close)
    local_indices: Dict[str, str] = None  # {"BXM": "data/benchmarks/BXM.csv", ...}


def _ensure_run_dir(run_dir: Optional[str]) -> str:
    if run_dir:
        os.makedirs(run_dir, exist_ok=True)
        return run_dir
    import datetime, pytz
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    path = os.path.join(REPORTS_DIR, f"benchmarks_{now}")
    os.makedirs(path, exist_ok=True)
    return path


def _load_bars(ticker: str, start: str, end: str, source: str) -> pd.DataFrame:
    if source == "alpaca":
        return load_alpaca_minutes(ticker, start, end, rth_only=True)
    elif source == "polygon":
        return load_polygon_minutes(ticker, start, end, rth_only=True)
    else:
        # For "local", expect a CSV under data/benchmarks/<ticker>.csv with columns [timestamp, open, high, low, close]
        fp = os.path.join("data", "benchmarks", f"{ticker}.csv")
        if not os.path.isfile(fp):
            raise FileNotFoundError(f"Local benchmark file not found: {fp}")
        df = pd.read_csv(fp, parse_dates=["timestamp"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df[["timestamp","open","high","low","close"]]


def _daily_from_minutes(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["date","close"])
    # Resample to daily close in ET (to align with your equity daily step)
    df = df.copy()
    ts_et = df["timestamp"].dt.tz_convert("US/Eastern")
    df["date"] = ts_et.dt.date
    daily = df.groupby("date")["close"].last().reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    return daily


def _equity_from_trades(trades_csv: str, initial_equity: float) -> pd.DataFrame:
    t = pd.read_csv(trades_csv, parse_dates=["exit_ts"])
    if t.empty:
        return pd.DataFrame({"date": [], "equity": []})
    t["date"] = t["exit_ts"].dt.tz_convert("US/Eastern").dt.normalize()
    daily_pnl = t.groupby("date")["pnl"].sum().sort_index()
    eq = daily_pnl.cumsum() + initial_equity
    out = eq.reset_index()
    out.columns = ["date","equity"]
    return out


def _metrics_from_returns(ret: pd.Series) -> dict:
    if len(ret) < 2:
        return {"days": len(ret), "cagr": 0.0, "vol_ann": 0.0, "sharpe": 0.0, "sortino": 0.0,
                "max_dd": 0.0, "calmar": 0.0, "profit_factor": 0.0}
    days = len(ret)
    mean = ret.mean()
    std = ret.std(ddof=0)
    ann_ret = (1+mean)**252 - 1
    ann_vol = std * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 1e-12 else 0.0
    downside = ret[ret < 0]
    downside_std = downside.std(ddof=0) * np.sqrt(252) if len(downside) else 0.0
    sortino = ann_ret / downside_std if downside_std > 1e-12 else 0.0
    # max drawdown from cumulative
    cum = (1+ret).cumprod()
    peak = cum.cummax()
    dd = (cum/peak - 1.0).min()
    calmar = ann_ret / abs(dd) if abs(dd) > 1e-12 else 0.0
    profit_factor = (ret[ret > 0].sum()) / abs(ret[ret < 0].sum()) if (ret[ret < 0].sum() != 0) else np.inf
    return {"days": days, "cagr": ann_ret, "vol_ann": ann_vol, "sharpe": sharpe,
            "sortino": sortino, "max_dd": dd, "calmar": calmar, "profit_factor": profit_factor}


def run_benchmarks(cfg: BenchConfig):
    run_dir = _ensure_run_dir(cfg.run_dir)

    # 1) Load your equity (from trades.csv in the run_dir)
    trades_fp = cfg.trades_csv if os.path.isabs(cfg.trades_csv) else os.path.join(run_dir, cfg.trades_csv)
    equity_df = _equity_from_trades(trades_fp, cfg.initial_equity)
    if equity_df.empty:
        raise RuntimeError(f"No trades found at {trades_fp}. Run a backtest first.")
    equity_df["ret"] = equity_df["equity"].pct_change().fillna(0.0)

    # 2) Load benchmarks and compute daily close series
    if cfg.tickers is None:
        cfg.tickers = ["SPY", "QQQ"]

    bench_series = {}
    for tkr in cfg.tickers:
        bars = _load_bars(tkr, cfg.start, cfg.end, cfg.source)
        daily = _daily_from_minutes(bars)
        bench_series[tkr] = daily.set_index("date")["close"]

    bench_df = pd.DataFrame(bench_series)
    bench_df = bench_df.reindex(equity_df["date"]).dropna(how="all")
    # Normalize to returns
    bench_rets = bench_df.pct_change().fillna(0.0)

    # 3) Metrics
    our_metrics = _metrics_from_returns(equity_df.set_index("date")["ret"])
    metrics_rows = [{"series": "strategy", **our_metrics}]
    for tkr in bench_rets.columns:
        metrics_rows.append({"series": tkr, **_metrics_from_returns(bench_rets[tkr])})
    metrics = pd.DataFrame(metrics_rows)

    # 4) Charts
    charts_dir = os.path.join(run_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)

    # Equity vs SPY (indexed to 100)
    eq_norm = (equity_df.set_index("date")["equity"] / cfg.initial_equity) * 100.0
    if "SPY" in bench_df.columns:
        spy_norm = (bench_df["SPY"] / bench_df["SPY"].iloc[0]) * 100.0
        plt.figure()
        eq_norm.plot(label="Strategy (Equity=100)")
        spy_norm.plot(label="SPY (Price=100)")
        plt.title("Strategy vs SPY (Indexed)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, "equity_vs_spy.png"))
        plt.close()

    # Drawdown plot
    cum = (1 + equity_df["ret"]).cumprod()
    peak = cum.cummax()
    dd = cum / peak - 1.0
    plt.figure()
    dd.plot()
    plt.title("Strategy Drawdown")
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "drawdown.png"))
    plt.close()

    # 5) Save outputs
    metrics_path = os.path.join(run_dir, "benchmarks_metrics.csv")
    metrics.to_csv(metrics_path, index=False)

    print(f"Benchmarks written under: {run_dir}")
    print(f" - {metrics_path}")
    print(f" - charts/equity_vs_spy.png")
    print(f" - charts/drawdown.png")
    return metrics_path


if __name__ == "__main__":
    # Example CLI usage pattern (basic):
    #   python benchmarks.py --run-dir reports/<YOUR_RUN_ID> --source alpaca --start 2025-06-01 --end 2025-08-31
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="The run folder that contains trades.csv and run_meta.json")
    ap.add_argument("--source", default="alpaca", choices=["alpaca","polygon","local"])
    ap.add_argument("--tickers", nargs="*", default=["SPY","QQQ"])
    ap.add_argument("--start", default="2025-06-01")
    ap.add_argument("--end",   default="2025-08-31")
    ap.add_argument("--initial-equity", type=float, default=100000.0)
    args = ap.parse_args()

    cfg = BenchConfig(
        source=args.source,
        tickers=args.tickers,
        start=args.start,
        end=args.end,
        run_dir=args.run_dir,
        initial_equity=args.initial_equity,
        trades_csv="trades.csv",
        local_indices=None,
    )
    run_benchmarks(cfg)
