
# report_combine.py
# Combine multiple run folders (Polygon/Alpaca/etc.) into one comparison report.
# Usage:
#   python report_combine.py --runs reports/<RUN1> reports/<RUN2> [--out reports/combined_<stamp>]
#
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def _fmt(x, nd=2):
    try:
        return f"{float(x):,.{nd}f}"
    except Exception:
        return str(x)


def _load_run(run_dir: str) -> dict:
    rd = Path(run_dir)
    meta = {}
    if (rd / "run_meta.json").exists():
        with open(rd / "run_meta.json", "r") as f:
            meta = json.load(f)
    trades = pd.read_csv(rd / "trades.csv", parse_dates=["entry_ts","exit_ts"], infer_datetime_format=True)
    # Metrics
    n = len(trades)
    total_pnl = float(trades["pnl"].sum())
    hit = float((trades["pnl"] > 0).mean()) if n else 0.0
    avg_win = float(trades.loc[trades["pnl"]>0, "pnl"].mean()) if (trades["pnl"]>0).any() else 0.0
    avg_loss = float(trades.loc[trades["pnl"]<=0, "pnl"].mean()) if (trades["pnl"]<=0).any() else 0.0
    wl = (avg_win/abs(avg_loss)) if avg_loss<0 else float("inf")

    # Daily equity
    if trades["exit_ts"].dt.tz is not None:
        d = trades["exit_ts"].dt.tz_convert("US/Eastern").dt.normalize()
    else:
        d = trades["exit_ts"].dt.tz_localize("UTC").dt.tz_convert("US/Eastern").dt.normalize()
    daily_pnl = trades.assign(date=d).groupby("date")["pnl"].sum().sort_index()
    initial_equity = float(meta.get("initial_equity", 100000.0))
    equity = daily_pnl.cumsum() + initial_equity
    rets = equity.pct_change().fillna(0.0)
    vol = rets.std(ddof=0) * np.sqrt(252)
    sharpe = (rets.mean()*252)/vol if vol>1e-12 else 0.0
    peak = equity.cummax()
    dd = (equity/peak - 1.0).min()

    # Monthly
    month = d.dt.strftime("%Y-%m")
    monthly = trades.assign(exit_month=month).groupby("exit_month")["pnl"].agg(trades="count", pnl_sum="sum").reset_index()

    label = meta.get("args", {}).get("source") if isinstance(meta.get("args"), dict) else meta.get("source", rd.name)
    if label is None:
        label = rd.name

    return {
        "dir": str(rd),
        "label": label,
        "symbols": meta.get("symbols"),
        "start": meta.get("start"),
        "end": meta.get("end"),
        "initial_equity": initial_equity,
        "metrics": {
            "n_trades": n,
            "total_pnl": total_pnl,
            "hit_rate": hit,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "win_loss_ratio": (wl if np.isfinite(wl) else "inf"),
            "final_equity": float(equity.iloc[-1]) if len(equity) else initial_equity,
            "sharpe": float(sharpe),
            "max_dd": float(dd),
        },
        "equity": equity,
        "monthly": monthly,
    }


def combine_runs(run_dirs: list[str], out_dir: str | None = None) -> dict:
    runs = [_load_run(rd) for rd in run_dirs]
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%SZ")
    out_base = out_dir or f"reports/combined_{stamp}"
    od = Path(out_base)
    od.mkdir(parents=True, exist_ok=True)

    # Combined metrics table
    rows = []
    for r in runs:
        m = r["metrics"]
        rows.append({
            "label": r["label"],
            "dir": r["dir"],
            "n_trades": m["n_trades"],
            "total_pnl": m["total_pnl"],
            "hit_rate": m["hit_rate"],
            "avg_win": m["avg_win"],
            "avg_loss": m["avg_loss"],
            "win_loss_ratio": m["win_loss_ratio"],
            "final_equity": m["final_equity"],
            "sharpe": m["sharpe"],
            "max_dd": m["max_dd"],
            "start": r["start"],
            "end": r["end"],
            "symbols": ",".join(r["symbols"]) if isinstance(r["symbols"], list) else r["symbols"],
        })
    metrics_df = pd.DataFrame(rows)
    metrics_csv = od / "combined_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    # JSON summary
    best_pnl = metrics_df.sort_values("total_pnl", ascending=False).iloc[0]["label"] if not metrics_df.empty else None
    best_sharpe = metrics_df.sort_values("sharpe", ascending=False).iloc[0]["label"] if not metrics_df.empty else None
    bot_json = od / "combined_bot_summary.json"
    with open(bot_json, "w") as f:
        json.dump({
            "runs": [{"label": r["label"], "dir": r["dir"], "metrics": r["metrics"]} for r in runs],
            "best_total_pnl": best_pnl,
            "best_sharpe": best_sharpe,
        }, f, indent=2)

    # PDF
    pdf_path = od / "comparison_report.pdf"
    with PdfPages(pdf_path) as pp:
        # Page 1: metrics table
        fig = plt.figure(figsize=(11, 8.5)); plt.axis("off"); plt.title("Combined Metrics", pad=16)
        table_df = metrics_df.copy()
        # pretty-print some columns
        for c in ["total_pnl","avg_win","avg_loss","final_equity"]:
            table_df[c] = table_df[c].map(lambda x: _fmt(x))
        for c in ["hit_rate","sharpe","max_dd","win_loss_ratio"]:
            table_df[c] = table_df[c].map(lambda x: _fmt(x, 4))
        tbl = plt.table(cellText=table_df.values,
                        colLabels=table_df.columns.tolist(), loc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1.2, 1.2)
        pp.savefig(fig); plt.close(fig)

        # Page 2: equity curves
        fig = plt.figure(figsize=(11, 8.5))
        for r in runs:
            eq = r["equity"]
            eq_norm = (eq / r["initial_equity"]) * 100.0
            eq_norm.plot(label=f"{r['label']} (start=100)")
        plt.legend(); plt.title("Equity (Indexed to 100)"); plt.tight_layout()
        pp.savefig(fig); plt.close(fig)

        # Page 3: drawdowns
        fig = plt.figure(figsize=(11, 8.5))
        for r in runs:
            eq = r["equity"]
            cum = (eq/eq.cummax()) - 1.0
            cum.plot(label=f"{r['label']}")
        plt.legend(); plt.title("Drawdowns"); plt.tight_layout()
        pp.savefig(fig); plt.close(fig)

        # Page 4: monthly pnl comparison (stack tables vertically)
        fig = plt.figure(figsize=(11, 8.5)); plt.axis("off")
        plt.title("Monthly P&L by Run", pad=16)
        y = 0.9
        for r in runs:
            mt = r["monthly"].copy()
            if mt.empty:
                continue
            mt["pnl_sum"] = mt["pnl_sum"].map(lambda x: _fmt(x))
            mt["trades"] = mt["trades"].astype(int)
            plt.text(0.02, y, f"{r['label']}", fontsize=10, weight="bold")
            y -= 0.04
            tbl = plt.table(cellText=mt.values, colLabels=mt.columns.tolist(), loc="upper left", colWidths=[0.15,0.12,0.12])
            tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1.0, 1.0)
            y -= 0.30
        pp.savefig(fig); plt.close(fig)

    return {"pdf": str(pdf_path), "metrics_csv": str(metrics_csv), "bot_json": str(bot_json)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="Run folders to combine (e.g., reports/<RUN1> reports/<RUN2>)")
    ap.add_argument("--out", default=None, help="Output folder (default: reports/combined_<timestamp>)")
    args = ap.parse_args()

    outs = combine_runs(args.runs, args.out)
    print("Combined report created:")
    for k, v in outs.items():
        print(f"  - {k}: {v}")
