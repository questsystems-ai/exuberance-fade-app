# reporter.py
# Builds a compact PDF + machine-readable summaries for a finished run folder.
from __future__ import annotations

import json
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# Headless PDF rendering
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def _fmt_float(x, nd=2):
    try:
        return f"{float(x):,.{nd}f}"
    except Exception:
        return str(x)


def generate_quick_report(run_dir: str) -> dict:
    """
    Creates:
      quick_report.pdf
      metrics_summary.csv / .json
      monthly_pnl.csv
      top5_trades.csv
      worst5_trades.csv
      bot_summary.json
    Returns dict of output paths (strings).
    """
    rd = Path(run_dir)
    trades_fp   = rd / "trades.csv"
    run_meta_fp = rd / "run_meta.json"

    outputs = {}

    # Load inputs
    if not trades_fp.exists():
        # Nothing to report; emit a tiny bot file and return
        bot_json = rd / "bot_summary.json"
        with open(bot_json, "w") as f:
            json.dump({"run_id": rd.name, "n_trades": 0}, f, indent=2)
        return {"bot_summary": str(bot_json)}

    trades = pd.read_csv(trades_fp, parse_dates=["entry_ts", "exit_ts"], infer_datetime_format=True)
    meta = {}
    if run_meta_fp.exists():
        with open(run_meta_fp, "r") as f:
            meta = json.load(f)

    # --------- Metrics ----------
    n_trades = len(trades)
    total_pnl = float(trades["pnl"].sum()) if "pnl" in trades else 0.0
    hit_rate = float((trades["pnl"] > 0).mean()) if "pnl" in trades and n_trades else 0.0
    avg_win  = float(trades.loc[trades["pnl"] > 0, "pnl"].mean()) if "pnl" in trades and (trades["pnl"] > 0).any() else 0.0
    avg_loss = float(trades.loc[trades["pnl"] <= 0, "pnl"].mean()) if "pnl" in trades and (trades["pnl"] <= 0).any() else 0.0
    wl_ratio = (avg_win / abs(avg_loss)) if avg_loss < 0 else float("inf")

    metrics_df = pd.DataFrame({
        "metric": ["n_trades", "total_pnl", "hit_rate", "avg_win", "avg_loss", "win_loss_ratio"],
        "value":  [n_trades,   total_pnl,  hit_rate,   avg_win,   avg_loss,   wl_ratio]
    })

    # Monthly P&L (by exit date in ET)
    if trades["exit_ts"].dt.tz is not None:
        exit_month = trades["exit_ts"].dt.tz_convert("US/Eastern").dt.strftime("%Y-%m")
    else:
        exit_month = trades["exit_ts"].dt.tz_localize("UTC").dt.tz_convert("US/Eastern").dt.strftime("%Y-%m")

    monthly_df = (trades.assign(exit_month=exit_month)
                  .groupby("exit_month")["pnl"]
                  .agg(trades="count", pnl_sum="sum", pnl_mean="mean")
                  .reset_index()
                  .sort_values("exit_month"))

    top5_df   = trades.sort_values("pnl", ascending=False).head(5)
    worst5_df = trades.sort_values("pnl", ascending=True).head(5)

    # --------- Write machine-readable files ----------
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%SZ")
    # Save directly in run_dir (flat files so bots find them easily)
    metrics_csv = rd / "metrics_summary.csv"
    metrics_json = rd / "metrics_summary.json"
    monthly_csv = rd / "monthly_pnl.csv"
    top5_csv = rd / "top5_trades.csv"
    worst5_csv = rd / "worst5_trades.csv"
    bot_json = rd / "bot_summary.json"
    pdf_path = rd / "quick_report.pdf"

    metrics_df.to_csv(metrics_csv, index=False)
    with open(metrics_json, "w") as f:
        json.dump({
            "run_id": meta.get("run_id", rd.name),
            "source": (meta.get("args", {}).get("source")
                       if isinstance(meta.get("args"), dict) else meta.get("source")),
            "symbols": meta.get("symbols"),
            "start": meta.get("start"),
            "end": meta.get("end"),
            "initial_equity": meta.get("initial_equity"),
            "metrics": {r["metric"]: r["value"] for _, r in metrics_df.iterrows()},
        }, f, indent=2)
    monthly_df.to_csv(monthly_csv, index=False)
    top5_df.to_csv(top5_csv, index=False)
    worst5_df.to_csv(worst5_csv, index=False)

    # Tiny bot summary
    with open(bot_json, "w") as f:
        json.dump({
            "run_id": meta.get("run_id", rd.name),
            "symbols": meta.get("symbols"),
            "source": (meta.get("args", {}).get("source")
                       if isinstance(meta.get("args"), dict) else meta.get("source")),
            "start": meta.get("start"),
            "end": meta.get("end"),
            "n_trades": n_trades,
            "total_pnl": round(total_pnl, 4),
            "hit_rate": round(hit_rate, 6),
            "avg_win": round(avg_win, 4),
            "avg_loss": round(avg_loss, 4),
            "win_loss_ratio": (round(wl_ratio, 4) if np.isfinite(wl_ratio) else "inf"),
        }, f, indent=2)

    # --------- PDF (compact, simple tables) ----------
    with PdfPages(pdf_path) as pp:
        # Page 1: Headline
        fig = plt.figure(figsize=(8.5, 11))
        plt.axis("off")
        lines = [
            "Exuberance Fade — Quick Report",
            "",
            f"Run ID: {meta.get('run_id', rd.name)}",
            f"Source: {(meta.get('args', {}).get('source') if isinstance(meta.get('args'), dict) else meta.get('source', '(unknown)'))}",
            f"Symbols: {', '.join(meta.get('symbols', [])) if isinstance(meta.get('symbols'), list) else meta.get('symbols')}",
            f"Date Range (UTC): {meta.get('start', '?')} → {meta.get('end', '?')}",
            "",
            f"Trades: {n_trades}",
            f"Total P&L: {_fmt_float(total_pnl)}",
            f"Hit rate: {hit_rate:.2%}",
            f"Avg win: {_fmt_float(avg_win)}",
            f"Avg loss: {_fmt_float(avg_loss)}",
            f"Win/Loss ratio: {_fmt_float(wl_ratio)}" if np.isfinite(wl_ratio) else "Win/Loss ratio: ∞",
        ]
        plt.text(0.05, 0.95, "\n".join(lines), va="top", ha="left")
        pp.savefig(fig); plt.close(fig)

        # Page 2: Metrics table
        fig = plt.figure(figsize=(8.5, 11)); plt.axis("off"); plt.title("Headline Metrics", pad=20)
        tbl = plt.table(cellText=[[m, _fmt_float(v, 4) if isinstance(v, float) else str(v)]
                                  for m, v in zip(metrics_df["metric"], metrics_df["value"])],
                        colLabels=["Metric", "Value"], loc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.2, 1.2)
        pp.savefig(fig); plt.close(fig)

        # Page 3: Monthly PnL
        fig = plt.figure(figsize=(8.5, 11)); plt.axis("off"); plt.title("Monthly P&L", pad=20)
        mt = monthly_df.copy()
        if not mt.empty:
            mt["pnl_sum"] = mt["pnl_sum"].map(lambda x: _fmt_float(x))
            mt["pnl_mean"] = mt["pnl_mean"].map(lambda x: _fmt_float(x))
            mt["trades"] = mt["trades"].astype(int)
            tbl = plt.table(cellText=mt.values, colLabels=mt.columns.tolist(), loc="center")
            tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.0, 1.0)
        pp.savefig(fig); plt.close(fig)

        # Page 4: Top 5 trades
        fig = plt.figure(figsize=(8.5, 11)); plt.axis("off"); plt.title("Top 5 Trades by P&L", pad=20)
        t5 = top5_df[["symbol","entry_ts","exit_ts","entry_price","exit_price","qty","pnl","exit_reason"]].copy()
        for c in ["entry_price","exit_price","qty","pnl"]:
            t5[c] = t5[c].map(lambda x: _fmt_float(x))
        if not t5.empty:
            tbl = plt.table(cellText=t5.values, colLabels=t5.columns.tolist(), loc="center")
            tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1.0, 1.0)
        pp.savefig(fig); plt.close(fig)

        # Page 5: Worst 5 trades
        fig = plt.figure(figsize=(8.5, 11)); plt.axis("off"); plt.title("Worst 5 Trades by P&L", pad=20)
        w5 = worst5_df[["symbol","entry_ts","exit_ts","entry_price","exit_price","qty","pnl","exit_reason"]].copy()
        for c in ["entry_price","exit_price","qty","pnl"]:
            w5[c] = w5[c].map(lambda x: _fmt_float(x))
        if not w5.empty:
            tbl = plt.table(cellText=w5.values, colLabels=w5.columns.tolist(), loc="center")
            tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1.0, 1.0)
        pp.savefig(fig); plt.close(fig)

    outputs.update({
        "pdf": str(pdf_path),
        "metrics_csv": str(metrics_csv),
        "metrics_json": str(metrics_json),
        "monthly_csv": str(monthly_csv),
        "top5_csv": str(top5_csv),
        "worst5_csv": str(worst5_csv),
        "bot_summary": str(bot_json),
    })
    return outputs
