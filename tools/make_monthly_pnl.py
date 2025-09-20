#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd

def build_monthly(trades_csv: Path, shard_name: str):
    df = pd.read_csv(trades_csv)
    if "pnl" not in df.columns:
        raise ValueError(f"{trades_csv} missing 'pnl' column")
    # prefer exit_ts; fall back to entry_ts
    ts_col = "exit_ts" if "exit_ts" in df.columns else ("entry_ts" if "entry_ts" in df.columns else None)
    if ts_col is None:
        raise ValueError(f"{trades_csv} missing time columns (exit_ts/entry_ts)")

    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
    df["month"] = df[ts_col].dt.to_period("M").astype(str)

    monthly = (df.dropna(subset=["month"])
                 .groupby("month", dropna=True)["pnl"]
                 .agg(trades="count", pnl_sum="sum", pnl_mean="mean")
                 .reset_index())

    out_csv = trades_csv.parent / f"monthly_pnl_{shard_name}.csv"
    monthly.to_csv(out_csv, index=False)
    return out_csv

def main():
    ap = argparse.ArgumentParser(description="Rebuild per-shard monthly PnL from trades.csv")
    ap.add_argument("run_dirs", nargs="+", help="Shard report directories (each must contain trades.csv)")
    args = ap.parse_args()

    wrote = []
    for rd in args.run_dirs:
        d = Path(rd)
        t = d / "trades.csv"
        if not t.exists():
            print(f"[skip] {d} has no trades.csv")
            continue
        shard = d.name  # e.g., 20250918_..._s8_s3
        out = build_monthly(t, shard_name=shard)
        wrote.append(out)

    print(f"✅ wrote {len(wrote)} monthly files:")
    for w in wrote:
        print(" -", w)

if __name__ == "__main__":
    main()
