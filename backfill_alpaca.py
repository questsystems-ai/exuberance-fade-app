from __future__ import annotations
import argparse, os
import pandas as pd
from pathlib import Path
from typing import List
from data_ingest import load_alpaca_minutes

DATA_DIR = Path("data/minute_bars")

def save_month(df: pd.DataFrame, symbol: str, year: int, month: int):
    outdir = DATA_DIR / symbol
    outdir.mkdir(parents=True, exist_ok=True)
    fn = outdir / f"{year:04d}-{month:02d}.parquet"
    if not df.empty:
        df.to_parquet(fn, index=False)

def backfill_symbol(symbol: str, start: str, end: str, rth_only: bool = True):
    # fetch once; write per-month parquet shards
    df = load_alpaca_minutes(symbol, start, end, rth_only=rth_only)
    if df is None or df.empty:
        print(f"{symbol}: no data returned.")
        return
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    gb = df.groupby([df["timestamp"].dt.year, df["timestamp"].dt.month], sort=True)
    for (y, m), chunk in gb:
        save_month(chunk, symbol, int(y), int(m))
        print(f"{symbol} {y} {m} ({len(chunk)}, {len(chunk.columns)})")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", help="single symbol")
    p.add_argument("--symbols", nargs="*", help="space-separated symbols")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--rth-only", action="store_true")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    syms: List[str] = []
    if args.symbol: syms.append(args.symbol)
    if args.symbols: syms.extend(args.symbols)
    if not syms:
        raise SystemExit("Provide --symbol TICKER or --symbols T1 T2 ...")
    for s in syms:
        backfill_symbol(s.upper(), args.start, args.end, rth_only=args.rth_only)
