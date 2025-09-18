# backfill_alpaca.py
import os, pandas as pd
from pandas.tseries.offsets import MonthEnd
from typing import Iterable
from data_ingest import load_alpaca_minutes
from config import DATA_DIR

def backfill_month(symbol: str, year: int, month: int, rth_only=True):
    start = pd.Timestamp(f"{year}-{month:02d}-01", tz="UTC")
    end   = (start + MonthEnd(1)).normalize() + pd.Timedelta(hours=23, minutes=59)
    df = load_alpaca_minutes(symbol, start.isoformat(), end.isoformat(), rth_only=rth_only)
    os.makedirs(f"{DATA_DIR}/{symbol}", exist_ok=True)
    df.to_parquet(f"{DATA_DIR}/{symbol}/{year}-{month:02d}.parquet", index=False)
    print(symbol, year, month, df.shape)

def backfill_range(symbols: Iterable[str], start="2025-06-01", end="2025-08-31", rth_only=True):
    s = pd.Timestamp(start, tz="UTC").to_period("M")
    e = pd.Timestamp(end,   tz="UTC").to_period("M")
    months = list(pd.period_range(s, e, freq="M"))
    for sym in symbols:
        for p in months:
            backfill_month(sym, p.year, p.month, rth_only=rth_only)

if __name__ == "__main__":
    # Example universe:
    UNIVERSE = ["AAPL","MSFT","NVDA","AMD","TSLA"]
    backfill_range(UNIVERSE, start="2025-06-01", end="2025-08-31", rth_only=True)
