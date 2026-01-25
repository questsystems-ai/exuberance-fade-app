# tools/polygon_smoke.py

import os
import sys
import pandas as pd

# Allow running from repo root without installing as a package
sys.path.append(".")

from data_ingest import load_polygon_minutes_multi  # noqa: E402

def main():
    symbols = ["MSFT", "AAPL", "NVDA", "AMZN"]
    start = "2025-06-01"
    end   = "2025-06-07"

    os.environ.setdefault("POLYGON_API_KEY", os.getenv("POLYGON_API_KEY", ""))  # ensure present

    dfs = load_polygon_minutes_multi(
        symbols,
        start,
        end,
        rth_only=True,
        adjusted=True,
        max_concurrency=6,
        reqs_per_min=100,  # soft clamp; adjust if you see 429s
    )

    for sym, df in dfs.items():
        n = 0 if df is None else len(df)
        f = df.iloc[0]["timestamp"].isoformat() if n else "-"
        l = df.iloc[-1]["timestamp"].isoformat() if n else "-"
        print(f"{sym}: rows={n} first={f} last={l}")

if __name__ == "__main__":
    main()
