# data_ingest.py
# Fully-featured data ingestion utilities for the exuberance-fade backtester.
# Includes: local parquet, synthetic demo, Alpaca stocks (1m), Polygon stocks (1m),
# and Polygon options chain snapshots + greeks and contract aggregates.

from __future__ import annotations

import os
import glob
from dataclasses import asdict
from typing import Iterable, List, Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd

from utils import ensure_dt

# --- Config keys ---
from config import (
    DATA_DIR,
    POLYGON_API_KEY,
    ALPACA_KEY_ID,
    ALPACA_SECRET_KEY,
)

# Optional imports guarded so the file loads even if packages aren't installed yet.
try:
    # Alpaca (official SDK)
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame as AlpacaTimeFrame
except Exception:
    StockHistoricalDataClient = None  # type: ignore
    StockBarsRequest = None  # type: ignore
    AlpacaTimeFrame = None  # type: ignore

try:
    # Polygon (official API client)
    from polygon import RESTClient as PolygonRESTClient
except Exception:
    PolygonRESTClient = None  # type: ignore


# ---------------------------------------------------------------------
# Local Parquet
# ---------------------------------------------------------------------
def load_local_parquet(symbol: str, start: str, end: str, data_dir: str = DATA_DIR) -> pd.DataFrame:
    """
    Load locally cached minute bars for a symbol between [start, end].
    Expected partitioning: data/minute_bars/<symbol>/<YYYY-MM>.parquet
    """
    path = os.path.join(data_dir, symbol)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"No local data for {symbol} at {path}")
    frames: List[pd.DataFrame] = []
    for fp in sorted(glob.glob(os.path.join(path, "*.parquet"))):
        df = pd.read_parquet(fp)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No parquet files under {path}")
    df = pd.concat(frames, ignore_index=True)
    df = ensure_dt(df)  # make timestamp tz-aware UTC
    df = df[(df['timestamp'] >= pd.Timestamp(start, tz='UTC')) & (df['timestamp'] <= pd.Timestamp(end, tz='UTC'))]
    return df.sort_values('timestamp').reset_index(drop=True)


# ---------------------------------------------------------------------
# Synthetic data (for --demo)
# ---------------------------------------------------------------------
def generate_synthetic_minutes(symbol: str, start: str, end: str, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic 1-minute bars in US/Eastern market hours 09:30–16:00, then convert to UTC.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start=start, end=end, freq='min', tz='US/Eastern')
    # Keep regular market hours only (09:30-16:00)
    idx = idx[(idx.hour > 9) | ((idx.hour == 9) & (idx.minute >= 30))]
    idx = idx[(idx.hour < 16) | ((idx.hour == 16) & (idx.minute == 0))]

    price = 100.0
    rows: List[Tuple[pd.Timestamp, float, float, float, float, int]] = []
    for ts in idx:
        ret = rng.normal(0, 0.0008)
        price *= (1 + ret)
        high = price * (1 + abs(rng.normal(0, 0.0005)))
        low  = price * (1 - abs(rng.normal(0, 0.0005)))
        open_ = price / (1 + ret)
        close = price
        vol = max(1, int(rng.lognormal(mean=10, sigma=0.3)))
        rows.append((ts.tz_convert('UTC'), open_, high, low, close, vol))

    df = pd.DataFrame(rows, columns=['timestamp','open','high','low','close','volume'])
    return df


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _filter_rth_et(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only 09:30–16:00 US/Eastern bars (inclusive)."""
    if df.empty:
        return df
    ts_et = df["timestamp"].dt.tz_convert("US/Eastern")
    mod = ts_et.dt.hour * 60 + ts_et.dt.minute
    rth = (mod >= 9*60 + 30) & (mod <= 16*60)
    return df.loc[rth].reset_index(drop=True)


# ---------------------------------------------------------------------
# Alpaca Stocks (minute bars)
# ---------------------------------------------------------------------
def load_alpaca_minutes(symbol: str, start: str, end: str, rth_only: bool = True) -> pd.DataFrame:
    """
    Fetch 1-minute bars for [start, end] (UTC) from Alpaca Market Data and return:
      columns: timestamp(UTC tz-aware), open, high, low, close, volume
    Requires ALPACA_KEY_ID / ALPACA_SECRET_KEY in config.py and `alpaca-py` installed.
    """
    if StockHistoricalDataClient is None:
        raise ImportError("alpaca-py is not installed. pip install alpaca-py")

    client = StockHistoricalDataClient(ALPACA_KEY_ID or None, ALPACA_SECRET_KEY or None)
    req = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=AlpacaTimeFrame.Minute,
        start=pd.Timestamp(start, tz="UTC"),
        end=pd.Timestamp(end, tz="UTC"),
        adjustment=None,  # 'raw' equivalent
    )
    out = client.get_stock_bars(req)
    if out is None or out.df is None or out.df.empty:
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])

    df = out.df.reset_index()
    df = df[df["symbol"] == symbol].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df[["timestamp","open","high","low","close","volume"]].sort_values("timestamp").reset_index(drop=True)
    if rth_only:
        df = _filter_rth_et(df)
    return df


# ---------------------------------------------------------------------
# Polygon Stocks (minute bars via Aggregates v2)
# ---------------------------------------------------------------------
def load_polygon_minutes(symbol: str, start: str, end: str, adjusted: bool = True, rth_only: bool = True) -> pd.DataFrame:
    """
    Fetch 1-minute aggregates for [start, end] (UTC date strings ok) and return:
      columns: timestamp(UTC tz-aware), open, high, low, close, volume
    Requires POLYGON_API_KEY in config.py and `polygon-api-client` installed.
    """
    if PolygonRESTClient is None:
        raise ImportError("polygon-api-client is not installed. pip install polygon-api-client")

    client = PolygonRESTClient(api_key=POLYGON_API_KEY)

    # Polygon aggregates v2; client paginates for you
    rows: List[Dict[str, Any]] = []
    for a in client.list_aggs(
        ticker=symbol,
        multiplier=1,
        timespan="minute",
        from_=pd.Timestamp(start, tz="UTC").strftime("%Y-%m-%d"),
        to=pd.Timestamp(end,   tz="UTC").strftime("%Y-%m-%d"),
        adjusted=adjusted,
        sort="asc",
        limit=50_000,
    ):
        rows.append({
            "timestamp": pd.to_datetime(a.timestamp, unit="ms", utc=True),
            "open": float(a.open),
            "high": float(a.high),
            "low": float(a.low),
            "close": float(a.close),
            "volume": float(a.volume),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if rth_only:
        df = _filter_rth_et(df)
    return df.sort_values("timestamp").reset_index(drop=True)


# ---------------------------------------------------------------------
# Polygon Options: chain snapshot (with greeks) + helpers
# ---------------------------------------------------------------------
def polygon_options_chain_snapshot(
    underlying: str,
    contract_type: Optional[str] = None,   # "call" | "put" | None (both)
    min_dte: Optional[int] = None,
    max_dte: Optional[int] = None,
) -> pd.DataFrame:
    """
    Return a DataFrame of the current options chain snapshot for `underlying` with fields:
    ['ticker','type','strike','expiration_date','delta','gamma','theta','vega','iv',
     'bid','ask','mid','last_price','open_interest','underlying_price','updated_at']

    Notes:
    - Snapshot is real-time (point-in-time). Historical greeks generally require specialized datasets.
    - DTE filtering is applied in Python using expiration_date vs now (UTC).
    """
    if PolygonRESTClient is None:
        raise ImportError("polygon-api-client is not installed. pip install polygon-api-client")

    client = PolygonRESTClient(api_key=POLYGON_API_KEY)

    items = client.list_snapshot_options_chain(underlying)  # iterable of option snapshot items

    now_utc = pd.Timestamp.utcnow().tz_localize("UTC")
    rows: List[Dict[str, Any]] = []
    for o in items:
        # defensive getattr to handle schema variations
        det = getattr(o, "details", None)
        greeks = getattr(o, "greeks", None)
        last_q = getattr(o, "last_quote", None)
        last_t = getattr(o, "last_trade", None)
        day    = getattr(o, "day", None)

        tkr = getattr(det, "ticker", None)
        ctype = getattr(det, "contract_type", None)
        strike = getattr(det, "strike_price", None)
        exp = getattr(det, "expiration_date", None)
        if not tkr or not exp:
            continue

        # filter by contract_type if provided
        if contract_type and str(ctype).lower() != contract_type.lower():
            continue

        # DTE filter
        try:
            exp_ts = pd.Timestamp(exp).tz_localize("UTC")
            dte = (exp_ts - now_utc).days
        except Exception:
            dte = None

        if min_dte is not None and (dte is None or dte < min_dte):
            continue
        if max_dte is not None and (dte is None or dte > max_dte):
            continue

        # quotes/price
        bid = getattr(getattr(last_q, "bid", None), "price", None)
        ask = getattr(getattr(last_q, "ask", None), "price", None)
        mid = None
        if bid is not None and ask is not None:
            try:
                mid = (float(bid) + float(ask)) / 2.0
            except Exception:
                mid = None

        last_price = getattr(getattr(last_t, "price", None), "__float__", lambda: None)()
        if isinstance(last_price, (int, float)):
            last_price = float(last_price)
        else:
            last_price = None

        oi = getattr(day, "open_interest", None)
        und_px = getattr(getattr(o, "underlying_asset", None), "price", None)
        upd_at = getattr(o, "updated", None)

        rows.append({
            "ticker": tkr,
            "type": (ctype or "").lower(),
            "strike": float(strike) if strike is not None else None,
            "expiration_date": exp,
            "dte": dte,
            "delta": getattr(greeks, "delta", None),
            "gamma": getattr(greeks, "gamma", None),
            "theta": getattr(greeks, "theta", None),
            "vega": getattr(greeks, "vega", None),
            "iv": getattr(greeks, "implied_volatility", None),
            "bid": float(bid) if bid is not None else None,
            "ask": float(ask) if ask is not None else None,
            "mid": float(mid) if mid is not None else None,
            "last_price": last_price,
            "open_interest": float(oi) if oi is not None else None,
            "underlying_price": float(und_px) if und_px is not None else None,
            "updated_at": upd_at,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # normalize types
    df["expiration_date"] = pd.to_datetime(df["expiration_date"], utc=True, errors="coerce")
    df["updated_at"] = pd.to_datetime(df["updated_at"], utc=True, errors="coerce")
    return df.sort_values(["expiration_date","strike"]).reset_index(drop=True)


def choose_bear_call_spread_polygon(
    underlying: str,
    target_delta: float = 0.20,
    min_dte: int = 0,
    max_dte: int = 7,
) -> Optional[Dict[str, Any]]:
    """
    Choose a bear-call spread from Polygon snapshot:
      - Short call with |delta| closest to target_delta
      - Long call = next further OTM call in same expiry
    Returns dict with short/long legs (tickers, strikes, expiry, greeks, mid quotes).
    """
    chain = polygon_options_chain_snapshot(
        underlying=underlying,
        contract_type="call",
        min_dte=min_dte,
        max_dte=max_dte,
    )
    if chain.empty:
        return None

    # choose short leg: closest |delta| to target_delta, above-the-money strikes preferred
    sel = chain.dropna(subset=["delta","strike","expiration_date"]).copy()
    if sel.empty:
        return None

    sel["delta_abs_diff"] = (sel["delta"].abs() - target_delta).abs()
    sel = sel.sort_values(["delta_abs_diff","expiration_date","strike"])
    short_leg = sel.iloc[0].to_dict()

    same_exp = chain[(chain["expiration_date"] == short_leg["expiration_date"]) &
                     (chain["strike"] > short_leg["strike"])].copy()
    if same_exp.empty:
        # fallback: pick any farther strike in later rows
        same_exp = chain[chain["strike"] > short_leg["strike"]].copy()

    long_leg = same_exp.iloc[0].to_dict() if not same_exp.empty else None

    return {"short": short_leg, "long": long_leg}


def load_polygon_option_aggregates(
    option_ticker: str,
    start: str,
    end: str,
    timespan: str = "minute",    # 'minute', '5minute', 'hour', 'day'
    adjusted: bool = True,
) -> pd.DataFrame:
    """
    Fetch aggregates for a specific option contract (ticker like 'O:TSLA240920C00300000').
    Returns timestamp(UTC), open, high, low, close, volume.
    """
    if PolygonRESTClient is None:
        raise ImportError("polygon-api-client is not installed. pip install polygon-api-client")

    client = PolygonRESTClient(api_key=POLYGON_API_KEY)

    rows: List[Dict[str, Any]] = []
    for a in client.list_aggs(
        ticker=option_ticker,
        multiplier=1,
        timespan=timespan,
        from_=pd.Timestamp(start, tz="UTC").strftime("%Y-%m-%d"),
        to=pd.Timestamp(end,   tz="UTC").strftime("%Y-%m-%d"),
        adjusted=adjusted,
        sort="asc",
        limit=50_000,
    ):
        rows.append({
            "timestamp": pd.to_datetime(a.timestamp, unit="ms", utc=True),
            "open": float(a.open),
            "high": float(a.high),
            "low": float(a.low),
            "close": float(a.close),
            "volume": float(a.volume),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("timestamp").reset_index(drop=True)


# ---------------------------------------------------------------------
# Convenience: simple monthly backfill helpers (can be called from tiny scripts)
# ---------------------------------------------------------------------
def backfill_month_with_loader(
    loader_fn,
    symbol: str,
    year: int,
    month: int,
    out_dir: str = DATA_DIR,
    rth_only: bool = True,
    **kwargs,
):
    """
    Generic monthly backfill wrapper:
      loader_fn(symbol, start_iso, end_iso, rth_only=?, **kwargs) -> DataFrame
    Saves to data/minute_bars/<symbol>/<YYYY-MM>.parquet
    """
    from pandas.tseries.offsets import MonthEnd
    start_ts = pd.Timestamp(f"{year}-{month:02d}-01", tz="UTC")
    end_ts   = (start_ts + MonthEnd(1)).normalize() + pd.Timedelta(hours=23, minutes=59)
    df = loader_fn(symbol, start_ts.isoformat(), end_ts.isoformat(), rth_only=rth_only, **kwargs)
    os.makedirs(f"{out_dir}/{symbol}", exist_ok=True)
    df.to_parquet(f"{out_dir}/{symbol}/{year}-{month:02d}.parquet", index=False)
    return df


def backfill_range_with_loader(
    loader_fn,
    symbols: Iterable[str],
    start: str,
    end: str,
    out_dir: str = DATA_DIR,
    rth_only: bool = True,
    **kwargs,
):
    """
    Iterate months from [start, end] for each symbol via the given loader_fn.
    """
    s = pd.Timestamp(start, tz="UTC").to_period("M")
    e = pd.Timestamp(end,   tz="UTC").to_period("M")
    months = list(pd.period_range(s, e, freq="M"))
    for sym in symbols:
        for p in months:
            df = backfill_month_with_loader(loader_fn, sym, p.year, p.month, out_dir=out_dir, rth_only=rth_only, **kwargs)
            print(sym, p.year, p.month, df.shape)
