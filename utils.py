import numpy as np
import pandas as pd

ET = 'US/Eastern'

def ensure_dt(df: pd.DataFrame, tz=None):
    if 'timestamp' not in df.columns:
        raise ValueError("Expected 'timestamp' column.")
    ts = pd.to_datetime(df['timestamp'], utc=True)
    if tz:
        ts = ts.tz_convert(tz)
    df = df.copy()
    df['timestamp'] = ts
    return df

def compute_vwap(df: pd.DataFrame):
    if 'vwap' in df.columns:
        return df
    tp = (df['high'] + df['low'] + df['close']) / 3.0
    v = df['volume'].replace(0, np.nan)
    cum_pv = (tp * v).cumsum()
    cum_v = v.cumsum()
    df = df.copy()
    df['vwap'] = (cum_pv / cum_v).bfill().ffill()
    return df

def minute_profile_volume(df: pd.DataFrame):
    ts_et = df['timestamp'].dt.tz_convert(ET)
    minute_of_day = ts_et.dt.hour * 60 + ts_et.dt.minute
    prof = df.assign(_mod=minute_of_day).groupby('_mod', as_index=True)['volume'].mean().rename('avg_vol_minute')
    return prof

def rsi(series: pd.Series, period: int = 14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).ewm(alpha=1/period, min_periods=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/period, min_periods=period).mean()
    rs = gain / (loss + 1e-12)
    return 100 - (100 / (1 + rs))

def rolling_vol(series: pd.Series, window: int = 60):
    return series.pct_change().rolling(window, min_periods=window//3).std().fillna(0.0)

def drawdown_curve(equity: pd.Series):
    peak = equity.cummax()
    dd = (equity - peak) / (peak + 1e-12)
    return dd
