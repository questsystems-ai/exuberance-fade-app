# signals.py
import pandas as pd
import numpy as np
from utils import compute_vwap, minute_profile_volume, rsi, rolling_vol, ET

OR_START_MIN = 9*60 + 30  # 09:30 ET

def add_intraday_features(df: pd.DataFrame, rsi_period=14, vol_window=60):
    df = compute_vwap(df).copy()
    df['rsi'] = rsi(df['close'], period=rsi_period)
    df['sigma'] = rolling_vol(df['close'], window=vol_window)
    df['vwap_dev'] = (df['close'] - df['vwap']) / (df['vwap'] + 1e-12)
    return df

def opening_range(df: pd.DataFrame, minutes: int = 15):
    # Compute the opening range (ORH/ORL) per day without groupby.apply (avoids warnings)
    df = df.copy()
    ts_et = df['timestamp'].dt.tz_convert(ET)
    df['day'] = ts_et.dt.date
    mod = ts_et.dt.hour * 60 + ts_et.dt.minute

    mask = (mod >= OR_START_MIN) & (mod < OR_START_MIN + minutes)
    or_bounds = (
        df.loc[mask]
          .groupby('day')
          .agg(ORH=('high','max'), ORL=('low','min'))
          .reset_index()
    )
    return df.merge(or_bounds, on='day', how='left')


def minute_volume_profile_flag(df: pd.DataFrame, multiple: float = 3.0):
    prof = minute_profile_volume(df)
    ts_et = df['timestamp'].dt.tz_convert(ET)
    df = df.copy()
    df['minute_of_day'] = ts_et.dt.hour * 60 + ts_et.dt.minute
    df = df.merge(prof, left_on='minute_of_day', right_index=True, how='left')
    df['vol_mult'] = df['volume'] / (df['avg_vol_minute'] + 1e-12)
    df['vol_exuberance'] = df['vol_mult'] >= multiple
    return df

def signal_gap_fade(df: pd.DataFrame, gap_th=0.03, hold_minutes=30):
    df = df.copy()
    ts_et = df['timestamp'].dt.tz_convert(ET)
    df['day'] = ts_et.dt.date
    df['mod'] = ts_et.dt.hour * 60 + ts_et.dt.minute

    # gap = today's open vs yesterday's close
    daily_open = df.groupby('day')['open'].first()
    daily_close = df.groupby('day')['close'].last()
    gaps = (daily_open - daily_close.shift(1)) / (daily_close.shift(1) + 1e-12)
    df['gap_pct'] = df['day'].map(gaps.to_dict())

    df = opening_range(df, minutes=15)
    df['after_hold'] = df['mod'] >= (OR_START_MIN + hold_minutes)
    df['gap_up'] = df['gap_pct'] >= gap_th
    df['fails_orh'] = df['after_hold'] & (df['close'] < df['ORH'])
    df['sig_gap_fade'] = df['gap_up'] & df['fails_orh']
    return df

def signal_vwap_extreme(df: pd.DataFrame, z_th=3.0, rsi_th=80):
    df = add_intraday_features(df)
    z = (df['vwap_dev'] - df['vwap_dev'].rolling(120, min_periods=60).mean()) / (
         df['vwap_dev'].rolling(120, min_periods=60).std() + 1e-12)
    df['vwap_z'] = z.fillna(0.0)
    df['sig_vwap_extreme'] = (df['vwap_z'] >= z_th) & (df['rsi'] >= rsi_th)
    return df

def signal_late_blowoff(df: pd.DataFrame, breakout_pct=0.5):
    df = df.copy()
    ts_et = df['timestamp'].dt.tz_convert(ET)
    df['day'] = ts_et.dt.date
    df['mod'] = ts_et.dt.hour * 60 + ts_et.dt.minute
    df['pm_window'] = df['mod'] >= (14*60 + 30)
    df['cum_high'] = df.groupby('day')['high'].cummax()
    prior_high = df.groupby('day')['cum_high'].shift(1)
    df['breakout'] = (df['high'] > (prior_high * (1 + breakout_pct/100.0)).fillna(np.inf))
    df['sig_late_blowoff'] = df['pm_window'] & df['breakout']
    return df
