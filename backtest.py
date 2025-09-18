import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from utils import ET

# Try to import the JIT engine
_jit_ok = True
try:
    from engine import simulate_short_njit as _simulate_core
    from engine import REASON_STOP, REASON_TARGET, REASON_ROLL, REASON_TIME, REASON_EOD
except Exception:
    _jit_ok = False
    REASON_STOP, REASON_TARGET, REASON_ROLL, REASON_TIME, REASON_EOD = 1,2,3,4,5


@dataclass
class Params:
    # signal toggles
    use_gap_fade: bool = True
    use_vwap_extreme: bool = True
    use_late_blowoff: bool = False
    # thresholds
    gap_th: float = 0.03
    hold_minutes: int = 30
    vwap_z: float = 3.0
    rsi_th: int = 80
    breakout_pct: float = 0.5
    # selection & liquidity
    vol_mult_th: Optional[float] = None
    price_min: float = 3.0
    dollar_vol_min: float = 5e5
    top_k_per_min: int = 5
    # profit logic
    profit_mode: str = "sigma05"   # "vwap" | "sigma05"
    take_profit_sigma: float = 0.5
    # risk
    stop_loss_pct: float = 0.012
    # execution
    entry_window: str = "both"     # 'am' | 'pm' | 'both'
    stake_pct: float = 0.01
    max_positions: int = 5
    # engine
    use_jit: bool = True           # NEW: route to Numba core if available


def in_window(ts: pd.Series):
    """Return boolean masks for AM and PM entry windows in US/Eastern."""
    ts_et = ts.dt.tz_convert(ET)
    minutes = ts_et.dt.hour * 60 + ts_et.dt.minute
    am = (minutes >= 9 * 60 + 45) & (minutes <= 11 * 60 + 30)     # 09:45–11:30 ET
    pm = (minutes >= 14 * 60 + 30) & (minutes <= 15 * 60 + 55)    # 14:30–15:55 ET
    return am, pm


def generate_entries(df: pd.DataFrame, p: Params) -> pd.Series:
    """Combine signal toggles + window + volume gate + candidate_ok + liquidity into one boolean mask."""
    am, pm = in_window(df['timestamp'])
    mask = pd.Series(False, index=df.index, dtype=bool)

    if p.use_gap_fade and 'sig_gap_fade' in df:
        mask |= df['sig_gap_fade'].fillna(False)
    if p.use_vwap_extreme and 'sig_vwap_extreme' in df:
        mask |= df['sig_vwap_extreme'].fillna(False)
    if p.use_late_blowoff and 'sig_late_blowoff' in df:
        mask |= df['sig_late_blowoff'].fillna(False)

    # window
    if p.entry_window == 'am':
        mask &= am
    elif p.entry_window == 'pm':
        mask &= pm
    else:
        mask &= (am | pm)

    # volume gate
    if getattr(p, "vol_mult_th", None) is not None and "vol_mult" in df:
        mask &= (df["vol_mult"] >= float(p.vol_mult_th))

    # candidate selection
    if "candidate_ok" in df:
        mask &= df["candidate_ok"].astype(bool)

    # liquidity
    mask &= (df["close"] >= float(p.price_min))
    if "volume" in df:
        mask &= (df["close"] * df["volume"] >= float(p.dollar_vol_min))

    return mask


def _simulate_symbol_python(df: pd.DataFrame, p: Params, initial_equity: float) -> Dict[str, Any]:
    """(fallback) Original Python loop for correctness parity."""
    entries = generate_entries(df, p)

    trades = []
    equity = initial_equity
    current_day = None
    open_trades: List[Dict[str, Any]] = []

    prev_price = None
    prev_ts = None

    for i, row in df.iterrows():
        ts = row['timestamp']
        ts_et = ts.tz_convert(ET)
        day = ts_et.date()
        price = float(row['close'])
        sigma = float(row['sigma']) if 'sigma' in row and pd.notna(row['sigma']) else 0.005

        # rollover
        if current_day is not None and day != current_day and open_trades and prev_price is not None:
            for t in open_trades:
                pnl = (t['entry_price'] - prev_price) * t['qty']
                equity += pnl
                t.update({'exit_ts': prev_ts, 'exit_price': prev_price, 'pnl': pnl, 'exit_reason': 'rollover'})
                trades.append(t)
            open_trades = []

        # entry
        if bool(entries.iloc[i]):
            if len(open_trades) < p.max_positions:
                risk_capital = equity * p.stake_pct
                q = 0.0 if price <= 1e-12 else risk_capital / price
                open_trades.append({
                    'entry_ts': ts,
                    'entry_price': price,
                    'qty': q,
                    'symbol': row.get('symbol', 'UNK')
                })

        # exits
        exits_idx = []
        for k, t in enumerate(open_trades):
            stop_price = t['entry_price'] * (1 + p.stop_loss_pct)
            do_exit = False
            exit_reason = None

            if price >= stop_price:
                do_exit = True; exit_reason = 'stop'
            else:
                if p.profit_mode == "vwap":
                    vwap_val = float(row.get("vwap", price))
                    if price <= vwap_val:
                        do_exit = True; exit_reason = 'target_vwap'
                else:
                    take_price = t['entry_price'] * (1 - max(0.0, p.take_profit_sigma) * max(sigma, 1e-5))
                    if price <= take_price:
                        do_exit = True; exit_reason = 'target_sigma'

            if (ts_et.hour == 15 and ts_et.minute >= 55) and not do_exit:
                do_exit = True; exit_reason = exit_reason or 'time'

            if do_exit:
                pnl = (t['entry_price'] - price) * t['qty']
                equity += pnl
                t.update({'exit_ts': ts, 'exit_price': price, 'pnl': pnl, 'exit_reason': exit_reason})
                exits_idx.append(k)

        for k in reversed(exits_idx):
            trades.append(open_trades.pop(k))

        prev_price = price
        prev_ts = ts
        current_day = day

    if prev_price is None:
        prev_price = float(df.iloc[-1]['close'])
        prev_ts = pd.Timestamp(df.iloc[-1]['timestamp'])

    for t in open_trades:
        pnl = (t['entry_price'] - prev_price) * t['qty']
        equity += pnl
        t.update({'exit_ts': prev_ts, 'exit_price': prev_price, 'pnl': pnl, 'exit_reason': 'eod'})
        trades.append(t)

    ledger = pd.DataFrame(trades)
    equity_curve, metrics = _metrics_from_ledger(ledger, initial_equity)
    return {'equity_curve': equity_curve, 'trades': ledger, 'metrics': metrics}


def _metrics_from_ledger(ledger: pd.DataFrame, initial_equity: float):
    if ledger.empty:
        return pd.Series([initial_equity]), {
            'n_trades': 0, 'total_pnl': 0.0, 'avg_pnl': 0.0, 'hit_rate': 0.0, 'max_dd': 0.0, 'final_equity': initial_equity
        }
    ledger['date'] = pd.to_datetime(ledger['exit_ts']).dt.tz_convert(ET).dt.date
    daily_pnl = ledger.groupby('date', dropna=False)['pnl'].sum()
    equity_curve = daily_pnl.cumsum() + initial_equity
    max_dd = (equity_curve / equity_curve.cummax() - 1.0).min()
    metrics = {
        'n_trades': int(len(ledger)),
        'total_pnl': float(ledger['pnl'].sum()),
        'avg_pnl': float(ledger['pnl'].mean()),
        'hit_rate': float((ledger['pnl'] > 0).mean()),
        'max_dd': float(max_dd),
        'final_equity': float(equity_curve.iloc[-1]),
    }
    return equity_curve, metrics


def simulate_symbol(df: pd.DataFrame, p: Params, initial_equity: float = 100_000.0) -> Dict[str, Any]:
    if df.empty:
        return {
            'equity_curve': pd.Series([initial_equity]),
            'trades': pd.DataFrame(),
            'metrics': {'n_trades': 0, 'total_pnl': 0.0, 'avg_pnl': 0.0, 'hit_rate': 0.0, 'max_dd': 0.0, 'final_equity': initial_equity}
        }

    # If JIT disabled or unavailable, use python fallback
    if not (p.use_jit and _jit_ok):
        return _simulate_symbol_python(df, p, initial_equity)

    # --- Prepare arrays for the JIT core ---
    entries = generate_entries(df, p).to_numpy(dtype=np.bool_)

    close = df['close'].to_numpy(dtype=np.float64)
    sigma = (df['sigma'] if 'sigma' in df else pd.Series(0.005, index=df.index)).fillna(0.005).to_numpy(dtype=np.float64)
    vwap  = (df['vwap']  if 'vwap'  in df else df['close']).to_numpy(dtype=np.float64)

    ts_et = df['timestamp'].dt.tz_convert(ET)
    near_close = ((ts_et.dt.hour == 15) & (ts_et.dt.minute >= 55)).to_numpy(dtype=np.bool_)
    # day id
    day_codes, _ = pd.factorize(ts_et.dt.date)
    day_id = day_codes.astype(np.int64)

    stop = float(p.stop_loss_pct)
    pmode = 0 if p.profit_mode == "vwap" else 1
    tps = float(p.take_profit_sigma)
    stake = float(p.stake_pct)
    maxpos = int(p.max_positions)

    # --- Call JIT core ---
    final_equity, recs = _simulate_core(entries, close, sigma, vwap, near_close, day_id,
                                        stop, pmode, tps, stake, maxpos, float(initial_equity))

    # --- Build ledger back in Python ---
    # recs is list of tuples: (ei, xi, ep, xp, q, reason)
    if len(recs) == 0:
        equity_curve, metrics = _metrics_from_ledger(pd.DataFrame(), initial_equity)
        return {'equity_curve': equity_curve, 'trades': pd.DataFrame(), 'metrics': metrics}

    idx_to_ts = df['timestamp'].to_numpy()
    data = []
    for (ei, xi, ep, xp, q, rc) in recs:
        pnl = (ep - xp) * q
        if rc == REASON_STOP:
            reason = 'stop'
        elif rc == REASON_TARGET:
            reason = 'target'
        elif rc == REASON_ROLL:
            reason = 'rollover'
        elif rc == REASON_TIME:
            reason = 'time'
        elif rc == REASON_EOD:
            reason = 'eod'
        else:
            reason = 'exit'
        data.append((idx_to_ts[ei], ep, idx_to_ts[xi], xp, q, pnl, reason))

    ledger = pd.DataFrame(data, columns=['entry_ts','entry_price','exit_ts','exit_price','qty','pnl','exit_reason'])
    equity_curve, metrics = _metrics_from_ledger(ledger, initial_equity)
    return {'equity_curve': equity_curve, 'trades': ledger, 'metrics': metrics}
