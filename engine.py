# engine.py
# Numba-compiled inner loop for the short, capped-risk intraday strategy.

from numba import njit
from numba.typed import List
import numpy as np

# reason codes: 1=stop, 2=target, 3=rollover, 4=time, 5=eod
REASON_STOP   = 1
REASON_TARGET = 2
REASON_ROLL   = 3
REASON_TIME   = 4
REASON_EOD    = 5

@njit(cache=True, fastmath=False)
def simulate_short_njit(
    entry_mask,         # bool[n]
    close,              # float64[n]
    sigma,              # float64[n]
    vwap,               # float64[n]
    near_close,         # bool[n]  (True when time>=15:55 ET)
    day_id,             # int64[n] (changes when ET "date" changes)
    stop_loss_pct,      # float64
    profit_mode,        # int64  (0 = vwap target, 1 = sigma target)
    take_profit_sigma,  # float64
    stake_pct,          # float64
    max_positions,      # int64
    initial_equity      # float64
):
    n = close.shape[0]
    equity = initial_equity

    # Active slots up to max_positions
    active = np.zeros(max_positions, dtype=np.uint8)
    entry_price = np.zeros(max_positions, dtype=np.float64)
    qty = np.zeros(max_positions, dtype=np.float64)
    entry_idx = np.zeros(max_positions, dtype=np.int64)

    trades = List()  # list of tuples: (entry_i, exit_i, entry_p, exit_p, qty, reason_code)

    prev_day = day_id[0] if n > 0 else 0
    prev_price = close[0] if n > 0 else 0.0
    prev_idx = 0

    for i in range(n):
        # Rollover at day change: close all at prev bar
        if day_id[i] != prev_day:
            for k in range(max_positions):
                if active[k] == 1:
                    pnl = (entry_price[k] - prev_price) * qty[k]
                    equity += pnl
                    trades.append((entry_idx[k], prev_idx, entry_price[k], prev_price, qty[k], REASON_ROLL))
                    active[k] = 0
            prev_day = day_id[i]

        price = close[i]
        sig = sigma[i]
        vw  = vwap[i]

        # Entries
        if entry_mask[i]:
            # count active
            count = 0
            for k in range(max_positions):
                if active[k] == 1:
                    count += 1
            if count < max_positions:
                risk_capital = equity * stake_pct
                q = 0.0
                if price > 1e-12:
                    q = risk_capital / price
                # first free slot
                for k in range(max_positions):
                    if active[k] == 0:
                        active[k] = 1
                        entry_price[k] = price
                        qty[k] = q
                        entry_idx[k] = i
                        break

        # Risk/exit loop
        for k in range(max_positions):
            if active[k] == 1:
                stop_price = entry_price[k] * (1.0 + stop_loss_pct)
                do_exit = False
                reason = 0

                # Stop
                if price >= stop_price:
                    do_exit = True
                    reason = REASON_STOP
                else:
                    # Take-profit
                    if profit_mode == 0:  # VWAP target
                        if price <= vw:
                            do_exit = True
                            reason = REASON_TARGET
                    else:  # sigma target
                        tp = entry_price[k] * (1.0 - max(0.0, take_profit_sigma) * max(sig, 1e-5))
                        if price <= tp:
                            do_exit = True
                            reason = REASON_TARGET

                # Time exit near 15:55 ET
                if (not do_exit) and near_close[i]:
                    do_exit = True
                    reason = REASON_TIME

                if do_exit:
                    pnl = (entry_price[k] - price) * qty[k]
                    equity += pnl
                    trades.append((entry_idx[k], i, entry_price[k], price, qty[k], reason))
                    active[k] = 0

        prev_price = price
        prev_idx = i

    # EOD: close remaining at last price
    for k in range(max_positions):
        if active[k] == 1:
            pnl = (entry_price[k] - prev_price) * qty[k]
            equity += pnl
            trades.append((entry_idx[k], prev_idx, entry_price[k], prev_price, qty[k], REASON_EOD))
            active[k] = 0

    return equity, trades
