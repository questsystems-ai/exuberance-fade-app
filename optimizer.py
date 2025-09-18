import itertools
import json
import os
import random
from dataclasses import asdict
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

import pandas as pd
from tqdm import tqdm

from backtest import Params, simulate_symbol
from utils import ET
from selector import annotate_candidates


def _month_key(s: pd.Series) -> pd.Series:
    # Timezone-safe month key (YYYY-MM in US/Eastern)
    return s.dt.tz_convert(ET).dt.strftime("%Y-%m")


def month_splits(df: pd.DataFrame):
    mk = _month_key(df['timestamp'])
    return sorted(pd.unique(mk))


def _full_grid() -> Dict[str, List[Any]]:
    return {
        # signals
        'use_gap_fade': [True],
        'use_vwap_extreme': [True],
        'gap_th': [0.02, 0.03, 0.04, 0.05],
        'vwap_z': [2.5, 3.0, 3.5],
        'rsi_th': [75, 80, 85],
        # volume & selection
        'vol_mult_th': [2.0, 3.0, 4.0],
        'price_min': [1.0, 3.0, 5.0],
        'dollar_vol_min': [2e5, 5e5, 1e6],
        'top_k_per_min': [3, 5, 8],
        # execution
        'entry_window': ['am', 'pm', 'both'],
        'stop_loss_pct': [0.008, 0.012, 0.016],
        'profit_mode': ['vwap', 'sigma05'],
        'stake_pct': [0.005, 0.01, 0.015],
        # constraint
        'trades_week_cap': [5, 10, 15],
    }


def _iter_combos(grid: Dict[str, List[Any]],
                 max_combos: Optional[int] = None,
                 seed: int = 1337,
                 shard_index: Optional[int] = None,
                 shard_count: Optional[int] = None):
    keys, vals = zip(*grid.items())
    all_combos = [dict(zip(keys, v)) for v in itertools.product(*vals)]
    # Optional randomization so shards are well-mixed
    rnd = random.Random(seed)
    rnd.shuffle(all_combos)

    # Optional sampling
    if max_combos is not None and max_combos < len(all_combos):
        all_combos = all_combos[:max_combos]

    # Shard slicing: 0-based index, take every k-th item
    if shard_index is not None and shard_count and shard_count > 1:
        all_combos = all_combos[shard_index::shard_count]

    return all_combos

def _apply_selector_to_slice(dfs_by_symbol: Dict[str, pd.DataFrame], p: Params) -> Dict[str, pd.DataFrame]:
    return annotate_candidates(dfs_by_symbol, p, top_k_per_min=getattr(p, "top_k_per_min", 5))


def walk_forward(
    dfs_by_symbol: Dict[str, pd.DataFrame],
    base_params: Params,
    train_months: int = 3,
    test_months: int = 1,
    grid: Optional[Dict[str, List[Any]]] = None,
    max_combos: Optional[int] = None,
    reports_dir: str = "reports",
    top_n_export: int = 25,
    show_progress: bool = True,
    progress_log: bool = True,
    shard_index: Optional[int] = None, shard_count: Optional[int] = None, seed: int = 1337,
) -> pd.DataFrame:
    """
    Rolling walk-forward with progress bars.
      - Shows total progress across windows × combos and a nested per-window bar.
      - Writes top_params_*.csv, plus an optional optimizer_progress.jsonl log.
    """
    os.makedirs(reports_dir, exist_ok=True)
    grid = grid or _full_grid()
    combos = _iter_combos(grid, max_combos=max_combos, seed=seed,
                          shard_index=shard_index, shard_count=shard_count)

    months = sorted(set().union(*[set(month_splits(df)) for df in dfs_by_symbol.values()]))
    n_windows = max(0, len(months) - train_months - test_months + 1)

    winners_rollup: Dict[str, Dict[str, Any]] = {}
    results_rows: List[Dict[str, Any]] = []

    # progress logging
    progress_fp = os.path.join(reports_dir, "optimizer_progress.jsonl") if progress_log else None
    def _log_progress(payload: dict):
        if not progress_fp:
            return
        payload = {"ts": datetime.now(timezone.utc).isoformat(), **payload}
        with open(progress_fp, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

    # total progress bar (windows × combos)
    total_steps = max(1, n_windows) * max(1, len(combos))
    pbar_total = tqdm(total=total_steps, desc="Optimizer total", unit="combo", disable=not show_progress)

    for wi in range(train_months, len(months) - test_months + 1):
        train_keys = months[wi - train_months:wi]
        test_keys  = months[wi:wi + test_months]

        dfs_train = {s: df[_month_key(df['timestamp']).isin(train_keys)] for s, df in dfs_by_symbol.items()}
        dfs_test  = {s: df[_month_key(df['timestamp']).isin(test_keys)]  for s, df in dfs_by_symbol.items()}

        # Determine total weeks in train slice (for trades/week cap)
        try:
            tmin = min([d['timestamp'].min() for d in dfs_train.values() if len(d)])
            tmax = max([d['timestamp'].max() for d in dfs_train.values() if len(d)])
            weeks = max(1.0, (tmax - tmin).total_seconds() / (7 * 24 * 3600))
        except ValueError:
            weeks = 1.0

        best_combo, best_score, best_train = None, float("-inf"), None

        # per-window progress bar
        pbar_win = tqdm(total=len(combos), desc=f"Window {len(results_rows)+1}/{n_windows}", unit="combo",
                        leave=False, disable=not show_progress)

        for ci, combo in enumerate(combos):
            params_dict = {**asdict(base_params), **{k: v for k, v in combo.items() if k in Params.__annotations__}}
            p = Params(**params_dict)

            # Apply selector on TRAIN
            dfs_train_sel = _apply_selector_to_slice(dfs_train, p)

            pnl, worst_dd, n_trades = 0.0, 0.0, 0
            for d in dfs_train_sel.values():
                out = simulate_symbol(d, p)
                m = out.get('metrics', {})
                pnl += float(m.get('total_pnl', 0.0))
                worst_dd = min(worst_dd, float(m.get('max_dd', 0.0)))
                n_trades += int(m.get('n_trades', 0))

            avg_trades_per_week = n_trades / weeks if weeks > 0 else n_trades
            cap = float(combo.get('trades_week_cap', 9_999))
            if avg_trades_per_week > cap:
                score = pnl - 1e6
            else:
                score = pnl if (worst_dd > -0.15 and n_trades >= 20) else (pnl - 1e6)

            # best-so-far bookkeeping
            if score > best_score:
                best_score = score
                best_combo = combo
                best_train = {'train_pnl': pnl, 'train_dd': worst_dd, 'train_trades': n_trades,
                              'avg_trades_per_week': avg_trades_per_week}

            # progress updates
            pbar_win.update(1)
            pbar_total.update(1)
            if progress_log and (ci % 25 == 0 or ci == len(combos) - 1):
                _log_progress({
                    "window_index": len(results_rows) + 1,
                    "window_total": n_windows,
                    "combo_index": ci + 1,
                    "combos_total": len(combos),
                    "best_score_train": best_score,
                    "best_combo": best_combo,
                })

        pbar_win.close()

        # Test the winner for this window
        params_dict = {**asdict(base_params), **{k: v for k, v in best_combo.items() if k in Params.__annotations__}}
        p_best = Params(**params_dict)
        dfs_test_sel = _apply_selector_to_slice(dfs_test, p_best)

        pnl_t, worst_dd_t, n_trades_t = 0.0, 0.0, 0
        for d in dfs_test_sel.values():
            out = simulate_symbol(d, p_best)
            m = out.get('metrics', {})
            pnl_t += float(m.get('total_pnl', 0.0))
            worst_dd_t = min(worst_dd_t, float(m.get('max_dd', 0.0)))
            n_trades_t += int(m.get('n_trades', 0))

        row = {
            'train_months': train_keys,
            'test_months': test_keys,
            'best_params': json.dumps(best_combo),
            'train_total_pnl': float(best_train['train_pnl']) if best_train else 0.0,
            'train_max_dd': float(best_train['train_dd']) if best_train else 0.0,
            'train_trades': int(best_train['train_trades']) if best_train else 0,
            'train_avg_trades_per_week': float(best_train['avg_trades_per_week']) if best_train else 0.0,
            'test_total_pnl': float(pnl_t),
            'test_max_dd': float(worst_dd_t),
            'test_trades': int(n_trades_t),
        }
        results_rows.append(row)

        # winners rollup
        key = json.dumps(best_combo, sort_keys=True)
        agg = winners_rollup.get(key, {'combo': best_combo, 'wins': 0, 'total_test_pnl': 0.0, 'total_test_trades': 0})
        agg['wins'] += 1
        agg['total_test_pnl'] += float(pnl_t)
        agg['total_test_trades'] += int(n_trades_t)
        winners_rollup[key] = agg

        _log_progress({
            "window_done": len(results_rows),
            "n_windows": n_windows,
            "winner_combo": best_combo,
            "winner_test_pnl": pnl_t,
        })

    pbar_total.close()

    # Save per-window winners
    df_results = pd.DataFrame(results_rows)
    os.makedirs(reports_dir, exist_ok=True)
    df_results.to_csv(os.path.join(reports_dir, "top_params_by_window.csv"), index=False)

    # Save overall winners (ranked) + top-N export
    if winners_rollup:
        overall = pd.DataFrame([
            {
                **{'combo': json.dumps(v['combo'])},
                **{'wins': v['wins'], 'total_test_pnl': v['total_test_pnl'], 'total_test_trades': v['total_test_trades']}
            }
            for v in winners_rollup.values()
        ])
        overall = overall.sort_values(['total_test_pnl', 'wins'], ascending=[False, False]).reset_index(drop=True)
        overall.to_csv(os.path.join(reports_dir, "top_params_overall.csv"), index=False)
        overall.head(top_n_export).to_csv(os.path.join(reports_dir, "top_params_overall_topN.csv"), index=False)

    return df_results


def save_results(df: pd.DataFrame, path="reports/summary.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
