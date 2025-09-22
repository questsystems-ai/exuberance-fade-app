
# optimizer.py — robust walk-forward optimizer (defaults-first + DEFAULTS_JSON overlay)
import json, os, random
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from backtest import Params, simulate_symbol
from utils import ET
from selector import annotate_candidates

# --------------------- helpers: month keys ---------------------
def _month_key(s: pd.Series) -> pd.Series:
    if s.dt.tz is None:
        s = s.dt.tz_localize("UTC")
    return s.dt.tz_convert(ET).dt.strftime("%Y-%m")

def month_splits(df: pd.DataFrame) -> List[str]:
    mk = _month_key(df["timestamp"]); return sorted(pd.unique(mk))

# --------------------- DEFAULTS_JSON overlay ---------------------
def _load_defaults_json() -> Dict[str, Any]:
    raw = os.getenv("DEFAULTS_JSON")
    if not raw: return {}
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return {k: v for k, v in obj.items() if k in Params.__annotations__}
    except Exception:
        pass
    return {}

def _build_params_from_defaults(combo: Dict[str, Any], defaults_overlay: Dict[str, Any]) -> Params:
    base = asdict(Params())
    for k, v in (defaults_overlay or {}).items():
        if k in base: base[k] = v
    for k, v in combo.items():
        if k in base: base[k] = v
    return Params(**base)

# --------------------- tiny fixed grid (10 combos) ---------------------
def build_param_grid() -> List[Dict[str, Any]]:
    fixed = os.getenv("GRID_FIXED_JSON")
    if fixed:
        try:
            g = json.loads(fixed)
            if isinstance(g, list) and all(isinstance(x, dict) for x in g): return g
        except Exception:
            pass
    return [
        {"lookback_days": 20, "z_entry": 1.5, "z_exit": 0.5, "stop_loss_pct": 2.0, "trades_week_cap": 3},
        {"lookback_days": 15, "z_entry": 1.5, "z_exit": 0.5, "stop_loss_pct": 2.0, "trades_week_cap": 3},
        {"lookback_days": 25, "z_entry": 1.5, "z_exit": 0.5, "stop_loss_pct": 2.0, "trades_week_cap": 3},
        {"lookback_days": 20, "z_entry": 1.2, "z_exit": 0.5, "stop_loss_pct": 2.0, "trades_week_cap": 3},
        {"lookback_days": 20, "z_entry": 1.8, "z_exit": 0.5, "stop_loss_pct": 2.0, "trades_week_cap": 3},
        {"lookback_days": 20, "z_entry": 1.5, "z_exit": 0.3, "stop_loss_pct": 2.0, "trades_week_cap": 3},
        {"lookback_days": 20, "z_entry": 1.5, "z_exit": 0.8, "stop_loss_pct": 2.0, "trades_week_cap": 3},
        {"lookback_days": 20, "z_entry": 1.5, "z_exit": 0.5, "stop_loss_pct": 1.5, "trades_week_cap": 3},
        {"lookback_days": 20, "z_entry": 1.5, "z_exit": 0.5, "stop_loss_pct": 2.5, "trades_week_cap": 3},
        {"lookback_days": 20, "z_entry": 1.5, "z_exit": 0.5, "stop_loss_pct": 2.0, "trades_week_cap": 2},
    ]

def _iter_combos(grid_list: List[Dict[str, Any]], max_combos: Optional[int] = None, seed: int = 1337,
                 shard_index: Optional[int] = None, shard_count: Optional[int] = None) -> List[Dict[str, Any]]:
    combos = list(grid_list); rnd = random.Random(seed); rnd.shuffle(combos)
    if max_combos is not None and max_combos < len(combos): combos = combos[:max_combos]
    if shard_index is not None and shard_count and shard_count > 1: combos = combos[shard_index::shard_count]
    return combos

def _apply_selector_to_slice(dfs_by_symbol: Dict[str, pd.DataFrame], p: Params) -> Dict[str, pd.DataFrame]:
    top_k = getattr(p, "top_k_per_min", 5); return annotate_candidates(dfs_by_symbol, p, top_k_per_min=top_k)

# --------------------- main WF ---------------------
def walk_forward(
    dfs_by_symbol: Dict[str, pd.DataFrame],
    base_params: Params,  # kept for API compatibility; not used for defaulting
    train_months: int = 2,
    test_months: int = 1,
    grid: Optional[List[Dict[str, Any]]] = None,
    max_combos: Optional[int] = None,
    reports_dir: str = "reports",
    show_progress: bool = True,
    progress_log: bool = True,
    shard_index: Optional[int] = None,
    shard_count: Optional[int] = None,
    seed: int = 1337,
) -> pd.DataFrame:
    os.makedirs(reports_dir, exist_ok=True)

    grid_list = grid if grid is not None else build_param_grid()
    combos = _iter_combos(grid_list, max_combos=max_combos, seed=seed, shard_index=shard_index, shard_count=shard_count)
    defaults_overlay = _load_defaults_json()

    months = sorted(set().union(*[set(month_splits(df)) for df in dfs_by_symbol.values()]))
    n_windows = max(0, len(months) - train_months - test_months + 1)

    MIN_TRAIN_TRADES = int(os.getenv("OPT_MIN_TRAIN_TRADES", "5"))
    MIN_TEST_TRADES = int(os.getenv("OPT_MIN_TEST_TRADES", "3"))
    MIN_TEST_TRADES_PER_WEEK = float(os.getenv("OPT_MIN_TEST_TRADES_PER_WEEK", "0"))
    RISK_PENALTY = float(os.getenv("OPT_RISK_PENALTY", "0.0"))
    SHORTLIST_TOP_K = max(1, int(os.getenv("OPT_SHORTLIST_TOP_K", "5")))
    SAVE_GRID = os.getenv("OPT_SAVE_GRID", "0") == "1"

    progress_fp = os.path.join(reports_dir, "optimizer_progress.jsonl") if progress_log else None
    def _log_progress(payload: dict):
        if not progress_fp: return
        payload = {"ts": datetime.now(timezone.utc).isoformat(), **payload}
        with open(progress_fp, "a", encoding="utf-8") as f: f.write(json.dumps(payload) + "\n")

    total_steps = max(1, n_windows) * max(1, len(combos))
    pbar_total = tqdm(total=total_steps, desc="Optimizer total", unit="combo", disable=not show_progress)

    winner_rows: List[Dict[str, Any]] = []

    for wi in range(train_months, len(months) - test_months + 1):
        train_keys = months[wi - train_months : wi]; test_keys = months[wi : wi + test_months]
        dfs_train = {s: df[_month_key(df["timestamp"]).isin(train_keys)] for s, df in dfs_by_symbol.items()}
        dfs_test  = {s: df[_month_key(df["timestamp"]).isin(test_keys)]  for s, df in dfs_by_symbol.items()}

        # weeks in train
        try:
            tmin = min([d["timestamp"].min() for d in dfs_train.values() if len(d)])
            tmax = max([d["timestamp"].max() for d in dfs_train.values() if len(d)])
            weeks_train = max(1.0, (tmax - tmin).total_seconds() / (7 * 24 * 3600))
        except ValueError:
            weeks_train = 1.0

        train_grid_rows: List[Dict[str, Any]] = []
        pbar_win = tqdm(total=len(combos), desc=f"Window {len(winner_rows)+1}/{n_windows}", unit="combo", leave=False, disable=not show_progress)

        for ci, combo in enumerate(combos):
            p = _build_params_from_defaults(combo, defaults_overlay)
            dfs_train_sel = _apply_selector_to_slice(dfs_train, p)

            train_pnl, train_worst_dd, train_trades = 0.0, 0.0, 0
            for d in dfs_train_sel.values():
                out = simulate_symbol(d, p); m = out.get("metrics", {})
                train_pnl += float(m.get("total_pnl", 0.0))
                train_worst_dd = min(train_worst_dd, float(m.get("max_dd", 0.0)))
                train_trades += int(m.get("n_trades", 0))

            avg_trades_per_week = train_trades / weeks_train if weeks_train > 0 else train_trades
            cap = float(combo.get("trades_week_cap", 9_999))
            allowed_chat = avg_trades_per_week <= cap
            score_train = train_pnl - RISK_PENALTY * abs(train_worst_dd)

            train_grid_rows.append({**{k: v for k, v in combo.items() if k in Params.__annotations__ or k == "trades_week_cap"},
                                    "score_train": score_train, "train_total_pnl": train_pnl,
                                    "train_max_dd": train_worst_dd, "train_trades": train_trades,
                                    "train_avg_trades_per_week": avg_trades_per_week, "allowed_chat": allowed_chat})

            pbar_win.update(1); pbar_total.update(1)
            if progress_log and (ci % 25 == 0 or ci == len(combos) - 1):
                _log_progress({"window_index": len(winner_rows)+1, "window_total": n_windows,
                               "combo_index": ci+1, "combos_total": len(combos)})

        pbar_win.close()
        train_grid = pd.DataFrame(train_grid_rows)
        if SAVE_GRID:
            start_key = train_keys[0] if train_keys else "NA"; end_key = test_keys[-1] if test_keys else "NA"
            train_grid.to_csv(os.path.join(reports_dir, f"grid_results_{start_key}_{end_key}.csv"), index=False)

        train_grid = train_grid[(train_grid["train_trades"] >= MIN_TRAIN_TRADES) & (train_grid["allowed_chat"] == True)]
        if train_grid.empty:
            winner_rows.append({"window": f"{train_keys[0]}→{test_keys[-1]}",
                                "window_start": train_keys[0] if train_keys else "",
                                "window_end": test_keys[-1] if test_keys else "",
                                "phase": "test", "status": "no_train_candidates",
                                "best_params": json.dumps({}), "opt_test_pnl": 0.0,
                                "opt_test_trades": 0, "test_drawdown": 0.0, "train_trades": 0})
            continue

        shortlist = train_grid.sort_values("score_train", ascending=False).head(SHORTLIST_TOP_K).copy()

        # Evaluate TEST on shortlist
        test_rows: List[Dict[str, Any]] = []
        try:
            tmin_t = min([d["timestamp"].min() for d in dfs_test.values() if len(d)])
            tmax_t = max([d["timestamp"].max() for d in dfs_test.values() if len(d)])
            weeks_test = max(1.0, (tmax_t - tmin_t).total_seconds() / (7 * 24 * 3600))
        except ValueError:
            weeks_test = 1.0

        for _, cand in shortlist.iterrows():
            combo_pick = {k: cand[k] for k in cand.index if k in Params.__annotations__}
            p_cand = _build_params_from_defaults(combo_pick, defaults_overlay)
            dfs_test_sel = _apply_selector_to_slice(dfs_test, p_cand)

            test_pnl, test_worst_dd, test_trades = 0.0, 0.0, 0
            for d in dfs_test_sel.values():
                out = simulate_symbol(d, p_cand); m = out.get("metrics", {})
                test_pnl += float(m.get("total_pnl", 0.0)); test_worst_dd = min(test_worst_dd, float(m.get("max_dd", 0.0)))
                test_trades += int(m.get("n_trades", 0))

            test_score = test_pnl - RISK_PENALTY * abs(test_worst_dd)
            test_rows.append({**combo_pick, "train_trades": int(cand["train_trades"]), "score_train": float(cand["score_train"]),
                              "opt_test_pnl": test_pnl, "test_drawdown": test_worst_dd,
                              "opt_test_trades": int(test_trades), "test_score": test_score,
                              "test_avg_trades_per_week": test_trades / weeks_test if weeks_test > 0 else test_trades})

        test_df = pd.DataFrame(test_rows)
        if not test_df.empty:
            ok = test_df["opt_test_trades"] >= MIN_TEST_TRADES
            if MIN_TEST_TRADES_PER_WEEK > 0:
                ok = ok & (test_df["test_avg_trades_per_week"] >= MIN_TEST_TRADES_PER_WEEK)
            test_df = test_df[ok]

        if test_df.empty:
            fallback = shortlist.sort_values("score_train", ascending=False).iloc[0].to_dict()
            best_params = {k: v for k, v in fallback.items() if k in Params.__annotations__}
            winner_rows.append({"window": f"{train_keys[0]}→{test_keys[-1]}",
                                "window_start": train_keys[0] if train_keys else "",
                                "window_end": test_keys[-1] if test_keys else "",
                                "phase": "test", "status": "fallback_train_pick",
                                "best_params": json.dumps(best_params), "opt_test_pnl": 0.0,
                                "opt_test_trades": 0, "test_drawdown": 0.0, "train_trades": int(fallback.get("train_trades", 0))})
            continue

        best = test_df.sort_values("test_score", ascending=False).iloc[0].to_dict()
        best_params = {k: v for k, v in best.items() if k in Params.__annotations__}
        winner_rows.append({"window": f"{train_keys[0]}→{test_keys[-1]}",
                            "window_start": train_keys[0] if train_keys else "",
                            "window_end": test_keys[-1] if test_keys else "",
                            "phase": "test", "status": "ok",
                            "best_params": json.dumps(best_params),
                            "opt_test_pnl": float(best["opt_test_pnl"]),
                            "opt_test_trades": int(best["opt_test_trades"]),
                            "test_drawdown": float(best["test_drawdown"]),
                            "train_trades": int(best.get("train_trades", 0))})

    pbar_total.close()

    winners = pd.DataFrame(winner_rows)
    desired = ["window","window_start","window_end","phase","status",
               "opt_test_pnl","opt_test_trades","test_drawdown","train_trades","best_params"]
    cols = [c for c in desired if c in winners.columns] + [c for c in winners.columns if c not in desired]
    return winners[cols] if not winners.empty else winners
