
# Exuberance Fade Backtester — User Manual

A lightweight research tool to test a **bearish intraday fade** strategy on U.S. equities using
minute bars. It can ingest data from **Alpaca** or **Polygon**, apply configurable **signals** to pick
overbought moments, simulate **short, capped-risk entries** (underlying-only), and produce **reports**
(PDF + CSV/JSON). An **optimizer** (walk-forward) can sweep parameter grids out-of-sample.

---

## What this app does (plain English)
- Pulls **minute bars** (prices & volume) for a list of tickers and date range.
- Computes **features** like VWAP, RSI, opening range, and volume-spike flags.
- Picks **candidates** each minute using a simple “exuberance score” (gap, VWAP stretch, RSI, volume).
- Enters **short** trades when signals fire (AM/PM windows), with **stop-loss** and **profit-take** rules.
- Closes positions at **15:55 ET** if still open.
- Writes a **run folder** with trades and a **quick report** (`quick_report.pdf` + machine-readable summaries).
- *(Optional)* Sweeps a **parameter grid** in a walk-forward loop to find robust settings.
- *(Optional)* Combines multiple runs (e.g., Alpaca vs Polygon) into one comparison PDF.

> Out of the box this simulates **underlying-only** shorts. A bear-call spread overlay is scaffolded
> (via Polygon options) but not enabled by default in the simulator yet.

---

## Glossary (plain English)

- **VWAP**: Volume-Weighted Average Price. “Fair” price weighted by trading volume.
- **VWAP z-score**: How far current price is from VWAP in standard deviations (bigger = more stretched).
- **RSI**: 0–100 momentum indicator; high RSI (e.g., 80+) can flag overbought conditions.
- **Opening range (ORH/ORL)**: Highest/lowest price in the first ~15 minutes after 9:30 ET.
- **Gap up**: Today’s open is X% above yesterday’s close.
- **Volume multiple**: Current minute’s volume vs typical volume for that minute of day.
- **Late-day blow-off**: Thin breakout to new highs after ~14:30 ET (often fades).
- **Exuberance score**: A combined score from gap, VWAP stretch, RSI, and volume to rank symbols.
- **AM/PM windows**: Allowed entry times (AM: 09:45–11:30 ET; PM: 14:30–15:55 ET).
- **Stop-loss**: Close the short if price rises L% from entry.
- **Profit mode**: Either “VWAP touch” or “σ-reversion” (price falls by 0.5× recent volatility from entry).
- **Walk-forward**: Train on earlier months, test on a later month; roll window forward to reduce overfitting.

---

## Prerequisites

- Python 3.11, `pip install -r requirements.txt`
- API keys (never commit secrets):
  - **Alpaca**: `ALPACA_KEY_ID`, `ALPACA_SECRET_KEY`
  - **Polygon**: `POLYGON_API_KEY`
- Keys can be set via environment or a `.env` you load in `config.py` (recommended).

---

## Quick Start (2-symbol smoke test, Alpaca)

**One-liner**
```powershell
python run_backtest.py --source alpaca --symbols AAPL MSFT --start 2025-07-01 --end 2025-08-31 --no-opt --run-tag alpaca_smoke2
```

**VS Code PowerShell with line breaks (use backticks):**
```powershell
python run_backtest.py --source alpaca --symbols AAPL MSFT `
  --start 2025-07-01 --end 2025-08-31 `
  --no-opt `
  --run-tag alpaca_smoke2
```

At the end you’ll see a **run folder path** like `reports\YYYYMMDD_HHMMSSZ_alpaca_smoke2`.
Open it to find:
- `trades.csv` — per-trade ledger
- `quick_report.pdf` — skim-friendly
- `metrics_summary.csv/json`, `monthly_pnl.csv`, `top5_trades.csv`, `worst5_trades.csv`, `bot_summary.json`
- `run_meta.json` — run arguments, params, and environment

---

## Typical Workflows

### A) Polygon smoke test (5 symbols)
```powershell
python run_backtest.py --source polygon --symbols AAPL MSFT NVDA AMD TSLA `
  --start 2025-06-01 --end 2025-08-31 `
  --no-opt `
  --run-tag poly_smoke
```

### B) One walk-forward window on a 3-month span (2 train / 1 test)
```powershell
python run_backtest.py --source alpaca --symbols AAPL MSFT `
  --start 2025-06-01 --end 2025-08-31 `
  --train-months 2 --test-months 1 `
  --max-combos 300 --fast `
  --run-tag alpaca_wf_2m1m_aapl_msft
```

### C) Wider universe (12 symbols), same window
```powershell
python run_backtest.py --source alpaca --symbols AAPL MSFT NVDA AMD TSLA META GOOG AMZN NFLX AVGO INTC MU `
  --start 2025-06-01 --end 2025-08-31 `
  --train-months 2 --test-months 1 `
  --max-combos 1000 --fast `
  --run-tag alpaca_wf_2m1m_top12
```

### D) Combine completed runs (e.g., Alpaca vs Polygon)
```powershell
python report_combine.py --runs "reports/<RUN_ID_1>" "reports/<RUN_ID_2>"
```
Outputs to `reports/combined_<timestamp>/`:
- `comparison_report.pdf` (metrics table, equity curves, drawdowns, monthly P&L)
- `combined_metrics.csv`, `combined_bot_summary.json`

---

## Command Reference — `run_backtest.py`

**Data source & scope**
- `--source {local|alpaca|polygon}`: choose provider
- `--symbols T1 T2 ...`: tickers (space-separated)
- `--start YYYY-MM-DD --end YYYY-MM-DD`: UTC dates
- `--demo`: generate synthetic minutes (ignores provider)
- `--rth-only`: use regular hours (09:30–16:00 ET)

**Simulation**
- `--initial_equity FLOAT` (default 100000.0)
- `--no-opt`: skip the walk-forward optimizer (baseline only)
- `--run-tag NAME`: label appended to the run folder
- `--no-report`: skip generating the PDF/CSV summaries

**Optimizer (walk-forward)**
- `--train-months N --test-months M`: rolling WF windows
- `--fast`: small internal defaults (for speed)
- `--max-combos K`: sample at most K parameter combos

**Rate/Runtimes**
- `--alpaca-limit INT`: warn near calls/min (default 200)
- `--polygon-limit INT`: warn near calls/min (default 120)
- `--warn-runtime-mins FLOAT`: warn if predicted runtime exceeds this

**Cache**
- `--write-cache`: write a Parquet cache of the loaded minutes

**Where outputs go**
- A new timestamped run folder: `reports/<RUN_ID>_<tag>/`
- Inside: logs, trades, metadata, quick report, and optimizer outputs (if enabled)

---

## Strategy Parameters (what’s being optimized)

**Entry filters**
- `gap_th` ∈ {0.02, 0.03, 0.04, 0.05} (Gap-up %)
- `vwap_z` ∈ {2.5, 3.0, 3.5} (VWAP stretch)
- `rsi_th` ∈ {75, 80, 85} (Overbought threshold)
- `vol_mult_th` ∈ {2.0, 3.0, 4.0} (Minute volume multiple)
- `entry_window` ∈ {am, pm, both}

**Selection & liquidity**
- `price_min` ∈ {$1, $3, $5}
- `dollar_vol_min` ∈ {$200k, $500k, $1M} (minute $-volume)
- `top_k_per_min` ∈ {3, 5, 8} (cross-sectional rank per minute)

**Risk & exits**
- `stop_loss_pct` ∈ {0.8%, 1.2%, 1.6%}
- `profit_mode` ∈ {vwap, sigma05} (VWAP touch vs 0.5×σ reversion)

**Sizing & pacing**
- `stake_pct` ∈ {0.5%, 1.0%, 1.5%} of equity
- `trades_week_cap` ∈ {5, 10, 15} (soft cap during training)

> The optimizer enforces guardrails (e.g., max drawdown and minimum trades) and picks winners by out-of-sample
> train/test P&L subject to constraints.

---

## Reports

Every run (unless `--no-report`) emits:
- `quick_report.pdf` — headline metrics, monthly P&L table, top/worst trades
- `metrics_summary.csv` / `.json` — easy to diff or feed to a bot
- `monthly_pnl.csv` — month-level counts and sums
- `top5_trades.csv` / `worst5_trades.csv`
- `bot_summary.json` — one-screen JSON for automations

**Alerts**
- `alerts.log` — rate-limit warnings and runtime predictions (e.g., “consider Runpod”).

---

## Performance Tips
- Use **`--no-opt`** for quick plumbing checks; add optimizer once outputs look sane.
- For a 3-month range, pass **`--train-months 2 --test-months 1`** to get **1 WF window**.
- Keep symbols modest at first (5–20); then scale up on a larger CPU instance.
- The trade loop uses **Numba JIT**; the heavy part can be **feature prep**—limit date span and symbols when iterating.

---

## Troubleshooting

- **Timezone/Period warnings**: informational; the code converts to **US/Eastern** for intraday logic.
- **Runtime estimate looks huge**: With `--no-opt` it’s just the baseline; for WF, constrain with `--max-combos` and `--fast`.
- **No WF windows**: With Jun–Aug and default train=3/test=1 → **0 windows**; use `--train-months 2 --test-months 1` or extend dates.
- **Rate-limit errors**: See `alerts.log`; reduce request rate or upgrade plan; use backfill helpers in smaller chunks.
