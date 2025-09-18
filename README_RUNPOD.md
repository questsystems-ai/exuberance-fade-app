# Runpod Deployment Guide (Exuberance Fade Backtester)

This repo is ready to run on **Runpod** with CPU instances (GPU not required).

## 1) Prep your repo
- Ensure `.gitignore` is present (excludes `.venv/`, `data/`, `reports/`, `.env`, etc.)
- Put your API keys in a local `.env` file (never commit secrets)

## 2) Push to GitHub
```bash
git init
git add .
git commit -m "initial commit"
git branch -M main
git remote add origin YOUR_GITHUB_SSH_OR_HTTPS_URL
git push -u origin main
```

## 3) Start a Runpod (CPU)
- **Template**: Start New Pod → “Secure Cloud CPU” (or any CPU instance)
- **Container Image**: `python:3.11-slim`
- **Volume**: 20–50 GB (for cached parquet + reports)
- **Ports**: none required
- Launch, then open the **Web Terminal**

## 4) Pull & install
```bash
# inside the pod terminal
apt-get update && apt-get install -y git build-essential
git clone YOUR_REPO_URL
cd YOUR_REPO_FOLDER

# (optional) create .env from example
cp .env.example .env && nano .env  # paste keys

# Python env
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install -r requirements.txt
```

> If `config.py` reads keys from the environment (recommended), exporting them is enough:
```bash
export ALPACA_KEY_ID=...
export ALPACA_SECRET_KEY=...
export POLYGON_API_KEY=...
```

## 5) Quick sanity runs
```bash
# Synthetic, no optimizer
python run_backtest_v3.py --demo --no-opt --symbols AAPL MSFT --start 2025-07-01 --end 2025-08-31 --run-tag demo

# Alpaca slice, cache parquet
python run_backtest_v3.py --source alpaca --write-cache --symbols AAPL MSFT NVDA AMD TSLA \
  --start 2025-06-01 --end 2025-08-31 --no-opt --run-tag alpaca_smoke
```

Check the printed **run folder** (under `reports/<RUN_ID>/`). You’ll see:
- `trades.csv`, `run_meta.json`, and if optimizer ran: `summary.csv`, `top_params_*`.

## 6) Optional: run optimizer in a sampled grid
```bash
python run_backtest_v3.py --source alpaca --symbols AAPL MSFT NVDA AMD TSLA \
  --start 2025-05-01 --end 2025-08-31 --train-months 3 --test-months 1 \
  --max-combos 1000 --fast --run-tag grid_sample
```

## 7) Benchmarks (SPY/QQQ)
```bash
python benchmarks.py --run-dir reports/<YOUR_RUN_ID> --source alpaca --start 2025-06-01 --end 2025-08-31
```

## Notes
- This workload is **CPU-bound**; pick CPU instances with more vCPUs for faster grids.
- If you hit API plan limits, see `alerts.log` in the run folder; adjust rate or backfill in smaller chunks.
