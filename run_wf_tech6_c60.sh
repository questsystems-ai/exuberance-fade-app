#!/usr/bin/env bash
set -euo pipefail
cd /app
source .venv/bin/activate
mkdir -p logs

# env & tuning
export SYMS_TECH6="MSFT AAPL AMZN GOOG META AVGO"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1 PYTHONUNBUFFERED=1

TS=$(date -u +%Y%m%d_%H%M%SZ)
LOG="logs/tech6_12m_c60_opt_${TS}.log"
echo "[run] logging to $LOG"

python wf_driver_multishard.py \
  --symbols "$SYMS_TECH6" \
  --source polygon \
  --train-months 2 --test-months 1 \
  --start 2024-09-01 --end 2025-08-31 \
  --max-combos 60 \
  --shard-count 8 --max-procs 4 --seed 1337 \
  --run-tag tech6_12m_c60_opt \
  --extra-args --rth-only --no-report --use-optimized \
               --polygon-max-concurrency 4 --polygon-reqs-per-min 60 \
| tee -a "$LOG"
