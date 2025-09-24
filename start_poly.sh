#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
: "${POLYGON_API_KEY:?Set POLYGON_API_KEY first}"
python run_backtest_polygon.py \
  --source polygon \
  --symbols MSFT AVGO \
  --start 2025-06-01 --end 2025-08-31 \
  --train-months 2 --test-months 1 \
  --max-combos 20 \
  --rth-only \
  --polygon-max-concurrency 6 \
  --polygon-reqs-per-min 100 \
  --run-tag "$(date -u +poly_%Y%m%d_%H%M%SZ)"
