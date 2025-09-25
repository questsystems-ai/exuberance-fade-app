#!/usr/bin/env bash
# Usage: run_one_shard.sh <shard-index> <shard-count> [run_tag]
# Example: run_one_shard.sh 0 2 poly_12m240_hiStake
set -euo pipefail

IDX="${1:?shard-index required (e.g., 0)}"
SHARDS="${2:?shard-count required (e.g., 2)}"
RUN_TAG="${3:-poly_12m240_hiStake}"

# ---- Tunables (override by exporting before calling this script) ----
export SYMS_TECH6="${SYMS_TECH6:-MSFT AAPL AMZN GOOG META AVGO}"
export QC_MIN_SYMS="${QC_MIN_SYMS:-4}"
export GRID_TAKE_PROFIT_SIGMA="${GRID_TAKE_PROFIT_SIGMA:-1.0,1.2,1.4}"
export GRID_STOP_LOSS_PCT="${GRID_STOP_LOSS_PCT:-0.006,0.008,0.010}"
export GRID_VOL_MULT_TH="${GRID_VOL_MULT_TH:-2.0,2.5,3.5}"
export GRID_VWAP_Z="${GRID_VWAP_Z:-3.0,3.5,4.0,4.5}"
export GRID_STAKE_PCT="${GRID_STAKE_PCT:-0.015,0.03,0.05}"
# Polygon requests per minute per shard (keep under your key’s total RPM)
PER_SHARD_RPM="${PER_SHARD_RPM:-30}"
# Any extra args you want to pass verbatim to the backtester (e.g., slippage):
#   export EXTRA_BACKTEST_ARGS='--slip-in-bps 8 --slip-out-bps 5'
EXTRA_BACKTEST_ARGS="${EXTRA_BACKTEST_ARGS:-}"

SESSION="${RUN_TAG}_s${IDX}"

tmux new -d -s "$SESSION" "bash -lc '
  set -e
  cd ~/app && source .venv/bin/activate
  # Export env for the grid override inside run_backtest_polygon.py
  export SYMS_TECH6=\"${SYMS_TECH6}\"
  export QC_MIN_SYMS=\"${QC_MIN_SYMS}\"
  export GRID_TAKE_PROFIT_SIGMA=\"${GRID_TAKE_PROFIT_SIGMA}\"
  export GRID_STOP_LOSS_PCT=\"${GRID_STOP_LOSS_PCT}\"
  export GRID_VOL_MULT_TH=\"${GRID_VOL_MULT_TH}\"
  export GRID_VWAP_Z=\"${GRID_VWAP_Z}\"
  export GRID_STAKE_PCT=\"${GRID_STAKE_PCT}\"

  mkdir -p logs
  LOG=logs/${RUN_TAG}_s${IDX}_\$(date -u +%Y%m%dT%H%M%SZ).log
  echo Running shard ${IDX}/${SHARDS} … logging to \$LOG

  stdbuf -oL -eL time python run_backtest_polygon.py \
    --source polygon \
    --symbols \"\$SYMS_TECH6\" \
    --start 2023-08-01 --end 2024-08-31 \
    --train-months 12 --test-months 1 \
    --max-combos 240 \
    --use-optimized \
    --qc-mode union \
    --rth-only \
    --shard-count \"${SHARDS}\" --shard-index \"${IDX}\" \
    --polygon-max-concurrency 2 --polygon-reqs-per-min ${PER_SHARD_RPM} \
    --run-tag ${RUN_TAG}_s${IDX} ${EXTRA_BACKTEST_ARGS} |& tee -a \"\$LOG\"
'"

echo "Started shard ${IDX}/${SHARDS} in tmux: ${SESSION}"
echo "Attach:   tmux attach -t ${SESSION}"
echo "Tail log: tail -F \$(ls -t ~/app/logs/${RUN_TAG}_s${IDX}_*.log | head -1) | grep -E '^[[]QC[]]|^Optimizer total:|^Window |^Report files created:|^Done\. Run folder:'"
