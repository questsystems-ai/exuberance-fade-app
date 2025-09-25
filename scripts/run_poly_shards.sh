#!/usr/bin/env bash
# Usage: run_poly_shards.sh [SHARDS] [RUN_TAG]
# Example: run_poly_shards.sh 2 poly_12m240_hiStake
set -euo pipefail

SHARDS="${1:-2}"
RUN_TAG="${2:-poly_12m240_hiStake}"

# Optionally override these before calling this script:
export SYMS_TECH6="${SYMS_TECH6:-MSFT AAPL AMZN GOOG META AVGO}"
export QC_MIN_SYMS="${QC_MIN_SYMS:-4}"
export GRID_TAKE_PROFIT_SIGMA="${GRID_TAKE_PROFIT_SIGMA:-1.0,1.2,1.4}"
export GRID_STOP_LOSS_PCT="${GRID_STOP_LOSS_PCT:-0.006,0.008,0.010}"
export GRID_VOL_MULT_TH="${GRID_VOL_MULT_TH:-2.0,2.5,3.5}"
export GRID_VWAP_Z="${GRID_VWAP_Z:-3.0,3.5,4.0,4.5}"
export GRID_STAKE_PCT="${GRID_STAKE_PCT:-0.015,0.03,0.05}"
export PER_SHARD_RPM="${PER_SHARD_RPM:-30}"
export EXTRA_BACKTEST_ARGS="${EXTRA_BACKTEST_ARGS:-}"

for IDX in $(seq 0 $((SHARDS-1))); do
  ~/app/scripts/run_one_shard.sh "$IDX" "$SHARDS" "$RUN_TAG"
done

echo
echo "Started ${SHARDS} shards for tag: ${RUN_TAG}"
echo "List sessions: tmux ls"
