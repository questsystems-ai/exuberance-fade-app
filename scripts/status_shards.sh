#!/usr/bin/env bash
# Usage: status_shards.sh RUN_TAG
# Example: status_shards.sh poly_12m240_hiStake
set -euo pipefail
TAG="${1:?run_tag required}"
shopt -s nullglob
for L in ~/app/logs/${TAG}_s*_*.log; do
  printf '%-55s : ' "$(basename "$L")"
  awk '/^Optimizer total:|^Window /{last=$0} END{print last}' "$L"
done
