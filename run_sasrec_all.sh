#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-python3}
RUNS=${RUNS:-5}
DATASETS=${DATASETS:-steam}

args_for() {
  case "$1" in
    beauty) echo "--max-len 50" ;;
    yambda_50m) echo "--batch 32" ;;
    *) echo "" ;;
  esac
}

for ds in $DATASETS; do
  extra=$(args_for "$ds")
  $PY sasrec_run.py --dataset "$ds" --runs "$RUNS" --only Multiplex --no-align-roles \
    --probes 2 --combine concat $extra
done
