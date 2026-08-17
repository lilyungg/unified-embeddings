#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-python3}
RUNS=${RUNS:-5}
DATASETS=${DATASETS:-ml1m beauty steam}

args_for() {
  case "$1" in
    beauty) echo "--max-len 50" ;;
    yambda_50m) echo "--batch 32" ;;
    *) echo "" ;;
  esac
}

for ds in $DATASETS; do
  extra=$(args_for "$ds")
  $PY sasrec_run.py --dataset "$ds" --runs "$RUNS" --no-align-roles $extra
  $PY sasrec_run.py --dataset "$ds" --runs "$RUNS" --only Multiplex --no-align-roles \
    --no-tied-baseline --probes 2 --combine concat $extra
  $PY sasrec_run.py --dataset "$ds" --runs "$RUNS" --only Multiplex --no-align-roles \
    --no-tied-baseline --probes 2 --combine mean $extra
done
