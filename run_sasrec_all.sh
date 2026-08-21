#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-python3}
DATASETS=${DATASETS:-steam}

for ds in $DATASETS; do
  $PY train_sasrec.py --config="configs/${ds}_sasrec_2probe.py"
done
