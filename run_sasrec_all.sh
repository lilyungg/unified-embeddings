#!/usr/bin/env bash
set -uo pipefail

PY=${PY:-python3}
DATASETS=${DATASETS:-ml1m beauty steam}

for ds in $DATASETS; do
  $PY train_sasrec.py --config="configs/${ds}_sasrec_tied_cl.py"
  $PY train_sasrec.py --config="configs/${ds}_sasrec_2probe_tied.py"
done
