#!/usr/bin/env bash
set -euo pipefail

PY=.venv/bin/python
RUNS=${RUNS:-1}
EPOCHS=${EPOCHS:-80}
PATIENCE=${PATIENCE:-8}
VK_SUBSET=${VK_SUBSET:-ur0.01_ip0.01}

setup() {
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  uv venv --python 3.12 .venv
  uv pip install --python $PY -r requirements.txt
  $PY -c "import torch; print('cuda:', torch.cuda.is_available())"
}

smoke() {
  $PY candgen_gts.py --dataset vklsvd --subset up0.001_ip0.001 --loss full \
    --budgets 1.0 --epochs 2 --patience 0 --only Collisionless
  $PY candgen_gts.py --dataset yambda_50m --interaction likes \
    --budgets 1.0 --epochs 2 --patience 0 --only Multiplex
}

yambda() {
  $PY candgen_gts.py --dataset yambda_50m --interaction multi \
    --batches-per-epoch 2000 --runs $RUNS --epochs $EPOCHS --patience $PATIENCE
  $PY candgen_gts.py --dataset yambda_50m --interaction likes \
    --runs $RUNS --epochs $EPOCHS --patience $PATIENCE
}

vklsvd() {
  $PY candgen_gts.py --dataset vklsvd --subset $VK_SUBSET \
    --batches-per-epoch 2000 --runs $RUNS --epochs $EPOCHS --patience $PATIENCE
}

case "${1:-all}" in
  setup) setup ;;
  smoke) smoke ;;
  yambda) yambda ;;
  vklsvd) vklsvd ;;
  all) yambda; vklsvd ;;
  *) echo "usage: bash run_server_gts.sh {setup|smoke|yambda|vklsvd|all}"; exit 1 ;;
esac
