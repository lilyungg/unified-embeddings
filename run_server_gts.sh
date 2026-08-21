#!/usr/bin/env bash
set -euo pipefail

PY=.venv/bin/python

setup() {
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  uv venv --python 3.12 .venv
  uv pip install --python $PY -r requirements.txt
  $PY -c "import torch; print('cuda:', torch.cuda.is_available())"
}

smoke() {
  $PY train_candgen_gts.py --config=configs/vklsvd_gts_smoke.py
  $PY train_candgen_gts.py --config=configs/yambda50m_gts_smoke.py
}

yambda() {
  $PY train_candgen_gts.py --config=configs/yambda50m_gts_multi.py
  $PY train_candgen_gts.py --config=configs/yambda50m_gts_likes.py
}

vklsvd() {
  $PY train_candgen_gts.py --config=configs/vklsvd_gts.py
}

case "${1:-all}" in
  setup) setup ;;
  smoke) smoke ;;
  yambda) yambda ;;
  vklsvd) vklsvd ;;
  all) yambda; vklsvd ;;
  *) echo "usage: bash run_server_gts.sh {setup|smoke|yambda|vklsvd|all}"; exit 1 ;;
esac
