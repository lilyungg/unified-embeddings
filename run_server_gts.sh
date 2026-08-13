#!/usr/bin/env bash
# Featured GTS arm on a GPU server: Yambda (all features, owners' protocol)
# and VK-LSVD. Run from the repo root. Data downloads itself into datasets/
# (~2 GB total for the default subsets). Smoke-test first:
#   bash run_server_gts.sh setup && bash run_server_gts.sh smoke
set -euo pipefail

PY=.venv/bin/python
RUNS=1               # bump to 5 for mean±std
VK_SUBSET=ur0.01_ip0.01   # 100K users / 196K items / ~5.6M train likes

setup() {
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  uv venv --python 3.12 .venv
  uv pip install --python $PY -r requirements.txt
  # If cuda is False, reinstall torch for the driver's CUDA (see nvidia-smi):
  #   uv pip install --python $PY --reinstall torch \
  #     --index-url https://download.pytorch.org/whl/cu124
  $PY -c "import torch; print('cuda:', torch.cuda.is_available())"
}

smoke() {   # ~10 min: tiny VK subsample end-to-end + yambda loader
  $PY candgen_gts.py --dataset vklsvd --subset up0.001_ip0.001 --loss full \
    --budgets 1.0 --epochs 2 --patience 0 --only Collisionless
  $PY candgen_gts.py --dataset yambda_50m --interaction likes \
    --budgets 1.0 --epochs 2 --patience 0 --only Multiplex
}

yambda() {   # all features: train on listens+likes+dis/un-likes, targets=likes
  $PY candgen_gts.py --dataset yambda_50m --interaction multi \
    --batches-per-epoch 2000 --runs $RUNS
  # likes-only row, directly comparable to their BPR/ALS setting:
  $PY candgen_gts.py --dataset yambda_50m --interaction likes --runs $RUNS
}

vklsvd() {   # weekly GTS; features: user_id+age+gender+geo / item+author+duration
  $PY candgen_gts.py --dataset vklsvd --subset $VK_SUBSET \
    --batches-per-epoch 2000 --runs $RUNS
}

case "${1:-all}" in
  setup) setup ;;
  smoke) smoke ;;
  yambda) yambda ;;
  vklsvd) vklsvd ;;
  all) yambda; vklsvd ;;
  *) echo "usage: bash run_server_gts.sh {setup|smoke|yambda|vklsvd|all}"; exit 1 ;;
esac
