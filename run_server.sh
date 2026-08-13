#!/usr/bin/env bash
set -euo pipefail

PY=.venv/bin/python
BATCH=4096
RUNS=1
WORKERS=8

setup() {
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  uv venv --python 3.12 .venv
  uv pip install --python $PY -r requirements.txt
  $PY -c "import torch; print('cuda:', torch.cuda.is_available())"
}

download() {
  mkdir -p raw
  [ -f raw/train.txt ] || {
    curl -L -o raw/criteo.tar.gz \
      https://go.criteo.net/criteo-research-kaggle-display-advertising-challenge-dataset.tar.gz
    tar xzf raw/criteo.tar.gz -C raw/
  }
  echo "Avazu: fetch raw/train.gz via Kaggle CLI manually (see comment)."
}

prepare() {
  if [ ! -f datasets/criteo_prepared.parquet ]; then
    [ -f raw/train.txt ] && $PY prepare_data.py criteo --raw raw/train.txt \
      || echo "skip criteo: raw/train.txt missing"
  fi
  if [ ! -f datasets/avazu_prepared.parquet ]; then
    [ -f raw/train.gz ] && $PY prepare_data.py avazu --raw raw/train.gz \
      || echo "skip avazu: raw/train.gz missing"
  fi
}

smoke() {
  [ -f datasets/criteo_prepared.parquet ] || {
    echo "datasets/criteo_prepared.parquet missing — run download + prepare first."; exit 1; }
  $PY run.py --skip movielens --fast --budgets 1.0 --epochs 2 \
    --batch $BATCH --workers $WORKERS
}

criteo() {
  $PY run.py --skip movielens avazu --budgets 2.0 1.0 0.2 \
    --batch $BATCH --runs $RUNS --workers $WORKERS
}

avazu() {
  $PY run.py --skip movielens criteo --budgets 10.0 1.0 0.1 \
    --batch $BATCH --runs $RUNS --workers $WORKERS
}

case "${1:-all}" in
  setup) setup ;;
  download) download ;;
  prepare) prepare ;;
  smoke) smoke ;;
  criteo) criteo ;;
  avazu) avazu ;;
  all) prepare; criteo; avazu ;;
  *) echo "usage: bash run_server.sh {setup|download|prepare|smoke|criteo|avazu|all}"; exit 1 ;;
esac
