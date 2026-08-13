# unified-embeddings

Three categorical embedding strategies — Non-multiplex (per-feature hash
tables, budget split by cardinality), Multiplex (one shared table, per-feature
hash salt), Collisionless (exact vocabulary, upper bound) — replicating
Feature Multiplexing / Unified Embedding (https://arxiv.org/abs/2305.12102)
on CTR ranking, then extended to retrieval: two-tower candidate generation,
SASRec, and featured runs under the dataset owners' GTS protocols.
Per-dataset processing and traps: [DATASETS.md](DATASETS.md). Formal theory
for the retrieval losses: [THEORY.md](THEORY.md).

## Setup

Python >= 3.10 (tested on 3.12). With [uv](https://docs.astral.sh/uv/):

```
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
source .venv/bin/activate
```

## Data (one time)

MovieLens-1M:

```
curl -O https://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip ml-1m.zip
```

Avazu and Criteo — raw Kaggle files through the one-time preparation (vocab
pruning to the paper's Tables 4/5, Criteo dense log-normalization, Avazu
hour-of-day):

```
python prepare_data.py criteo --raw train.txt
python prepare_data.py avazu  --raw train.gz
```

Retrieval datasets (Beauty, Steam, Gowalla, Yambda, VK-LSVD) download
themselves into datasets/ on first use.

## Running

### Ranking (paper replication)

```
python run.py --ml1m ./ml-1m --skip avazu criteo --budgets 1.0 0.5 0.1            # MovieLens sweep
python run.py --ml1m ./ml-1m --skip avazu criteo --budgets 1.0 0.5 0.1 --runs 5   # paper protocol, mean±std
python run.py --ml1m ./ml-1m --fast                                               # 1M-row Avazu/Criteo sample
bash run_server.sh setup && bash run_server.sh all                                # full Criteo+Avazu on a GPU server
python report.py experiment_logs/<ts>_<dataset>.json                              # table with diff vs paper Table 1
python plots.py  experiment_logs/<ts>_movielens.json                              # tradeoff / norms / curves PNGs
python orthogonality.py --ml1m ./ml-1m                                            # Sec. 4.2 / Fig. 2 experiment
```

All options: `python run.py --help` (labels, splits, architecture overrides,
budgets, seeds, TensorBoard).

### Candidate generation (two-tower)

```
python candgen.py --dataset movielens --budgets 1.0 0.5 0.1 --runs 5
python candgen.py --dataset beauty|steam|gowalla|yambda_50m
python candgen.py --loss sampled          # in-batch sampled softmax + logQ
python candgen.py --score cosine
python candgen.py --worst-init --only Multiplex
```

### SASRec

```
python sasrec_run.py --dataset ml1m --budgets 1.0 0.5 0.1     # + tied baseline by default
python sasrec_run.py --dataset ml1m --no-align-roles          # salted item_in/item_out
```

### GTS arm (features + owners' protocol: Yambda, VK-LSVD)

```
bash run_server_gts.sh setup && bash run_server_gts.sh smoke
RUNS=5 bash run_server_gts.sh all
python candgen_gts.py --dataset yambda_50m --interaction multi|likes|listens
python candgen_gts.py --dataset vklsvd --subset up0.001_ip0.001 --positive like|watch
```

### TensorBoard

```
tensorboard --logdir runs
python tb_export.py            # convert any experiment_logs/*.json to runs/
```

Tags: `<metric>/<experiment>/b<budget>` (`auc_val` for ranking,
`ndcg10_val` / `ndcg100_val` / `recall100_val` for retrieval); multi-seed
runs are averaged into one curve per method.

## Results — ranking (paper Table 1)

DCN-V2, `--ml-labels wang`. Full histories in experiment_logs/, per-dataset
configs in [DATASETS.md](DATASETS.md).

### MovieLens-1M — 5 runs (mean ± std)

Budgets map to Table 1 columns: 1.0x = 1.6MB, 0.5x = 791kB, 0.1x = 158kB.

| Budget | Experiment | AUC (5 runs) | Paper | Diff |
|---|---|---|---|---|
| 1.0x | Non-multiplex + DCN | 0.8684 ± 0.0005 | 0.8537 | +0.015 |
| 1.0x | Multiplex + DCN     | 0.8872 ± 0.0002 | 0.8774 | +0.010 |
| 1.0x | Collisionless + DCN | 0.8985 ± 0.0004 | 0.8872 | +0.011 |
| 0.5x | Non-multiplex + DCN | 0.8407 ± 0.0004 | 0.8300 | +0.011 |
| 0.5x | Multiplex + DCN     | 0.8759 ± 0.0007 | 0.8693 | +0.007 |
| 0.1x | Non-multiplex + DCN | 0.7712 ± 0.0008 | 0.7707 | +0.000 |
| 0.1x | Multiplex + DCN     | 0.8263 ± 0.0006 | 0.8200 | +0.006 |

MLP arm (ours, single run): Non-multiplex 0.8664 / 0.8387 / 0.7682,
Multiplex 0.8861 / 0.8751 / 0.8264, Collisionless 0.8963.

### Criteo — full dataset, single run

Budgets map to Table 1 columns: 2.0x = 25MB, 1.0x = 12.5MB, 0.2x = 2.5MB.

| Budget | Experiment | AUC | Paper | Diff |
|---|---|---|---|---|
| 25MB   | Non-multiplex + DCN | 0.8067 | 0.8021 | +0.005 |
| 25MB   | Multiplex + DCN     | 0.8092 | 0.8043 | +0.005 |
| 25MB   | Collisionless + DCN | 0.8095 | 0.8070 | +0.003 |
| 12.5MB | Non-multiplex + DCN | 0.8058 | 0.7998 | +0.006 |
| 12.5MB | Multiplex + DCN     | 0.8090 | 0.8047 | +0.004 |
| 2.5MB  | Non-multiplex + DCN | 0.7989 | 0.7944 | +0.005 |
| 2.5MB  | Multiplex + DCN     | 0.8082 | 0.8049 | +0.003 |

### Avazu — full dataset, single run

Budgets: 10.0x = 32.4MB, 1.0x = 3.24MB, 0.1x = 324kB.

| Budget | Experiment | AUC | Paper | Diff |
|---|---|---|---|---|
| 32.4MB | Non-multiplex + DCN | 0.7748 | 0.7724 | +0.002 |
| 32.4MB | Multiplex + DCN     | 0.7761 | 0.7735 | +0.003 |
| 32.4MB | Collisionless + DCN | 0.7765 | 0.7735 | +0.003 |
| 3.24MB | Non-multiplex + DCN | 0.7679 | 0.7671 | +0.001 |
| 3.24MB | Multiplex + DCN     | 0.7737 | 0.7718 | +0.002 |
| 324kB  | Non-multiplex + DCN | 0.7525 | 0.7510 | +0.002 |
| 324kB  | Multiplex + DCN     | 0.7702 | 0.7686 | +0.002 |

### Weight orthogonalization (Sec. 4.2 / Fig. 2)

Single-layer model, worst-case init (all feature weights aligned); mean
pairwise angle grows 0° → 53° (M=27K) → 70-72° (M≤1.4K) as the table shrinks.

![Orthogonalization](plots/orthogonality.png)

![Parameter-accuracy tradeoff](plots/tradeoff.png)

![Embedding norm scaling](plots/norms.png)

![Training curves](plots/curves.png)

## Results — candidate generation (two-tower, full softmax)

Linear towers (d=30, k=32), softmax over the full catalog, seen items masked,
test HR@10. MovieLens: 5 runs (mean ± std); other datasets: single run.
Collisionless runs once at the max budget.

| Dataset | Budget | Non-multiplex | Multiplex | Collisionless |
|---|---|---|---|---|
| MovieLens-1M | 1.0x | 0.0892 ± 0.0008 | 0.1144 ± 0.0003 | 0.1319 ± 0.0008 |
| | 0.5x | 0.0520 ± 0.0005 | 0.1018 ± 0.0007 | — |
| | 0.1x | 0.0185 ± 0.0000 | 0.0387 ± 0.0004 | — |
| Gowalla | 1.0x | 0.0396 | 0.0453 | 0.0715 |
| | 0.5x | 0.0248 | 0.0357 | — |
| | 0.1x | 0.0043 | 0.0089 | — |
| Steam | 1.0x | 0.0401 | 0.0593 | 0.0591 |
| | 0.5x | 0.0416 | 0.0595 | — |
| | 0.1x | 0.0084 | 0.0615 | — |
| Beauty | 1.0x | 0.0040 | 0.0081 | 0.0134 |
| | 0.5x | 0.0053 | 0.0059 | — |
| | 0.1x | 0.0007 | 0.0026 | — |
| Yambda-50M (ids) | 1.0x | 0.0044 | 0.0069 | 0.0096 |
| | 0.5x | 0.0038 | 0.0044 | — |
| | 0.1x | 0.0000 | 0.0008 | — |

## Results — SASRec (ML-1M, leave-one-out, full catalog)

GSASRec backbone (github.com/NonameUntitled/logq, architecture untouched),
authors' ml1m config; only the embedding tables are swapped. Test NDCG@10 /
HR@10, single run. aligned = one hash for item_in/item_out; tied = shared
input/output item embeddings (no collisions, 0.5x the untied rows).

| Memory | Configuration | NDCG@10 | HR@10 |
|---|---|---|---|
| 3.60 MB | Collisionless (untied) | 0.1703 | 0.2892 |
| | Multiplex | 0.1432 | 0.2442 |
| | Multiplex (aligned) | 0.1386 | 0.2500 |
| | Non-multiplex | 0.1252 | 0.2124 |
| 1.85 MB | Collisionless (tied) | 0.1708 | 0.2987 |
| | Multiplex | 0.1164 | 0.2017 |
| | Multiplex (aligned) | 0.1090 | 0.1987 |
| | Non-multiplex | 0.0932 | 0.1573 |
| 0.45 MB | Multiplex | 0.0419 | 0.0732 |
| | Multiplex (aligned) | 0.0352 | 0.0644 |
| | Non-multiplex | 0.0275 | 0.0485 |

## Results — GTS arm (features, owners' protocol)

Two-tower as above; in-batch sampled softmax + logQ; Global Temporal Split,
top-100 over the train catalog, no seen-item masking, cold target items kept,
recall@K = hits / min(|T|, K), macro over users; model selection on val
NDCG@100. Deviations from the owners' code: [DATASETS.md](DATASETS.md).
Test recall@100, 5 seeds (mean ± std).

**Yambda-50M multi** — train on all events (listens with played_ratio ≥ 50,
likes, dis/unlikes; 30.4M events, 696K-item catalog), targets = next-day
likes. Towers: user_id + event_type / item_id + track-length bucket.

| Budget | Non-multiplex | Multiplex | Collisionless |
|---|---|---|---|
| 1.0x | 0.0214 ± 0.0015 | 0.0352 ± 0.0030 | 0.0536 ± 0.0025 |
| 0.5x | 0.0142 ± 0.0012 | 0.0284 ± 0.0046 | — |
| 0.1x | 0.0040 ± 0.0008 | 0.0106 ± 0.0016 | — |

**Yambda-50M likes** — likes only (872K train events, 180K catalog), the
owners' default interaction. Towers: user_id / item_id.

| Budget | Non-multiplex | Multiplex | Collisionless |
|---|---|---|---|
| 1.0x | 0.0282 ± 0.0014 | 0.0395 ± 0.0033 | 0.0501 ± 0.0037 |
| 0.5x | 0.0204 ± 0.0029 | 0.0295 ± 0.0048 | — |
| 0.1x | 0.0068 ± 0.0007 | 0.0085 ± 0.0017 | — |

**VK-LSVD ur0.01_ip0.01** — official subsample: 1% random users x 1% most
popular items (u=user, i=item, r=random, p=popular by train interaction
rank); weekly GTS (train weeks 00-24, val 25, test 26); positives = likes
(4.06M train events, 64K users, 173K-item catalog). Towers: user_id + age +
gender + geo / item_id + author_id + duration.

| Budget | Non-multiplex | Multiplex | Collisionless |
|---|---|---|---|
| 1.0x | 0.0253 ± 0.0008 | 0.0294 ± 0.0004 | 0.0300 ± 0.0008 |
| 0.5x | 0.0203 ± 0.0003 | 0.0274 ± 0.0006 | — |
| 0.1x | 0.0096 ± 0.0004 | 0.0196 ± 0.0005 | — |

JSONs: experiment_logs/ (5-seed GTS: 20260813_090758 multi, 20260813_102949
likes, 20260813_103823 vklsvd).
