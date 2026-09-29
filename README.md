# Unified Embeddings for Retrieval

Research code for studying categorical embedding compression and feature
multiplexing in two-tower retrieval, motivated by
[Unified Embedding: Battle-Tested Feature Representations for Web-Scale ML Systems](https://arxiv.org/abs/2305.12102).

This branch contains the basic **RQ1: static retrieval with BPR** entry points:
collisionless embeddings, six compressed embedding families, per-feature and
multiplex layouts, and a train-only popularity baseline. It is a retrieval study,
not an exact reproduction of the original paper's datasets or experimental grid.
Paper results, checkpoints, research diagnostics, and job queues are not included.

## Repository layout

```text
src/
  data/                  # preprocessing, datasets, collate, device-aware loaders
  embeddings/            # lookup strategies and FeatureEmbedder
  models/                # towers, linear feature readers, sequential components
  losses/                # BPR, sampled/full CE, optional feature regularisation
  metrics/               # single-target Recall and NDCG
  training/              # trainer, retrieval objective, evaluation, callbacks
runs/
  retrieval/
    train.py             # common training CLI
    utils.py             # configuration, budgets, model and optimizer setup
    datasets.py          # dataset paths and default feature sets
    popularity.py        # standalone popularity evaluation
    qr_initialization.py # explicitly labelled QR initialization control
  <dataset>/retrieval/
    prepare_retrieval.py # one preprocessing entry point per dataset
data/
  raw/<dataset>/
  preprocessed/<dataset>/retrieval/{static,sequential}/
```

Run commands from the repository root. Datasets and generated outputs are ignored
by Git; `.gitkeep` files preserve the directory structure.

## Installation

The development environment uses Linux, Python 3.12.13, PyTorch 2.9.0 with CUDA
12.8, and NVIDIA A40 GPUs. Training uses FP32, without AMP or TF32. The commands
below use the [official PyTorch CUDA 12.8 wheel](https://pytorch.org/get-started/previous-versions/#v290).

```bash
git clone --branch paper-rq1 --single-branch https://github.com/lilyungg/unified-embeddings.git
cd unified-embeddings
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python runs/retrieval/train.py --help
```

Use a CUDA-capable environment for training. NumPy/pandas preprocessing and
DataLoader workers run on CPU. No experiment-tracking service is required.

## Data preparation

Download data from the original providers and follow their usage and citation
requirements. Data is not redistributed here. The scripts expect these files:

| Dataset | Files relative to `data/raw/<dataset>/` | Source |
| --- | --- | --- |
| `ml1m` | `ml-1m.zip`, containing `ml-1m/ratings.dat` and `ml-1m/users.dat` | [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) |
| `beauty` | `ratings_Beauty.csv`; `metadata_.json.zip`, containing `metadata.json` | [Amazon Reviews 2014](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html) |
| `steam` | `steam_new.json.zip`, containing `steam_new.json`; uncompressed `steam_games.json` | [Steam, Version 2](https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data) |
| `yambda` | `flat/500m/likes.parquet`; `artist_item_mapping.parquet`; `album_item_mapping.parquet` | [Yandex Yambda](https://huggingface.co/datasets/yandex/yambda) |

### MovieLens

```bash
wget -P data/raw/ml1m https://files.grouplens.org/datasets/movielens/ml-1m.zip
python runs/ml1m/retrieval/prepare_retrieval.py
```

### Amazon Beauty

Use the **full 2014 ratings corpus**, not the 5-core subset or the 2018/2023
versions. We use the full product metadata file and keep metadata only for items
present in Beauty interactions.

```bash
wget -P data/raw/beauty https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Beauty.csv
wget -P data/raw/beauty https://snap.stanford.edu/data/amazon/productGraph/metadata.json.gz
gzip -dk data/raw/beauty/metadata.json.gz
python -m zipfile -c data/raw/beauty/metadata_.json.zip data/raw/beauty/metadata.json
python runs/beauty/retrieval/prepare_retrieval.py
```

The one-time gzip-to-ZIP conversion preserves the metadata lines; leave enough
disk space for the uncompressed file. The ZIP filename and member name above
match the preprocessing contract. Already prepared archives need no conversion.

### Steam

Download Version 2 reviews and item metadata from the source page. Unpack their
gzip containers if needed. Preserve the original line order and Python-dictionary
records. Name the uncompressed reviews `steam_new.json`, then package them once:

```bash
python -m zipfile -c data/raw/steam/steam_new.json.zip data/raw/steam/steam_new.json
python runs/steam/retrieval/prepare_retrieval.py
```

Review records must contain `user_id`, `product_id`, and `date`; metadata uses
`id`, `developer`, `publisher`, and `genres`. Reviews without a nonempty `user_id`
are excluded: this release does **not** substitute `username`. Dates have day
precision. These choices matter when comparing against other Steam protocols.

### Yambda

Only the likes event file and two categorical mappings are needed. No audio
embeddings, listens, or provider-generated sequential files are used. With the
optional [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli):

```bash
python -m pip install huggingface_hub
hf download yandex/yambda flat/500m/likes.parquet artist_item_mapping.parquet album_item_mapping.parquet --repo-type dataset --local-dir data/raw/yambda
python runs/yambda/retrieval/prepare_retrieval.py
```

`500m` denotes the source variant across event types, not 500 million likes.
The likes file used in our prepared corpus contains 9,033,960 interactions.
We retain duplicate interactions and both organic and non-organic likes.

### Prepared artifacts and features

Preprocessing writes `metadata.json`, dense entity/feature codes, interaction
arrays, vocabularies, and four fixed namespace-aware hashes per categorical
value. Training reads these artifacts, **not the raw archives**. Existing
artifacts are reused when `metadata.json` exists. If inputs or preprocessing
settings change, explicitly set `REBUILD_STATIC = True` in the corresponding
preparation script; do not rebuild artifacts used by an active run.

The table below describes the prepared corpora used during development, before
the evaluation-only first-event exclusion:

| Dataset | Interactions | Users | Items | Default user features | Default item features |
| --- | ---: | ---: | ---: | --- | --- |
| MovieLens 1M | 1,000,209 | 6,040 | 3,706 | `user_idx`, `gender`, `age`, `occupation`, `zipcode` | `item_idx` |
| Beauty 2014 | 2,023,070 | 1,210,271 | 249,274 | `user_idx` | `item_idx`, `brand`, `categories` |
| Steam | 3,176,223 | 1,485,611 | 14,513 | `user_idx` | `item_idx`, `developer`, `publisher`, `genres` |
| Yambda likes | 9,033,960 | 82,788 | 638,230 | `user_idx` | `item_idx`, `artist`, `album` |

All model features are scalar categorical codes. A sorted collection of category
paths, genres, artist IDs, or album IDs is encoded as **one category**, not pooled
as separate tokens. Missing item metadata maps to `MISSING`. Rating and
`is_organic` fields are audit information, not model inputs.

## Run static BPR experiments

Start with MovieLens or select `beauty`, `steam`, or `yambda`. Defaults include all
features listed above. `--features` replaces that list in the specified order.

```bash
# Collisionless reference.
python runs/retrieval/train.py --dataset ml1m --model static --loss bpr --approach collisionless --seed 42 --device cuda:0

# Per-feature Hashing Trick: four lookups, 10x smaller main lookup budget.
python runs/retrieval/train.py --dataset ml1m --model static --loss bpr --approach hashing_trick --num-hashes 4 --budget-fraction 0.1 --seed 42 --device cuda:0

# Multiplex counterpart, with the same main lookup budget.
python runs/retrieval/train.py --dataset ml1m --model static --loss bpr --approach hashing_trick --multiplex --num-hashes 4 --budget-fraction 0.1 --seed 42 --device cuda:0

# Popularity: train interaction counts, the same evaluation targets and catalog.
python runs/retrieval/popularity.py --dataset ml1m
```

Substitute `--approach` to run other families. Use `--budget-fraction 1.0`, `0.1`,
or `0.04` for the x1/x10/x25 comparisons. Omit `--multiplex` for per-feature (PF)
tables; include it for multiplex (MP). For compressed methods, specify
`--num-hashes` explicitly: defaults differ between families.

### Embedding families

Here `d` is `--embedding-dim` (32 by default). Every tower projects its feature
embeddings to `--output-dim` and sums them; user/item scores are dot products.

| `--approach` | Representation | Meaning of `--num-hashes H` | Feature output width |
| --- | --- | --- | ---: |
| `collisionless` | Separate exact lookup table per feature | Not used | `d` |
| `hashing_trick` | H lookups into one table, concatenated | Hash lookups | `H*d` |
| `hash_embedding` | H lookups into one component table, learned weighted sum | Hash lookups; separate importance-weight table | `d` |
| `pq` | One lookup into each of H component tables, concatenated | Separate tables, splitting the row budget | `H*d` |
| `qr` | Mixed-radix dense-ID lookups, elementwise product | Components, **not independent hashes** | `d` |
| `hashed_net` | Signed coordinate-wise assembly from a scalar pool | H assemblies, concatenated | `H*d` |
| `robe_z` | Contiguous blocks from a circular scalar pool | Width multiplier; global block numbering | `H*d` |

Multiplex shares the method's table or collection of tables across hashed
features. Per-feature gives each hashed feature its own allocation. In both
layouts, features with cardinality at most `--hash-threshold 5` retain separate
collisionless tables. Hash-based methods using prepared hashes support H up to
4 with the supplied artifacts. ROBE-Z defaults to `--block-size 8`.

These are explicit research variants: `pq` is hash-based compositional concat,
not post-training k-means quantization; QR extends the two-component construction
to mixed radix. ROBE-Z uses circular addressing without sign multiplication and
a fixed 61-bit prime. It was informed by the
[ROBE-Z reference implementation](https://github.com/apd10/universal_memory_allocation/tree/d685e03e787645616d2017069c7bbddcf6b03b65),
but does not reproduce that commit's non-wrapping block addresses or hash mapping.
H>1 concat variants are our parameterization, not a claim to duplicate the
original Unified Embedding experiment grid.

### What is budget-matched?

For selected feature cardinalities `N_f`, fraction `b`, and row width `d`, the
main lookup budget is `ceil(b * sum(N_f)) * d` trainable scalars. Protected
collisionless tables are included. The remaining rows are allocated
proportionally for PF or shared for MP; PQ splits each allocation across its H
tables. HashedNet/ROBE-Z use the same scalar count in a one-dimensional pool.

Hash Embedding's learned importance weights are **additional**, outside this
main budget. Readers are also outside it and may be wider for concat variants.
Each run prints main lookup, importance, and reader parameter counts separately.
Thus this is not a total-model-size or equal-runtime comparison. x1 hashing
still permits collisions; it is not the collisionless reference.

QR must represent its dense namespace using reachable mixed-radix tables with
exactly the allocated row sum. An infeasible combination raises an error; report
it as N/A rather than silently enlarging its budget. In particular, H=1 does not
give compressed QR, and some small per-feature allocations cannot support H=4.

QR's default initialization remains uniform(-0.05, 0.05) per component, like
other lookup tables. Products can have much smaller initial scale. An explicit
variance-matched control is available and should be reported separately:

```bash
python runs/retrieval/qr_initialization.py --dataset steam --budget-fraction 0.1 --num-hashes 4 --seed 42 --device cuda:0
```

Add `--multiplex` for its shared-table counterpart. This control changes only
initialization, not the lookup budget.

## Training and evaluation protocol

- Stable global timestamp ordering, approximately 80/10/10 train/validation/test;
  split boundaries move to the end of a timestamp group. All retained
  interactions are positive, including low ratings. No additional k-core or
  rating-threshold filtering is applied.
- Vocabularies and catalog come from the **full interaction corpus**, not just
  train. Metadata does not introduce extra catalog items. This is a transductive
  catalog protocol; cold means unseen in training interactions, not unknown to
  the preprocessing vocabulary.
- Static training uses one sample per training interaction. Validation/test
  exclude each user's first-ever interaction, matching the nonempty-history
  sequential target protocol. Users/items unseen in train are not otherwise
  excluded; four warm/cold user-item slices are reported.
- BPR samples one uniform negative per target, with replacement from the entire
  catalog. Positive matches and previously observed items are not masked.
  `--num-sampled-negatives` controls sampled CE, not BPR.
- Exact full-catalog evaluation, without ANN or candidate sampling. Previously
  observed and cold items remain candidates. Each evaluation interaction has
  one target; Recall/NDCG are averaged over interactions, not users.
- Recall/NDCG at 10, 100, and 1000. **Validation Recall@1000** selects the best
  checkpoint. Test is evaluated after restoring that checkpoint. Ranking uses
  scores only, without an extra tie-breaking key.
- Training defaults: seed 42, batch 1024, eval batch 128, four DataLoader workers,
  four CPU threads, at most 100,000 optimizer steps, and patience 5 validation
  checks. CUDA loaders use pinned memory and non-blocking transfers.
- SparseAdam for lookup parameters and AdamW for readers, both at `3e-4`.
  AdamW retains its framework-default weight decay. Additional feature-embedding
  regularisation is off (`--embedding-l2 0`).
- Validation runs at epoch boundaries, including the final partial epoch when
  the step limit is reached. Training metrics are calculated on each current
  batch and averaged in logging windows (default 100 steps); there is no second
  training-data pass.

Full-catalog scoring materializes a `[batch_size, num_items]` matrix. Training
metrics also score the full catalog at each step. This is significant for
Beauty/Yambda: reduce batch sizes if necessary and keep them fixed within a
comparison. Even `--num-steps 1` performs full validation and test; it is not a
cheap import check. Use `--help` to verify the entry point without loading data.

## Outputs and reproducibility

By default, runs create a new timestamped directory under
`experiment_logs/<dataset>/retrieval/static/<method>_bpr/`. An explicit
`--output-dir` must not already exist. Each completed run writes:

- `metrics.jsonl`: training windows/epochs, validation, test, and timing.
- `best.pt`: model weights, best epoch/step, and validation metrics.
- `result.json`: best validation metrics, final test metrics (including
  warm/cold slices), and total optimizer steps.

Keep the Git commit, exact command-line overrides, console parameter counts,
and preprocessing artifacts together when reporting results. This launcher
does not automatically save a separate configuration manifest. Seeds control
initialization, shuffling, and negative sampling; preprocessing hash seeds are
fixed independently. Deterministic FP32 settings are enabled, but bitwise
identity across hardware/software versions is not promised.

Checkpoints are for evaluation, not exact optimizer/RNG resume. Reconstruct the
same feature configuration and preprocessed hashes before loading weights:
hash buffers are intentionally not serialized into the checkpoint.

The common source also supports `sampled_ce`, `full_ce`, and a sequential tower.
Their dedicated paper experiments and diagnostics are outside this branch's
RQ1 scope. Sequential preparation is opt-in through `PREPARE_SEQUENTIAL = True`
in the relevant preparation script. No data processing is triggered by the
training entry point.
