# Download, prepare, and verify the retrieval datasets

Run commands from the repository root after installing `requirements.txt`.
The reference artifacts were prepared on little-endian Linux with Python
3.12.13, NumPy 2.4.3, pandas 2.3.3, PyArrow 23.0.1, and xxhash 3.6.0.
Preprocessing is CPU-only and separate from training. No dataset is distributed
in Git. Follow the providers' usage and citation requirements.

## What the checksums establish

[data/checksums/](data/checksums/) contains SHA256 manifests measured on
30 September 2026 from the actual research inputs and artifacts:

- `<dataset>_raw.sha256`: the eight inputs across the four datasets. Beauty
  metadata and Steam reviews are hashed as **uncompressed bytes**, not their
  locally created ZIP containers. MovieLens uses the provider's original ZIP.
- `<dataset>_static.sha256`: every static `.npy` and `metadata.json`.
- `<dataset>_sequential.sha256`: every sequential `.npy` and `metadata.json`.

There are 92 prepared files: ML1M 21 static + 6 sequential; Beauty 15 + 6;
Steam 17 + 6; Yambda 15 + 6. Paths in the manifests are relative to the repo
root. Verification reads file bytes only; it does not parse raw records,
rebuild data, or load a model. Run it once after preparation or transfer, not
at every experiment launch.

An `OK` for every entry establishes byte identity with the reference files,
including ID mappings, feature vocabularies, hashes, splits and sequence order.
Counts alone do not establish identity. A checksum mismatch is not silently
accepted even if dimensions match. Changes to NPY/JSON serialization or platform
can also cause mismatches: use the pinned environment and investigate them.
These checks do not promise identical training metrics on different hardware.

## 1. Obtain the exact raw inputs

Skip downloading and conversion if you already have the expected files.
Keep source row order, encoding and line endings unchanged. Do not reserialize
the line-based Python dictionary records as JSON.

### MovieLens 1M

Source: [GroupLens MovieLens 1M](https://grouplens.org/datasets/movielens/1m/).
The preprocessor reads `ml-1m/ratings.dat` and `ml-1m/users.dat` inside the ZIP;
`movies.dat` is not used.

```bash
mkdir -p data/raw/ml1m
wget -c -P data/raw/ml1m https://files.grouplens.org/datasets/movielens/ml-1m.zip
sha256sum -c data/checksums/ml1m_raw.sha256
```

### Amazon Beauty 2014

Source: [Amazon Reviews 2014](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html).
Use the **full ratings-only Beauty** file, not 5-core or the 2018/2023 releases.
Use the full product metadata file; replacing it with category-only metadata
has not been established to produce identical artifacts.

```bash
mkdir -p data/raw/beauty
wget -c -P data/raw/beauty https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Beauty.csv
wget -c -P data/raw/beauty https://snap.stanford.edu/data/amazon/productGraph/metadata.json.gz
gzip -dk data/raw/beauty/metadata.json.gz
sha256sum -c data/checksums/beauty_raw.sha256
python -m zipfile -c data/raw/beauty/metadata_.json.zip data/raw/beauty/metadata.json
```

The required ZIP member is `metadata.json`. Conversion changes only the
container, not the contents. Allow space for compressed and uncompressed
copies; reference metadata is 10,544,467,811 uncompressed bytes.

### Steam Version 2

Source: [Steam Video Game and Bundle Data](https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data),
**Version 2: Review Data** and **Version 2: Item metadata**, not Version 1.

```bash
mkdir -p data/raw/steam
wget -c -P data/raw/steam https://cseweb.ucsd.edu/~wckang/steam_reviews.json.gz
wget -c -P data/raw/steam https://cseweb.ucsd.edu/~wckang/steam_games.json.gz
gzip -dc data/raw/steam/steam_reviews.json.gz > data/raw/steam/steam_new.json
gzip -dk data/raw/steam/steam_games.json.gz
sha256sum -c data/checksums/steam_raw.sha256
python -m zipfile -c data/raw/steam/steam_new.json.zip data/raw/steam/steam_new.json
```

`steam_new.json` is our local filename for the uncompressed reviews, not a
different release. The ZIP member must have that same name. Reference reviews
are 4,273,159,526 uncompressed bytes. `steam_games.json` remains uncompressed.

For already transferred ZIPs, do not extract gigabytes only to verify them:

```bash
unzip -p data/raw/beauty/metadata_.json.zip metadata.json | sha256sum
unzip -p data/raw/steam/steam_new.json.zip steam_new.json | sha256sum
```

Compare the printed digests with the corresponding `metadata.json` and
`steam_new.json` entries in the raw manifests. ZIP timestamps/compression
settings and unrelated `__MACOSX` members do not affect these content hashes.

### Yambda, likes from the 500M variant

Source: [yandex/yambda at the recorded revision](https://huggingface.co/datasets/yandex/yambda/tree/dd6f3a19eef5866e346c3270e098baa641a44948).
The revision below was recovered from the original downloads' local metadata.
Do not replace it with the moving `main` branch.

```bash
python -m pip install huggingface_hub==1.9.0
HF_ENDPOINT=https://huggingface.co HF_HUB_DISABLE_XET=1 \
hf download yandex/yambda \
  flat/500m/likes.parquet artist_item_mapping.parquet album_item_mapping.parquet \
  README.md LICENSE --repo-type dataset \
  --revision dd6f3a19eef5866e346c3270e098baa641a44948 \
  --local-dir data/raw/yambda --max-workers 3
sha256sum -c data/checksums/yambda_raw.sha256
```

`HF_ENDPOINT` can be changed to a mirror required by your infrastructure;
keep the revision and checksums unchanged. No authentication token is required
for this public dataset. The `500M` source variant includes multiple event types;
its likes file contains **9,033,960** interactions. No listens, audio embeddings
or provider-generated sequences are used.

## 2. Prepare static and sequential artifacts once

In each selected `runs/<dataset>/retrieval/prepare_retrieval.py`, use:

```python
SPLIT_RATIOS = (0.8, 0.1, 0.1)
REBUILD_STATIC = False
REBUILD_SEQUENTIAL = False
PREPARE_SEQUENTIAL = True
```

The public branch defaults to `PREPARE_SEQUENTIAL = False` for all datasets.
Set it to `True` for both scenarios; leave it `False` for static only.
Then run the entry point for each dataset you need:

```bash
python runs/ml1m/retrieval/prepare_retrieval.py
python runs/beauty/retrieval/prepare_retrieval.py
python runs/steam/retrieval/prepare_retrieval.py
python runs/yambda/retrieval/prepare_retrieval.py
```

Outputs go to `data/preprocessed/<dataset>/retrieval/{static,sequential}/`.
Sequential preparation derives its arrays from static artifacts and records
`../static` as a relative reference; it does not reread raw archives.
Training reads these prepared files and never triggers raw preprocessing.

Existing artifacts are reused when `metadata.json` exists. This is a readiness
marker, not a source-content check. If you deliberately need to replace stale
artifacts, set `REBUILD_STATIC = True` (and `PREPARE_SEQUENTIAL = True` to rebuild
its dependent sequences), run preparation once, and reset the rebuild flag.
Do not overwrite artifacts being used by an experiment.

## 3. Verify before training

For example, after preparing Beauty:

```bash
sha256sum -c data/checksums/beauty_static.sha256
sha256sum -c data/checksums/beauty_sequential.sha256
```

Use `ml1m`, `steam`, or `yambda` in the filenames for the other datasets.
Do not use `--ignore-missing`: missing files must fail verification.
For all four datasets, once both modes have been prepared:

```bash
sha256sum -c data/checksums/*_static.sha256 data/checksums/*_sequential.sha256
```

Every listed file must report `OK`, and the command must exit with status 0.
No optional expensive verification step is added to training itself.

## Protocol and expected dimensions

All retained interactions are positives, with no rating threshold, k-core or
deduplication. Steam alone drops reviews without a nonempty raw `user_id`;
it does not fall back to `username`. Steam dates become midnight UTC seconds.
Yambda retains duplicate likes and both organic/non-organic events, using
the original timestamp scale.

Stable timestamp sorting preserves source order within ties. Global 80/10/10
boundaries move right to the end of a timestamp group. User/item vocabularies
come from the full retained interaction corpus, not train alone. Metadata
does not add items to the catalog. Missing item features use `MISSING`.
Sorted category-path collections, genres, artist/album ID lists each become
one scalar category, not a pooled set of tokens. Artist/album IDs sort numerically.

Dense codes, timestamps, numeric audit arrays and prepared hashes use int64.
String IDs/vocabulary values remain strings. Each feature has four namespace-
aware xxHash64 values with seeds 0/1/2/3, restricted to 63 bits; serialization
is `namespace_utf8_nul_typed_int64le_or_utf8_v1`. Training seeds and embedding
table budgets do not change these hashes. Modulo is applied in model forward.

| Dataset | Interactions | Users | Items | Train | Validation | Test |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ML1M | 1,000,209 | 6,040 | 3,706 | 800,168 | 100,020 | 100,021 |
| Beauty | 2,023,070 | 1,210,271 | 249,274 | 1,619,567 | 201,990 | 201,513 |
| Steam | 3,176,223 | 1,485,611 | 14,513 | 2,542,579 | 319,012 | 314,632 |
| Yambda | 9,033,960 | 82,788 | 638,230 | 7,227,170 | 903,394 | 903,396 |

These counts precede evaluation-only first-event exclusion. Both static and
sequential evaluation exclude targets without prior user history, retain cold
users/items otherwise, and score the full catalog including previously seen
items. A single target is evaluated per interaction. Sequential histories use
only earlier interactions in the fixed order, including earlier events of the
same validation/test split. `max_len=200` belongs to the runtime Dataset,
not to preprocessing; no fixed-length histories are saved.

These manifests verify the existing research files, not a newly downloaded
and reprocessed copy. No complete preprocessing rerun was performed while
adding checksums. Successful reconstruction should be verified on the new
machine; if a provider changes or removes a file, use these fingerprints to
identify the correct version rather than silently substituting another release.
For protection against source unavailability, keep a private backup of the
raw/prepared artifacts. Resuming an optimizer state is a separate issue from
reconstructing the data or rerunning the experiment recipe.
