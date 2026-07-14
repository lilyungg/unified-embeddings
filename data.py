import pathlib
import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

CACHE_DIR = pathlib.Path("datasets")

# Every loader returns (df_categorical, dense_or_None, labels, tr, va, te):
# dense is a float32 matrix of normalized continuous features (Criteo only).


class EmbDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, dense: np.ndarray = None):
        self.x = torch.from_numpy(x).long()
        self.y = torch.from_numpy(y).float()
        self.dense = (torch.from_numpy(dense).float() if dense is not None
                      else torch.empty((len(self.x), 0)))

    def __len__(self): return len(self.x)

    def __getitem__(self, i): return self.x[i], self.dense[i], self.y[i]


def random_split(n: int, train: float = 0.8, val: float = 0.1, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    t = int(train * n)
    v = int((train + val) * n)
    return idx[:t], idx[t:v], idx[v:]


def temporal_split(n: int, train: float = 6 / 7, seed: int = 42):
    """Paper's Criteo protocol (DCN-V2): first 6 days train, day 7 split
    50/50 into val/test randomly. The Kaggle file has no day column but is
    chronological, so day boundaries are approximated by row count."""
    cut  = int(train * n)
    rng  = np.random.default_rng(seed)
    tail = rng.permutation(np.arange(cut, n))
    half = len(tail) // 2
    return np.arange(cut), tail[:half], tail[half:]


def load_movielens(path: str, label_mode: str = "wang") -> tuple:
    # label_mode="wang": preprocessing of Wang et al. 2021 (DCN-V2), which the
    # Unified Embedding paper follows — ratings 1-2 -> 0, 4-5 -> 1, 3s removed.
    # label_mode="ge3": rating >= 3 -> 1, others 0, nothing removed.
    path = pathlib.Path(path)
    user_info = {}
    with open(path / "users.dat") as f:
        for line in f:
            uid, gender, age, occ, zip_ = line.strip().split("::")
            user_info[uid] = (gender, age, occ, zip_)
    rows = []
    with open(path / "ratings.dat") as f:
        for line in f:
            uid, mid, rating, _ = line.strip().split("::")
            rating = int(rating)
            if label_mode == "wang":
                if rating == 3:
                    continue
                label = int(rating >= 4)
            else:
                label = int(rating >= 3)
            gender, age, occ, zip_ = user_info[uid]
            rows.append({
                "user_id": uid, "movie_id": mid,
                "gender": gender, "age": age, "occupation": occ, "zip": zip_,
                "label": label,
            })
    df     = pl.DataFrame(rows)
    labels = df["label"].to_numpy().astype(np.float32)
    cols   = ["user_id", "movie_id", "gender", "age", "occupation", "zip"]
    tr, va, te = random_split(len(df))
    return df.select(cols), None, labels, tr, va, te


def load_avazu(path: str = None, n_rows: int = None) -> tuple:
    prepared = CACHE_DIR / "avazu_prepared.parquet"
    if path and path.endswith(".parquet"):
        prepared = pathlib.Path(path)
    if prepared.exists():
        # prepare_data.py output: int32 codes, Table 5 pruning, hour-of-day.
        # Paper split (AutoInt): shuffle, 10% test; we carve val from the
        # remainder -> 81/9/10.
        df = pl.read_parquet(prepared)
        if n_rows:
            df = df.sample(n=n_rows, seed=42)
        labels = df["label"].to_numpy()
        df     = df.drop("label")
        tr, va, te = random_split(len(df), train=0.81, val=0.09)
        return df, None, labels, tr, va, te

    if path is not None:
        df = pl.read_csv(path, infer_schema_length=10_000,
                         schema_overrides={"id": pl.Utf8})
    else:
        cache = CACHE_DIR / "avazu.parquet"
        if cache.exists():
            df = pl.read_parquet(cache)
        else:
            from datasets import load_dataset
            df = pl.DataFrame(load_dataset("reczoo/Avazu_x4", split="train").to_dict())
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            df.write_parquet(cache)
    if n_rows:
        df = df.sample(n=n_rows, seed=42)

    labels = df["click"].cast(pl.Float32).to_numpy()
    cols   = [c for c in df.columns if c not in ("id", "click")]
    df     = df.select(cols).with_columns(pl.all().cast(pl.Utf8))
    tr, va, te = random_split(len(df))
    return df, None, labels, tr, va, te


def load_criteo(n_rows: int = None, path: str = None) -> tuple:
    prepared = pathlib.Path(path) if path else CACHE_DIR / "criteo_prepared.parquet"
    if prepared.exists():
        # prepare_data.py output: int32 codes (Table 4 pruning) + log-normalized
        # I1-I13 + label, in original (chronological) row order.
        df = pl.read_parquet(prepared)
        n  = len(df)
        if n_rows and n_rows < n:
            # keep chronology for the temporal split: contiguous head sample
            df = df.head(n_rows)
        labels = df["label"].to_numpy()
        dense  = df.select([f"I{i}" for i in range(1, 14)]).to_numpy()
        cat    = df.select([f"C{i}" for i in range(1, 27)])
        tr, va, te = temporal_split(len(cat))
        return cat, dense, labels, tr, va, te

    cache = CACHE_DIR / "criteo.parquet"
    if cache.exists():
        df = pl.read_parquet(cache)
    else:
        from datasets import load_dataset
        df = pl.DataFrame(load_dataset("reczoo/Criteo_x4", split="train").to_dict())
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.write_parquet(cache)
    if n_rows:
        df = df.sample(n=n_rows, seed=42)

    labels = df["Label"].cast(pl.Float32).to_numpy()
    cols   = [f"C{i}" for i in range(1, 27)]
    df     = df.select(cols).with_columns(pl.all().cast(pl.Utf8).fill_null("nan"))
    tr, va, te = random_split(len(df))
    return df, None, labels, tr, va, te
