import torch
import torch.nn as nn
import xxhash
import numpy as np


def _mix64(x: np.ndarray) -> np.ndarray:
    # splitmix64 finalizer — vectorized uniform mixing for integer codes
    x = x.astype(np.uint64, copy=True)
    x += np.uint64(0x9E3779B97F4A7C15)
    x ^= x >> np.uint64(30)
    x *= np.uint64(0xBF58476D1CE4E5B9)
    x ^= x >> np.uint64(27)
    x *= np.uint64(0x94D049BB133111EB)
    x ^= x >> np.uint64(31)
    return x


def prehash(values: np.ndarray, seeds: tuple, emb_levels: int,
            feature_id: str = "") -> np.ndarray:
    values = np.asarray(values)
    out = np.empty((len(values), len(seeds)), dtype=np.int64)
    if np.issubdtype(values.dtype, np.integer):
        # fast path for pre-encoded datasets: salt integer codes with the
        # feature id in the high bits, then mix (string xxhash on web-scale
        # data takes hours; this is vectorized and equally uniform)
        salt = np.uint64(xxhash.xxh32(feature_id, 0).intdigest())
        v = values.astype(np.uint64)
        for j, seed in enumerate(seeds):
            salted = v + ((salt + np.uint64(seed)) << np.uint64(32))
            out[:, j] = (_mix64(salted) % np.uint64(emb_levels)).astype(np.int64)
        return out
    prefix = feature_id + ":" if feature_id else ""
    for j, seed in enumerate(seeds):
        out[:, j] = np.vectorize(
            lambda v, s=seed: xxhash.xxh32(prefix + str(v), s).intdigest() % emb_levels
        )(values)
    return out


def build_vocabs(df, cols: list) -> dict:
    vocabs = {}
    for col in cols:
        values = df[col].to_numpy()
        if np.issubdtype(values.dtype, np.integer):
            # pre-encoded dense codes 0..K-1: vocab spans the full code range
            # (a subsample may miss codes; preencode maps code -> code+1)
            vocabs[col] = {str(i): i + 1 for i in range(int(values.max()) + 1)}
        else:
            vals = sorted(df[col].unique().to_list(), key=str)
            vocabs[col] = {str(v): i + 1 for i, v in enumerate(vals)}
    return vocabs


def preencode(df, cols: list, vocabs: dict) -> np.ndarray:
    parts = []
    for col in cols:
        values = df[col].to_numpy()
        if np.issubdtype(values.dtype, np.integer):
            # pre-encoded datasets are already dense 0..K-1 codes; shift by 1
            # to reserve 0 for OOV, matching the vs+1 collisionless table
            encoded = values.astype(np.int64) + 1
        else:
            vocab = vocabs[col]
            encoded = np.array(
                [vocab.get(str(v), 0) for v in values], dtype=np.int64
            )
        parts.append(encoded.reshape(-1, 1))
    return np.concatenate(parts, axis=1)


def prehash_split(df, cols: list, levels: list) -> np.ndarray:
    parts = []
    for col, lev in zip(cols, levels):
        values = df[col].to_numpy()
        if np.issubdtype(values.dtype, np.integer):
            hashed = (_mix64(values.astype(np.uint64))
                      % np.uint64(lev)).astype(np.int64)
        else:
            hashed = np.vectorize(
                lambda v: xxhash.xxh32(str(v), 0).intdigest() % lev
            )(values)
        parts.append(hashed.reshape(-1, 1))
    return np.concatenate(parts, axis=1)


def _init_embeddings(module: nn.Module):
    # Keras-style init (paper is TF2): uniform(-0.05, 0.05); PyTorch default N(0,1)
    # starts embeddings ~100x too large and wastes epochs shrinking them
    for m in module.modules():
        if isinstance(m, nn.Embedding):
            nn.init.uniform_(m.weight, -0.05, 0.05)


def embedding_table_stats(module: nn.Module, total_vocab: int = 0) -> dict:
    rows, params = 0, 0
    for m in module.modules():
        if isinstance(m, nn.Embedding):
            rows += m.num_embeddings
            params += m.weight.numel()
    stats = {
        "rows": rows,
        "params": params,
        "size_mb": round(params * 4 / 1e6, 4),
    }
    if total_vocab:
        stats["rows_over_vocab"] = round(rows / total_vocab, 4)
    return stats


def embedding_l2_mean(module: nn.Module) -> float:
    total, rows = 0.0, 0
    with torch.no_grad():
        for m in module.modules():
            if isinstance(m, nn.Embedding):
                total += m.weight.norm(dim=1).sum().item()
                rows += m.num_embeddings
    return total / max(rows, 1)


class CollisionlessEmbedding(nn.Module):
    def __init__(self, vocab_sizes: list, emb_dim: int):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vs + 1, emb_dim) for vs in vocab_sizes]
        )
        _init_embeddings(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = [emb(x[:, i]) for i, emb in enumerate(self.embeddings)]
        return torch.cat(parts, dim=1)


class NonMultiplexedEmbedding(nn.Module):
    def __init__(self, vocab_sizes: list, total_emb_levels: int, emb_dim: int):
        super().__init__()
        total_vocab = sum(vocab_sizes)
        self.levels = [
            max(1, round(vs / total_vocab * total_emb_levels)) for vs in vocab_sizes
        ]
        self.tables = nn.ModuleList(
            [nn.Embedding(lev, emb_dim) for lev in self.levels]
        )
        _init_embeddings(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = [tbl(x[:, i]) for i, tbl in enumerate(self.tables)]
        return torch.cat(parts, dim=1)


class UnifiedEmbedding(nn.Module):
    def __init__(self, emb_levels: int, emb_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(emb_levels, emb_dim)
        _init_embeddings(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x).reshape(x.size(0), -1)
