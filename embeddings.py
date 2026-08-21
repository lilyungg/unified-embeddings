import numpy as np
import torch
import xxhash
from torch import nn


def _mix64(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.uint64, copy=True)
    x += np.uint64(0x9E3779B97F4A7C15)
    x ^= x >> np.uint64(30)
    x *= np.uint64(0xBF58476D1CE4E5B9)
    x ^= x >> np.uint64(27)
    x *= np.uint64(0x94D049BB133111EB)
    x ^= x >> np.uint64(31)
    return x


def prehash(values: np.ndarray, seeds: tuple, emb_levels: int,
            feature_id: str = '') -> np.ndarray:
    values = np.asarray(values)
    out = np.empty((len(values), len(seeds)), dtype=np.int64)
    if np.issubdtype(values.dtype, np.integer):
        salt = np.uint64(xxhash.xxh32(feature_id.encode(), 0).intdigest())
        v = values.astype(np.uint64)
        for j, seed in enumerate(seeds):
            salted = v + ((salt + np.uint64(seed)) << np.uint64(32))
            out[:, j] = (_mix64(salted) % np.uint64(emb_levels)).astype(np.int64)
        return out
    prefix = feature_id + ':' if feature_id else ''
    for j, seed in enumerate(seeds):
        out[:, j] = np.vectorize(
            lambda v, s=seed: xxhash.xxh32((prefix + str(v)).encode(), s).intdigest() % emb_levels
        )(values)
    return out


def build_vocabs(df, cols: list) -> dict:
    vocabs = {}
    for col in cols:
        values = df[col].to_numpy()
        if np.issubdtype(values.dtype, np.integer):
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
        salt = np.uint64(xxhash.xxh32(col.encode(), 0).intdigest())
        if np.issubdtype(values.dtype, np.integer):
            salted = values.astype(np.uint64) + (salt << np.uint64(32))
            hashed = (_mix64(salted) % np.uint64(lev)).astype(np.int64)
        else:
            hashed = np.vectorize(
                lambda v, p=col + ':': xxhash.xxh32((p + str(v)).encode(), 0).intdigest() % lev
            )(values)
        parts.append(hashed.reshape(-1, 1))
    return np.concatenate(parts, axis=1)


def _init_embeddings(module: nn.Module) -> None:
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
        'rows': rows,
        'params': params,
        'size_mb': round(params * 4 / 1e6, 4),
    }
    if total_vocab:
        stats['rows_over_vocab'] = round(rows / total_vocab, 4)
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
    def __init__(self, vocab_sizes: list, emb_dim: int) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vs + 1, emb_dim) for vs in vocab_sizes]
        )
        _init_embeddings(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = [emb(x[:, i]) for i, emb in enumerate(self.embeddings)]
        return torch.cat(parts, dim=1)


class NonMultiplexedEmbedding(nn.Module):
    def __init__(self, vocab_sizes: list, total_emb_levels: int, emb_dim: int) -> None:
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
    def __init__(self, emb_levels: int, emb_dim: int, probes: int = 1) -> None:
        super().__init__()
        if emb_dim % probes:
            raise ValueError('emb_dim must be divisible by probes')
        self.probes = probes
        self.embedding = nn.Embedding(emb_levels * probes, emb_dim // probes)
        _init_embeddings(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x).reshape(x.size(0), -1)


class MultiplexedEmbeddings(nn.Module):

    NEVER_HASHED = ('position',)

    ALIGNED_ROLES = {'item_in': 'item', 'item_out': 'item'}

    def __init__(self, sizes: dict, emb_dim: int, method: str, budget: float,
                 base_levels: int = None, align_roles: bool = True,
                 probes: int = 1, combine: str = 'concat') -> None:
        super().__init__()
        if probes > 1 and method != 'Multiplex':
            raise ValueError('multi-probe is defined for Multiplex only')
        if combine == 'concat' and emb_dim % probes:
            raise ValueError('emb_dim must be divisible by probes for concat')
        self.method, self.sizes = method, sizes
        self.align_roles, self.probes, self.combine = align_roles, probes, combine
        width = emb_dim // probes if combine == 'concat' else emb_dim
        row_scale = probes if combine == 'concat' else 1

        hashed = {k: v for k, v in sizes.items() if k not in self.NEVER_HASHED}
        total = sum(hashed.values())
        levels = max(1, round((base_levels or total) * budget))

        codes, offset = {}, 0
        for name in self.NEVER_HASHED:
            if name in sizes:
                n = sizes[name]
                if combine == 'concat':
                    c = np.arange(n * probes).reshape(n, probes)
                else:
                    c = np.repeat(np.arange(n), probes).reshape(n, probes)
                codes[name] = torch.from_numpy(c + offset)
                offset += n * row_scale
        if method == 'Collisionless':
            for name, n in hashed.items():
                codes[name] = torch.arange(n, dtype=torch.long).reshape(n, 1) + offset
                offset += n
            n_rows = offset
        elif method == 'Non-multiplex':
            for name, n in hashed.items():
                c = prehash(np.arange(n), (0,), lev := max(1, round(n / total * levels)),
                            feature_id=name) + offset
                codes[name] = torch.from_numpy(c)
                offset += lev
            n_rows = offset
        elif method == 'Multiplex':
            pool = levels * row_scale
            for name, n in hashed.items():
                salt = (self.ALIGNED_ROLES.get(name, name) if align_roles else name)
                c = prehash(np.arange(n), tuple(range(probes)), pool,
                            feature_id=salt) + offset
                codes[name] = torch.from_numpy(c)
            n_rows = offset + pool
        else:
            raise ValueError(method)

        self.table = nn.Embedding(n_rows, width)
        nn.init.trunc_normal_(self.table.weight, std=0.02, a=-0.04, b=0.04)
        for name, c in codes.items():
            self.register_buffer(f'codes_{name}', c)
        self.n_rows, self.emb_dim, self.width = n_rows, emb_dim, width

    def rows(self, name: str) -> torch.Tensor:
        return getattr(self, f'codes_{name}')

    def _combine(self, e: torch.Tensor) -> torch.Tensor:
        if self.combine == 'concat':
            return e.reshape(*e.shape[:-2], -1)
        return e.mean(dim=-2)

    def forward(self, name: str, ids: torch.Tensor) -> torch.Tensor:
        return self._combine(self.table(self.rows(name)[ids.long()]))

    def weight_of(self, name: str) -> torch.Tensor:
        return self._combine(self.table(self.rows(name)))

    def stats(self) -> dict:
        total_vocab = sum(self.sizes.values())
        return {'rows': self.n_rows,
                'size_mb': round(self.n_rows * self.width * 4 / 1e6, 4),
                'rows_over_vocab': round(self.n_rows / total_vocab, 4),
                'total_vocab': total_vocab}

    @torch.no_grad()
    def emb_l2(self) -> float:
        used = torch.cat([self.rows(n).flatten() for n in self.sizes]).unique()
        return self.table.weight[used].norm(dim=1).mean().item()


class _TableView(nn.Module):

    def __init__(self, tables, name) -> None:
        super().__init__()
        object.__setattr__(self, '_tables', tables)
        self.name = name

    @property
    def weight(self):
        return self._tables.weight_of(self.name)

    def forward(self, ids):
        return self._tables(self.name, ids)
