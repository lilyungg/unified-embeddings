"""Pluggable embedding tables for SASRec — the feature-multiplexing axis.

SASRec holds three vocabularies in its embedding layers:
  item_in   items as sequence input   (num_items + 2 rows)
  item_out  items as scoring targets  (num_items + 2 rows; tied to item_in
            when reuse_item_embeddings=True)
  position  sequence positions        (sequence_length rows)

The architecture is untouched; only where the rows live changes:

  Collisionless  every vocabulary gets its exact rows (upper bound)
  Non-multiplex  per-vocabulary hash tables, budget split by vocabulary share
  Multiplex      one shared table, feature-salted hashing (Unified Embedding)

Row codes are precomputed lookup tensors, so the hot path stays a single
nn.Embedding gather — no hashing at train time.
"""
import numpy as np
import torch
import xxhash
from torch import nn
from ue import _mix64


def _hash_codes(n: int, levels: int, feature_id: str, offset: int = 0) -> torch.Tensor:
    """Salted hash of ids [0, n) into `levels` rows, shifted by `offset`."""
    salt = np.uint64(xxhash.xxh32(feature_id, 0).intdigest())
    v = np.arange(n, dtype=np.uint64) + (salt << np.uint64(32))
    codes = (_mix64(v) % np.uint64(levels)).astype(np.int64) + offset
    return torch.from_numpy(codes)


class MultiplexedEmbeddings(nn.Module):
    """Holds one flat table plus per-vocabulary row-code lookups."""

    # Positions are never hashed: seq_t = E_in[x_t] + E_pos[t], so
    # dL/dE_in[x_t] = dL/dseq_t = dL/dE_pos[t] — the two roles share one
    # gradient direction exactly. A position/item collision is therefore an
    # intra-feature collision (value merging), which no reader can undo, and
    # positions cost ~0.3% of the budget anyway. Keep them collisionless.
    NEVER_HASHED = ('position',)

    # Vocabularies that are the SAME tokens in different roles. The paper (§3,
    # "Shared vocabulary") salts each feature by default but notes that for
    # semantically-similar features "performance may slightly improve by using
    # the same hash function". item_in and item_out are not merely similar —
    # they are one vocabulary, so they share a salt: item j's input row IS its
    # output row (weight tying, plus whatever collisions the budget forces).
    ALIGNED_ROLES = {'item_in': 'item', 'item_out': 'item'}

    def __init__(self, sizes: dict, emb_dim: int, method: str, budget: float,
                 base_levels: int = None, align_roles: bool = True) -> None:
        super().__init__()
        self.method, self.sizes = method, sizes
        self.align_roles = align_roles
        hashed = {k: v for k, v in sizes.items() if k not in self.NEVER_HASHED}
        total = sum(hashed.values())
        levels = max(1, round((base_levels or total) * budget))

        codes, offset = {}, 0
        for name in self.NEVER_HASHED:                  # exact rows, always
            if name in sizes:
                codes[name] = torch.arange(sizes[name], dtype=torch.long) + offset
                offset += sizes[name]
        if method == 'Collisionless':
            for name, n in hashed.items():
                codes[name] = torch.arange(n, dtype=torch.long) + offset
                offset += n
            n_rows = offset
        elif method == 'Non-multiplex':
            for name, n in hashed.items():
                lev = max(1, round(n / total * levels))
                codes[name] = _hash_codes(n, lev, name, offset)
                offset += lev
            n_rows = offset
        elif method == 'Multiplex':
            for name, n in hashed.items():          # one shared space
                salt = (self.ALIGNED_ROLES.get(name, name) if align_roles
                        else name)                  # same salt => roles aligned
                codes[name] = _hash_codes(n, levels, salt, offset)
            n_rows = offset + levels
        else:
            raise ValueError(method)

        self.table = nn.Embedding(n_rows, emb_dim)
        nn.init.trunc_normal_(self.table.weight, std=0.02, a=-0.04, b=0.04)
        for name, c in codes.items():
            self.register_buffer(f'codes_{name}', c)
        self.n_rows, self.emb_dim = n_rows, emb_dim

    def rows(self, name: str) -> torch.Tensor:
        return getattr(self, f'codes_{name}')

    def forward(self, name: str, ids: torch.Tensor) -> torch.Tensor:
        return self.table(self.rows(name)[ids.long()])

    def weight_of(self, name: str) -> torch.Tensor:
        """Materialized vocabulary x dim matrix (for full-catalog scoring)."""
        return self.table(self.rows(name))

    def stats(self) -> dict:
        total_vocab = sum(self.sizes.values())
        return {'rows': self.n_rows,
                'size_mb': round(self.n_rows * self.emb_dim * 4 / 1e6, 4),
                'rows_over_vocab': round(self.n_rows / total_vocab, 4),
                'total_vocab': total_vocab}

    @torch.no_grad()
    def emb_l2(self) -> float:
        used = torch.cat([self.rows(n) for n in self.sizes]).unique()
        return self.table.weight[used].norm(dim=1).mean().item()


class _TableView(nn.Module):
    """nn.Embedding-compatible view of one vocabulary, so the untouched
    scoring path (`get_output_embeddings().weight`) keeps working."""

    def __init__(self, tables, name) -> None:
        super().__init__()
        object.__setattr__(self, '_tables', tables)   # no re-registration
        self.name = name

    @property
    def weight(self):
        return self._tables.weight_of(self.name)

    def forward(self, ids):
        return self._tables(self.name, ids)
