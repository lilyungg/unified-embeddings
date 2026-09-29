import torch
from torch import nn


PRIME = (1 << 61) - 1
FEATURE_COEFFICIENT = 418579384120320914
VALUE_COEFFICIENT = 2248908517734713837
BLOCK_COEFFICIENT = 346858970817626714
HASH_BIAS = 610822009286219984
MASK31 = (1 << 31) - 1
MASK30 = (1 << 30) - 1


def _multiply_mod_prime(values, coefficient):
    """Multiply residues in [0, PRIME) without overflowing int64."""
    low, high = values & MASK31, values >> 31
    coefficient_low, coefficient_high = coefficient & MASK31, coefficient >> 31
    low_product = low * coefficient_low
    cross_product = low * coefficient_high + high * coefficient_low
    high_product = high * coefficient_high

    # 2**61 == 1 (mod PRIME); fold the split product before it can overflow.
    result = (
        (low_product & PRIME) + (low_product >> 61)
        + ((cross_product & MASK30) << 31) + (cross_product >> 30)
        + (high_product << 1)
    )
    result = (result & PRIME) + (result >> 61)
    return torch.where(result >= PRIME, result - PRIME, result)


class ROBEZEmbedding(nn.Module):
    """Concatenate H block-based assemblies from one circular pool: [...] -> [..., H*d]."""

    requires_hashes = False

    def __init__(
        self,
        features,
        num_rows,
        embedding_dim,
        feature_ids,
        block_size=8,
    ):
        super().__init__()
        self.features = dict(features)
        self.embedding_dim = embedding_dim

        weight = torch.empty(num_rows * embedding_dim, 1, dtype=torch.float32)
        nn.init.uniform_(weight, -0.05, 0.05)
        self.table = nn.Embedding.from_pretrained(weight, freeze=False, sparse=True)
        self.register_buffer(
            "block_offsets",
            torch.arange(block_size, dtype=torch.int64, device=weight.device),
            persistent=False,
        )

        # Only feature/block terms are cached, never vocabulary-wide addresses.
        for i, (name, num_hashes) in enumerate(self.features.items()):
            num_blocks = (num_hashes * embedding_dim + block_size - 1) // block_size
            block_terms = [
                (FEATURE_COEFFICIENT * feature_ids[name] + BLOCK_COEFFICIENT * block + HASH_BIAS) % PRIME
                for block in range(num_blocks)
            ]
            self.register_buffer(
                f"block_terms_{i}",
                torch.tensor(block_terms, dtype=torch.int64, device=weight.device),
                persistent=False,
            )

    @property
    def output_dims(self):
        return {
            name: num_hashes * self.embedding_dim
            for name, num_hashes in self.features.items()
        }

    def forward(self, inputs):
        embeddings = {}
        pool_size = self.table.num_embeddings

        for i, (name, num_hashes) in enumerate(self.features.items()):
            if name not in inputs:
                continue

            base = _multiply_mod_prime(inputs[name], VALUE_COEFFICIENT)
            hashes = base[..., None] + getattr(self, f"block_terms_{i}")
            hashes = torch.where(hashes >= PRIME, hashes - PRIME, hashes)
            indices = ((hashes % pool_size)[..., None] + self.block_offsets) % pool_size
            parts = self.table(indices).squeeze(-1)
            embeddings[name] = parts.flatten(-2)[..., :num_hashes * self.embedding_dim]

        return embeddings
