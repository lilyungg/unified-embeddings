import torch
import xxhash
from torch import nn


IMPORTANCE_HASH_SEED = 4
HASH_MASK = (1 << 63) - 1


class HashEmbedding(nn.Module):
    """Map flat feature inputs [...] to learned weighted-sum embeddings [..., d]."""

    requires_hashes = True

    def __init__(
        self,
        features,
        component_rows,
        importance_rows,
        embedding_dim,
        num_hashes,
        feature_hashes,
    ):
        super().__init__()
        self.features = tuple(features)
        self.num_hashes = num_hashes

        component_weight = torch.empty(component_rows, embedding_dim, dtype=torch.float32)
        nn.init.uniform_(component_weight, -0.05, 0.05)
        self.component_table = nn.Embedding.from_pretrained(
            component_weight,
            freeze=False,
            sparse=True,
        )

        importance_weight = torch.empty(importance_rows, num_hashes, dtype=torch.float32)
        nn.init.normal_(importance_weight, mean=0, std=num_hashes ** -0.5)
        self.importance_table = nn.Embedding.from_pretrained(
            importance_weight,
            freeze=False,
            sparse=True,
        )

        for i, name in enumerate(self.features):
            hashes = torch.as_tensor(
                feature_hashes[name],
                dtype=torch.int64,
                device=component_weight.device,
            )
            # Rehash full component-0 values as uint64 little-endian, before modulo.
            importance_hashes = torch.tensor(
                [
                    xxhash.xxh64(
                        value.to_bytes(8, byteorder="little", signed=False),
                        seed=IMPORTANCE_HASH_SEED,
                    ).intdigest() & HASH_MASK
                    for value in hashes[:, 0].tolist()
                ],
                dtype=torch.int64,
                device=component_weight.device,
            )
            self.register_buffer(f"hashes_{i}", hashes, persistent=False)
            self.register_buffer(f"importance_hashes_{i}", importance_hashes, persistent=False)

    @property
    def output_dims(self):
        return {name: self.component_table.embedding_dim for name in self.features}

    def forward(self, inputs):
        embeddings = {}

        for i, name in enumerate(self.features):
            if name not in inputs:
                continue

            hashes = getattr(self, f"hashes_{i}")
            importance_hashes = getattr(self, f"importance_hashes_{i}")
            component_idx = hashes[inputs[name], :self.num_hashes] % self.component_table.num_embeddings
            importance_idx = importance_hashes[inputs[name]] % self.importance_table.num_embeddings
            components = self.component_table(component_idx)
            weights = self.importance_table(importance_idx)
            embeddings[name] = (components * weights.unsqueeze(-1)).sum(dim=-2)

        return embeddings
