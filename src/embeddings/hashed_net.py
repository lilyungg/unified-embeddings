import torch
from torch import nn


ADDRESS_SEED = 0x243F6A8885A308D3
SIGN_SEED = 0x13198A2E03707344
HASH_MASK = (1 << 63) - 1


def _mix64(values):
    """SplitMix64 finalizer with unsigned shifts on int64 bit patterns."""
    values = (values ^ ((values >> 30) & ((1 << 34) - 1))) * -4658895280553007687
    values = (values ^ ((values >> 27) & ((1 << 37) - 1))) * -7723592293110705685
    return values ^ ((values >> 31) & ((1 << 33) - 1))


class HashedNetEmbedding(nn.Module):
    """Concatenate H signed scalar-hash assemblies from one shared pool: [...] -> [..., H*d]."""

    requires_hashes = True

    def __init__(
        self,
        features,
        num_rows,
        embedding_dim,
        feature_hashes,
    ):
        super().__init__()
        self.features = dict(features)
        self.embedding_dim = embedding_dim

        weight = torch.empty(num_rows * embedding_dim, 1, dtype=torch.float32)
        nn.init.uniform_(weight, -0.05, 0.05)
        self.table = nn.Embedding.from_pretrained(weight, freeze=False, sparse=True)

        # Only 2*d fixed coordinate keys; addresses and signs are computed in forward.
        coordinates = torch.arange(embedding_dim, dtype=torch.int64, device=weight.device)
        seeds = torch.tensor([ADDRESS_SEED, SIGN_SEED], dtype=torch.int64, device=weight.device)
        self.register_buffer(
            "coordinate_keys",
            _mix64(seeds[:, None] + coordinates),
            persistent=False,
        )

        for i, name in enumerate(self.features):
            hashes = torch.as_tensor(
                feature_hashes[name],
                dtype=torch.int64,
                device=weight.device,
            )
            self.register_buffer(f"hashes_{i}", hashes, persistent=False)

    @property
    def output_dims(self):
        return {
            name: num_hashes * self.embedding_dim
            for name, num_hashes in self.features.items()
        }

    def forward(self, inputs):
        embeddings = {}

        for i, (name, num_hashes) in enumerate(self.features.items()):
            if name not in inputs:
                continue

            hashes = getattr(self, f"hashes_{i}")
            base = hashes[inputs[name], :num_hashes]
            mixed = _mix64(base[..., None, None] ^ self.coordinate_keys)
            address_hashes, sign_hashes = mixed.unbind(dim=-2)
            indices = (address_hashes & HASH_MASK) % self.table.num_embeddings
            signs = (sign_hashes & 1) * 2 - 1
            parts = self.table(indices).squeeze(-1) * signs
            embeddings[name] = parts.flatten(-2)

        return embeddings
