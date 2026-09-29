import torch
from torch import nn


class HashingTrickEmbedding(nn.Module):
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

        weight = torch.empty(
            num_rows,
            embedding_dim,
            dtype=torch.float32,
        )
        nn.init.uniform_(weight, -0.05, 0.05)

        self.table = nn.Embedding.from_pretrained(
            weight,
            freeze=False,
            sparse=True,
        )

        for i, name in enumerate(self.features):
            hashes = torch.as_tensor(
                feature_hashes[name],
                dtype=torch.int64,
                device=weight.device,
            )
            self.register_buffer(
                f"hashes_{i}",
                hashes,
                persistent=False,
            )

    @property
    def output_dims(self):
        return {
            name: num_hashes * self.table.embedding_dim
            for name, num_hashes in self.features.items()
        }

    def forward(self, inputs):
        embeddings = {}

        for i, (name, num_hashes) in enumerate(self.features.items()):
            if name not in inputs:
                continue

            hashes = getattr(self, f"hashes_{i}")
            rows = hashes[inputs[name], :num_hashes] % self.table.num_embeddings
            embeddings[name] = self.table(rows).flatten(-2)

        return embeddings
