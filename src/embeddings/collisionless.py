import torch
from torch import nn


class CollisionlessEmbedding(nn.Module):
    requires_hashes = False

    def __init__(self, features, num_rows, embedding_dim):
        super().__init__()
        self.features = tuple(features)

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

    @property
    def output_dims(self):
        return {
            name: self.table.embedding_dim
            for name in self.features
        }

    def forward(self, inputs):
        return {
            name: self.table(inputs[name])
            for name in self.features
            if name in inputs
        }
