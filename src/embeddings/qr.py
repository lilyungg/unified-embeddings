import torch
from torch import nn


class QREmbedding(nn.Module):
    """Multiply K mixed-radix lookups over feature-offset dense IDs."""

    requires_hashes = False

    def __init__(self, features, table_rows, embedding_dim):
        super().__init__()
        self.features = dict(features)
        self.offsets = {}
        self.tables = nn.ModuleList()

        offset = 0
        for name, cardinality in self.features.items():
            self.offsets[name] = offset
            offset += cardinality

        for num_rows in table_rows:
            weight = torch.empty(
                num_rows,
                embedding_dim,
                dtype=torch.float32,
            )
            nn.init.uniform_(weight, -0.05, 0.05)
            self.tables.append(
                nn.Embedding.from_pretrained(weight, freeze=False, sparse=True)
            )

    @property
    def output_dims(self):
        return {
            name: self.tables[0].embedding_dim
            for name in self.features
        }

    def forward(self, inputs):
        embeddings = {}

        for name in self.features:
            if name not in inputs:
                continue

            remaining = inputs[name] + self.offsets[name]
            embedding = None
            for table in self.tables:
                component = table(remaining % table.num_embeddings)
                remaining = remaining // table.num_embeddings
                embedding = component if embedding is None else embedding * component

            embeddings[name] = embedding

        return embeddings
