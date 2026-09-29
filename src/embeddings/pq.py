import torch
from torch import nn


class PQEmbedding(nn.Module):
    """Concatenate K hashed component tables; num_rows is their total row budget."""

    requires_hashes = True

    def __init__(
        self,
        features,
        num_rows,
        embedding_dim,
        num_hashes,
        feature_hashes,
    ):
        super().__init__()
        self.features = tuple(features)
        self.num_hashes = num_hashes
        self.tables = nn.ModuleList()

        rows, remainder = divmod(num_rows, num_hashes)
        for component in range(num_hashes):
            weight = torch.empty(
                rows + (component < remainder),
                embedding_dim,
                dtype=torch.float32,
            )
            nn.init.uniform_(weight, -0.05, 0.05)
            self.tables.append(
                nn.Embedding.from_pretrained(weight, freeze=False, sparse=True)
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
            name: self.num_hashes * self.tables[0].embedding_dim
            for name in self.features
        }

    def forward(self, inputs):
        embeddings = {}

        for i, name in enumerate(self.features):
            if name not in inputs:
                continue

            hashes = getattr(self, f"hashes_{i}")[inputs[name], :self.num_hashes]
            components = [
                table(hashes[..., component] % table.num_embeddings)
                for component, table in enumerate(self.tables)
            ]
            embeddings[name] = torch.cat(components, dim=-1)

        return embeddings
