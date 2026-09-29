from torch import nn


class LinearFeatureEncoder(nn.Module):
    """Sum per-feature linear projections while preserving leading dimensions."""

    def __init__(self, input_dims, output_dim):
        super().__init__()

        self.readers = nn.ModuleDict({
            name: nn.Linear(dim, output_dim, bias=False)
            for name, dim in input_dims.items()
        })

    def forward(self, feature_embeddings):
        embedding = None

        for name, reader in self.readers.items():
            current = reader(feature_embeddings[name])
            embedding = current if embedding is None else embedding + current

        return embedding
