from torch import nn

from src.data.item_features import gather_item_features

from .transformer_encoder import SequentialEncoder


class SequentialTower(nn.Module):
    """Return sequence embeddings and history feature embeddings before readers.

    catalog_features holds item-aligned tensors already on the batch device.
    """

    def __init__(
        self, item_tower, catalog_features, *,
        d_model, num_layers, num_heads,
        dim_feedforward, dropout, max_len,
        user_tower=None,
    ):
        super().__init__()
        self.item_tower = item_tower
        self.catalog_features = catalog_features
        self.user_tower = user_tower
        self.encoder = SequentialEncoder(
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=max_len,
        )

    def forward(self, batch):
        history_features = gather_item_features(
            batch["input_item_idx"], self.catalog_features,
        )
        history_output = self.item_tower(history_features)

        user_embedding = None
        if self.user_tower is not None:
            user_embedding = self.user_tower(batch)["embedding"]

        embedding = self.encoder(
            history_output["embedding"],
            batch["history_length"],
            user_embedding=user_embedding,
        )
        return {**history_output, "embedding": embedding}

    def prepare_training_pairs(self, output, batch):
        """Select valid pairs from a forward output; IDs are not feature embeddings."""
        positions = batch["valid_positions"]
        pairs = {
            name: embedding.flatten(0, 1).index_select(0, positions)
            for name, embedding in output.items()
        }
        for name in ("target_item_idx", "interaction_idx"):
            pairs[name] = batch[name].flatten().index_select(0, positions)
        return pairs
