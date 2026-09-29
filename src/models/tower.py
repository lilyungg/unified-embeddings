from torch import nn


class Tower(nn.Module):
    """Return the final embedding and feature embeddings before the encoder."""

    def __init__(self, feature_encoder, encoder):
        super().__init__()
        self.feature_encoder = feature_encoder
        self.encoder = encoder

    def forward(self, inputs):
        feature_embeddings = self.feature_encoder(inputs)
        embedding = self.encoder(feature_embeddings)
        return {"embedding": embedding, **feature_embeddings}

    def prepare_training_pairs(self, output, batch):
        """Attach target and interaction IDs without changing the feature embeddings."""
        return {
            **output,
            "target_item_idx": batch["target_item_idx"],
            "interaction_idx": batch["interaction_idx"],
        }
