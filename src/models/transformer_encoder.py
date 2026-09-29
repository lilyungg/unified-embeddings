import torch
from torch import nn


class SequentialEncoder(nn.Module):
    """Causal pre-norm Transformer over right-padded item embeddings."""

    def __init__(
        self, d_model, num_layers, num_heads,
        dim_feedforward, dropout, max_len,
    ):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_len + 1, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.register_buffer("positions", torch.arange(max_len + 1), persistent=False)
        self.register_buffer(
            "causal_mask", torch.ones(max_len + 1, max_len + 1, dtype=torch.bool).triu(1),
            persistent=False,
        )

    def forward(self, item_embeddings, history_length, user_embedding=None):
        """Return [B, L, D] history embeddings, excluding the prefix and zeroing padding."""
        prefix_length = int(user_embedding is not None)
        if user_embedding is not None:
            item_embeddings = torch.cat(
                (user_embedding.unsqueeze(1), item_embeddings), dim=1,
            )

        length = item_embeddings.shape[1]
        positions = self.positions[:length]
        padding_mask = positions[None, :] >= (history_length + prefix_length)[:, None]
        causal_mask = self.causal_mask[:length, :length]

        embedding = self.dropout(item_embeddings + self.position_embeddings(positions))
        embedding = embedding.masked_fill(padding_mask.unsqueeze(-1), 0)
        for layer in self.layers:
            embedding = layer(
                embedding,
                src_mask=causal_mask,
                src_key_padding_mask=padding_mask,
            )
        embedding = self.norm(embedding).masked_fill(padding_mask.unsqueeze(-1), 0)
        return embedding[:, prefix_length:]
