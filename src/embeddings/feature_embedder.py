from torch import nn

from .collisionless import CollisionlessEmbedding
from .hash_embedding import HashEmbedding
from .hashed_net import HashedNetEmbedding
from .hashing_trick import HashingTrickEmbedding
from .pq import PQEmbedding
from .qr import QREmbedding
from .robe_z import ROBEZEmbedding


EMBEDDING_METHODS = {
    "collisionless": CollisionlessEmbedding,
    "hashing_trick": HashingTrickEmbedding,
    "hash_embedding": HashEmbedding,
    "pq": PQEmbedding,
    "qr": QREmbedding,
    "hashed_net": HashedNetEmbedding,
    "robe_z": ROBEZEmbedding,
}


class FeatureEmbedder(nn.Module):
    def __init__(self, tables, feature_hashes):
        super().__init__()
        self.embedders = nn.ModuleList()

        for table in tables:
            settings = dict(table)
            method_class = EMBEDDING_METHODS[settings.pop("method")]

            if method_class.requires_hashes:
                settings["feature_hashes"] = feature_hashes

            self.embedders.append(method_class(**settings))

    @property
    def output_dims(self):
        return {
            name: dim
            for embedder in self.embedders
            for name, dim in embedder.output_dims.items()
        }

    def forward(self, inputs):
        embeddings = {}

        for embedder in self.embedders:
            embeddings.update(embedder(inputs))

        return embeddings
