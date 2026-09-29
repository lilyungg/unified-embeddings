from torch import nn


class FeatureRegularisationLoss(nn.Module):
    """Sum weighted squared feature norms before readers; ignore 'embedding'.

    The run supplies 1/H for Hashing Trick concat, otherwise 1; embedding_l2 stays outside.
    Select valid sequential positions before averaging.
    """

    def __init__(self, feature_weights):
        super().__init__()
        self.feature_weights = dict(feature_weights)

    def per_example(self, output):
        """Return per-object penalties, preserving the leading dimensions."""
        penalty = None

        for name, embedding in output.items():
            if name == "embedding":
                continue

            current = embedding.square().sum(dim=-1) * self.feature_weights[name]
            penalty = current if penalty is None else penalty + current

        return penalty

    def forward(self, output):
        return self.per_example(output).mean()
