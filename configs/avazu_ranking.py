from config import RankingConfig

config = RankingConfig(
    dataset='avazu',
    emb_dim=32, emb_levels=26_542, cross=1, dnn=(512, 512),
    batch=4096, lr=2e-4, workers=8,
    budgets=(10.0, 1.0, 0.1),
)
