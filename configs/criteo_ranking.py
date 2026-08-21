from config import RankingConfig

config = RankingConfig(
    dataset='criteo',
    emb_dim=39, emb_levels=83_886, cross=2, dnn=(748, 748),
    batch=4096, lr=2e-4, workers=8,
    budgets=(2.0, 1.0, 0.2),
)
