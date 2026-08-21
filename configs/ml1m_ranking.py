from config import RankingConfig

config = RankingConfig(
    dataset='movielens', ml1m='ml-1m', ml_labels='wang',
    emb_dim=30, emb_levels=13_653, cross=1, dnn=(192,),
    batch=128, lr=1e-3,
    budgets=(1.0, 0.5, 0.1),
)
