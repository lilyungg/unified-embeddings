from config import GTSConfig

config = GTSConfig(dataset='yambda_50m', interaction='likes',
                   epochs=80, patience=8, runs=5)
