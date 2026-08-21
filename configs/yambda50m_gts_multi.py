from config import GTSConfig

config = GTSConfig(dataset='yambda_50m', interaction='multi',
                   batches_per_epoch=2000, epochs=80, patience=8, runs=5)
