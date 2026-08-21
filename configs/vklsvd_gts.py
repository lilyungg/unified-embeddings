from config import GTSConfig

config = GTSConfig(dataset='vklsvd', subset='ur0.01_ip0.01', positive='like',
                   batches_per_epoch=2000, epochs=80, patience=8, runs=5)
