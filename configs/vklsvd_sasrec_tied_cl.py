from config import SasrecConfig

config = SasrecConfig(dataset='vklsvd', subset='ur0.01_ip0.01', batch=32,
                      tie_io=True, only='Collisionless', budgets=(1.0,), runs=5)
