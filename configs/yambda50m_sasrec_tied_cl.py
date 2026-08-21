from config import SasrecConfig

config = SasrecConfig(dataset='yambda_50m', batch=32, tie_io=True,
                      only='Collisionless', budgets=(1.0,), runs=5)
