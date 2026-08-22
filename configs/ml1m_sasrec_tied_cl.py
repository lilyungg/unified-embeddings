from config import SasrecConfig

config = SasrecConfig(dataset='ml1m', tie_io=True, only='Collisionless',
                      budgets=(1.0,), runs=5)
