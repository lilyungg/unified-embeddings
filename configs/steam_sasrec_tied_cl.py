from config import SasrecConfig

config = SasrecConfig(dataset='steam', tie_io=True, only='Collisionless',
                      budgets=(1.0,), runs=5)
