from config import SasrecConfig

config = SasrecConfig(dataset='beauty', max_len=50, tie_io=True,
                      only='Collisionless', budgets=(1.0,), runs=5)
