from config import SasrecConfig

config = SasrecConfig(dataset='yambda_50m', batch=32, tie_io=True,
                      only='Multiplex', probes=2, combine='concat',
                      budgets=(1.0,), runs=5)
