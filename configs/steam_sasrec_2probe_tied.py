from config import SasrecConfig

config = SasrecConfig(dataset='steam', tie_io=True, only='Multiplex',
                      probes=2, combine='concat', budgets=(1.0,), runs=5)
