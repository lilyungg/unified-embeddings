from config import SasrecConfig

config = SasrecConfig(dataset='ml1m', tie_io=True, only='Multiplex',
                      probes=2, combine='concat', runs=5)
