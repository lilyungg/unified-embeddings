from config import SasrecConfig

config = SasrecConfig(dataset='ml1m', only='Multiplex', align_roles=False,
                      probes=2, combine='concat', runs=5)
