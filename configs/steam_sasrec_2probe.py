from config import SasrecConfig

config = SasrecConfig(dataset='steam', only='Multiplex', align_roles=False,
                      probes=2, combine='concat', runs=5)
