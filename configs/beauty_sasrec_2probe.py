from config import SasrecConfig

config = SasrecConfig(dataset='beauty', max_len=50, only='Multiplex',
                      align_roles=False, probes=2, combine='concat', runs=5)
