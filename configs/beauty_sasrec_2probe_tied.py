from config import SasrecConfig

config = SasrecConfig(dataset='beauty', max_len=50, tie_io=True, only='Multiplex',
                      probes=2, combine='concat', runs=5)
