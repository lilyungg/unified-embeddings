from config import GTSConfig

config = GTSConfig(dataset='yambda_50m', interaction='likes',
                   budgets=(1.0,), epochs=2, patience=0, only='Multiplex')
