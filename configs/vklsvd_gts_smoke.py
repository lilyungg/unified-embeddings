from config import GTSConfig

config = GTSConfig(dataset='vklsvd', subset='up0.001_ip0.001', loss='full',
                   budgets=(1.0,), epochs=2, patience=0, only='Collisionless')
