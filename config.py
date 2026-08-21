from dataclasses import dataclass


@dataclass
class RankingConfig:
    dataset: str
    emb_dim: int
    emb_levels: int
    cross: int
    dnn: tuple
    batch: int
    lr: float
    ml1m: str = 'ml-1m'
    ml_labels: str = 'wang'
    avazu: str = None
    criteo: str = None
    epochs: int = 30
    patience: int = 5
    wd: float = 1e-5
    dropout: float = 0.0
    bn: bool = False
    budgets: tuple = (1.0,)
    runs: int = 1
    seed: int = 42
    only: str = None
    with_mlp: bool = False
    workers: int = 4
    paper_protocol: bool = False
    fast: bool = False
    probes: int = 1
    tb: str = 'runs'
    out: str = 'experiment_logs'


@dataclass
class CandgenConfig:
    dataset: str
    ml1m: str = 'ml-1m'
    budgets: tuple = (1.0, 0.5, 0.1)
    base_levels: int = None
    emb_dim: int = 30
    k: int = 32
    batch: int = 1024
    lr: float = 1e-3
    epochs: int = 20
    patience: int = 3
    runs: int = 1
    seed: int = 42
    loss: str = 'full'
    score: str = 'dot'
    temp: float = 0.05
    only: str = None
    worst_init: bool = False
    probes: int = 1
    out: str = 'experiment_logs'


@dataclass
class SasrecConfig:
    dataset: str
    subset: str = 'ur0.01_ip0.01'
    positive: str = 'like'
    budgets: tuple = (1.0, 0.5, 0.1)
    emb_dim: int = 128
    max_len: int = 200
    heads: int = 1
    blocks: int = 2
    dropout: float = 0.5
    batch: int = 128
    batches_per_epoch: int = 100
    lr: float = 1e-3
    epochs: int = 300
    patience: int = 20
    tie_io: bool = False
    align_roles: bool = True
    probes: int = 1
    combine: str = 'concat'
    tied_baseline: bool = True
    runs: int = 1
    seed: int = 42
    only: str = None
    out: str = 'experiment_logs'


@dataclass
class GTSConfig:
    dataset: str
    subset: str = 'up0.001_ip0.001'
    positive: str = 'like'
    interaction: str = 'multi'
    budgets: tuple = (1.0, 0.5, 0.1)
    base_levels: int = None
    emb_dim: int = 30
    k: int = 32
    batch: int = 4096
    lr: float = 1e-3
    epochs: int = 30
    patience: int = 5
    batches_per_epoch: int = 0
    loss: str = 'sampled'
    runs: int = 1
    seed: int = 42
    only: str = None
    probes: int = 1
    tb: str = 'runs'
    out: str = 'experiment_logs'


@dataclass
class OrthogonalityConfig:
    ml1m: str = 'ml-1m'
    emb_levels: int = 13_653
    emb_dim: int = 30
    budgets: tuple = (2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01)
    epochs: int = 10
    batch: int = 512
    lr: float = 1e-3
    out: str = 'experiment_logs'
