import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from runs.retrieval.train import parse_args
from runs.retrieval.utils import run


CONFIG = {
    "dataset": "beauty",
    "approach": "hashing_trick",
    "loss": "full_ce",
    "budget_fraction": 0.1,
    "num_hashes": 4,
    "num_sampled_negatives": 256,
    "patience": 5,
}


def main(argv=None):
    overrides, output_dir = parse_args(argv, model="static")
    return run({**CONFIG, **overrides}, output_dir=output_dir)


if __name__ == "__main__":
    main()
