import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from runs.retrieval.train import parse_args
from runs.retrieval.utils import run


CONFIG = {
    "dataset": "steam",
    "approach": "hashing_trick",
    "budget_fraction": 0.1,
    "num_hashes": 4,
    "patience": 5,
}


def main(argv=None):
    overrides, output_dir = parse_args(argv, model="sequential", loss="bpr")
    config = {**CONFIG, **overrides}
    # Full-catalog metrics over all train positions need a smaller batch on Beauty.
    config.setdefault("train_batch_size", 128 if config["dataset"] == "beauty" else 256)
    return run(config, output_dir=output_dir)


if __name__ == "__main__":
    main()
