import argparse
import sys
from functools import partial
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from runs.rq2.paired_hashing import prepare_embeddings
from runs.retrieval.utils import resolve_config, run
from src.analysis.reader_dynamics import ReaderDynamicsCallback


def main(argv=None):
    parser = argparse.ArgumentParser(description="Unconstrained reader-rank dynamics under the usual Adam recipe.")
    parser.add_argument("--dataset", choices=("beauty", "steam"), required=True)
    parser.add_argument("--layout", choices=("multiplex", "extended_per_feature"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    config = resolve_config({
        "dataset": args.dataset, "model": "static", "approach": "hashing_trick", "loss": "bpr",
        "multiplex": True, "budget_fraction": 0.1, "num_hashes": 1, "seed": args.seed,
        "embedding_dim": 64, "output_dim": 32,
    })
    callback = ReaderDynamicsCallback(args.output_dir, seed=config["seed"], layout=args.layout, config=config)
    return run(
        config, args.output_dir,
        prepare_embeddings=partial(prepare_embeddings, layout=args.layout, seed=config["seed"]),
        callbacks=[callback],
    )


if __name__ == "__main__":
    main()
