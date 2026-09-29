import argparse
import sys
from pathlib import Path


if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from runs.retrieval.datasets import DATASETS
from runs.retrieval.utils import APPROACH_DEFAULTS, MODEL_DEFAULTS, run


def parse_args(argv=None, *, model=None, loss=None):
    parser = argparse.ArgumentParser(
        description="Train a retrieval model. Defaults are defined in runs/retrieval/utils.py.",
        # Omitted options must not overwrite model- or method-specific defaults.
        argument_default=argparse.SUPPRESS,
    )
    experiment = parser.add_argument_group("experiment")
    experiment.add_argument("--dataset", choices=DATASETS)
    if model is None:
        experiment.add_argument("--model", choices=MODEL_DEFAULTS)
    experiment.add_argument("--approach", choices=APPROACH_DEFAULTS)
    if loss is None:
        experiment.add_argument("--loss", choices=("bpr", "sampled_ce", "full_ce"))
    experiment.add_argument("--multiplex", action="store_true")
    experiment.add_argument(
        "--features", nargs="+",
        help="Replace the default feature list in this order; sequential user features enable a prefix token.",
    )
    experiment.add_argument("--seed", type=int)
    experiment.add_argument("--device")
    experiment.add_argument("--output-dir", type=Path, default=None)

    embeddings = parser.add_argument_group("embeddings")
    embeddings.add_argument("--embedding-dim", type=int)
    embeddings.add_argument("--output-dim", type=int)
    embeddings.add_argument("--budget-fraction", type=float)
    embeddings.add_argument("--num-hashes", type=int)
    embeddings.add_argument("--hash-threshold", type=int)
    embeddings.add_argument("--min-table-rows", type=int)
    embeddings.add_argument("--block-size", type=int, help="ROBE-Z block size.")

    training = parser.add_argument_group("training")
    training.add_argument("--num-steps", type=int)
    training.add_argument("--train-batch-size", type=int)
    training.add_argument("--eval-batch-size", type=int)
    training.add_argument("--num-workers", type=int)
    training.add_argument("--num-threads", type=int)
    training.add_argument("--num-sampled-negatives", type=int)
    training.add_argument("--embedding-lr", type=float)
    training.add_argument("--reader-lr", type=float)
    training.add_argument("--embedding-l2", type=float)
    training.add_argument("--patience", type=int)
    training.add_argument("--log-every-n-steps", type=int)

    sequential = parser.add_argument_group("sequential")
    sequential.add_argument("--max-len", type=int)
    sequential.add_argument("--num-layers", type=int)
    sequential.add_argument("--num-heads", type=int)
    sequential.add_argument("--dim-feedforward", type=int)
    sequential.add_argument("--dropout", type=float)

    config = vars(parser.parse_args(argv))
    output_dir = config.pop("output_dir")
    if model is not None:
        config["model"] = model
    if loss is not None:
        config["loss"] = loss
    return config, output_dir


def main(argv=None):
    config, output_dir = parse_args(argv)
    return run(config, output_dir=output_dir)


if __name__ == "__main__":
    main()
