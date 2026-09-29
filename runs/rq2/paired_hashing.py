import argparse
import json
import sys
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import torch

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from runs.retrieval import utils
from src.embeddings import FeatureEmbedder
from src.utils import seed_everything


CONFIG = {
    "dataset": "beauty",
    "model": "static",
    "approach": "hashing_trick",
    "loss": "bpr",
    "multiplex": True,
    "seed": 42,
    "embedding_dim": 32,
    "output_dim": 32,
    "embedding_lr": 3e-4,
    "reader_lr": 3e-4,
    "embedding_l2": 0.0,
    "train_batch_size": 1024,
    "eval_batch_size": 128,
    "num_workers": 4,
    "num_threads": 4,
    "patience": 20,
    "num_steps": 100_000,
    "log_every_n_steps": 100,
    "hash_threshold": 5,
    "min_table_rows": 5,
}


def build_split_tables(tables, feature_groups):
    expanded = []
    for table in tables:
        if table["method"] == "hashing_trick":
            for group in feature_groups:
                features = {
                    name: num_hashes for name, num_hashes in table["features"].items()
                    if name in group
                }
                if features:
                    expanded.append({**table, "features": features})
        else:
            expanded.append(table)
    return expanded


def prepare_embeddings(feature_encoder, tables, feature_hashes, *, layout, seed, tower_features=None):
    """Split canonical multiplex weights into independent copies; keep its readers."""
    if layout != "multiplex":
        feature_groups = (
            tower_features
            if layout == "tower_local"
            else [(name,) for name in feature_encoder.output_dims]
        )
        sources = {
            name: embedder
            for table, embedder in zip(tables, feature_encoder.embedders)
            for name in table["features"]
        }
        tables = build_split_tables(tables, feature_groups)
        expanded = FeatureEmbedder(tables, feature_hashes).to(next(feature_encoder.parameters()).device)
        with torch.no_grad():
            for table, embedder in zip(tables, expanded.embedders):
                source = sources[next(iter(table["features"]))]
                embedder.table.weight.copy_(source.table.weight)
        # Both towers already reference this encoder; readers and output dimensions stay unchanged.
        feature_encoder.embedders = expanded.embedders

    # Model construction differs across layouts. Training randomness must not depend on it.
    seed_everything(seed)
    print(f"Paired layout: {layout}; training RNG reset to {seed}", flush=True)
    return tables


def run_variant(layout, overrides, output_dir=None):
    config = utils.resolve_config({**CONFIG, **overrides})
    data_dir = utils.DATASETS[config["dataset"]]["static"]["data_dir"]
    metadata = json.loads((data_dir / "metadata.json").read_text())
    tower_features = [
        [name for name in config["features"] if name in metadata["features"][owner]]
        for owner in ("user", "item")
    ]
    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        output_dir = (
            utils.PROJECT_ROOT / "experiment_logs" / config["dataset"]
            / "retrieval/static/paired_hashing_trick"
            / f"budget_{config['budget_fraction']}_h_{config['num_hashes']}"
            / f"{layout}_seed_{config['seed']}_{stamp}"
        )
    return utils.run(
        config, output_dir,
        prepare_embeddings=partial(
            prepare_embeddings, layout=layout, seed=config["seed"], tower_features=tower_features,
        ),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="One branch of the paired HT/BPR sharing control.")
    parser.add_argument("--dataset", choices=utils.DATASETS, default=CONFIG["dataset"])
    parser.add_argument(
        "--layout", choices=("multiplex", "tower_local", "extended_per_feature"), required=True,
    )
    parser.add_argument("--budget-fraction", type=float, choices=(1.0, 0.1, 0.04), required=True)
    parser.add_argument("--num-hashes", type=int, choices=(1, 4), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=CONFIG["patience"])
    parser.add_argument("--output-dir", type=Path)
    args = vars(parser.parse_args(argv))
    layout = args.pop("layout")
    output_dir = args.pop("output_dir")
    run_variant(layout, args, output_dir)


if __name__ == "__main__":
    main()
