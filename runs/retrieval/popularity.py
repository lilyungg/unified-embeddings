import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if not __package__:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics.retrieval import retrieval_metric_sums


def evaluate_popularity(static_dir, *, ks=(10, 100, 1000), eval_batch_size=8192):
    """Rank by train counts; evaluate the shared nonempty-history target protocol."""
    static_dir = Path(static_dir)
    metadata = json.loads((static_dir / "metadata.json").read_text(encoding="utf-8"))
    user_idx = np.load(static_dir / "interactions/user_idx.npy", allow_pickle=False)
    item_idx = np.load(static_dir / "interactions/item_idx.npy", allow_pickle=False)
    train_end = metadata["split"]["train_end"]
    validation_end = metadata["split"]["validation_end"]
    num_items = metadata["features"]["item"]["item_idx"]["cardinality"]

    counts = np.bincount(item_idx[:train_end], minlength=num_items)
    seen_users = np.zeros(metadata["features"]["user"]["user_idx"]["cardinality"], dtype=bool)
    seen_users[user_idx[:train_end]] = True
    seen_items = counts > 0
    # Zero-count items remain in the catalog; ties have no secondary key.
    top_item_idx = np.argsort(-counts)[:max(ks)]
    _, first_events = np.unique(user_idx, return_index=True)

    results = {}
    for split, start, end in (
        ("val", train_end, validation_end),
        ("test", validation_end, len(item_idx)),
    ):
        interaction_idx = np.setdiff1d(
            np.arange(start, end, dtype=np.int64), first_events, assume_unique=True,
        )
        totals = {}
        for offset in range(0, len(interaction_idx), eval_batch_size):
            batch_idx = interaction_idx[offset:offset + eval_batch_size]
            targets = item_idx[batch_idx]
            warm_user = seen_users[user_idx[batch_idx]]
            warm_item = seen_items[targets]
            groups = {
                "warm_user/warm_item": warm_user & warm_item,
                "warm_user/cold_item": warm_user & ~warm_item,
                "cold_user/warm_item": ~warm_user & warm_item,
                "cold_user/cold_item": ~warm_user & ~warm_item,
            }
            sums = retrieval_metric_sums(top_item_idx, targets, ks=ks, groups=groups)
            for name, value in sums.items():
                totals[name] = totals.get(name, 0) + value
        metrics = {
            name: int(value.item()) for name, value in totals.items()
            if name.endswith("/num_interactions")
        }
        for name, value in totals.items():
            if name.endswith("/num_interactions"):
                continue
            group = name.rpartition("/")[0]
            count = metrics[f"{group}/num_interactions"] if group else len(interaction_idx)
            if count:
                metrics[name] = (value / count).item()
        results[split] = metrics
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate a train-only popularity baseline.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--eval-batch-size", type=int, default=8192)
    args = parser.parse_args()

    torch.set_num_threads(4)
    static_dir = PROJECT_ROOT / "data/preprocessed" / args.dataset / "retrieval/static"
    results = evaluate_popularity(static_dir, eval_batch_size=args.eval_batch_size)
    print(json.dumps(results, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
