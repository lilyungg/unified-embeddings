import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import Subset

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from runs.retrieval.datasets import DATASETS
from runs.retrieval.utils import PROJECT_ROOT, build_static_model, build_tables, resolve_config
from src.analysis.bpr_gradients import (
    accumulate_bpr_gradients,
    decomposition_report,
    feature_layout,
    gradient_error,
)
from src.data.dataloader import create_dataloader
from src.embeddings import EMBEDDING_METHODS, FeatureEmbedder
from src.utils import seed_everything


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check fixed-weight, full-train H=1 BPR gradients.")
    parser.add_argument("--dataset", choices=("ml1m", "steam"), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--layout", choices=("multiplex", "per_feature", "collisionless"), default="multiplex")
    parser.add_argument("--budget-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--mc-samples", type=int, nargs="*", default=[1, 4, 16, 64])
    parser.add_argument("--mc-repeats", type=int, default=3)
    parser.add_argument("--max-interactions", type=int, help="Optional prefix for a smoke check, not the full experiment.")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.num_threads)
    seed_everything(args.seed)
    config = resolve_config({
        "dataset": args.dataset,
        "model": "static",
        "approach": "collisionless" if args.layout == "collisionless" else "hashing_trick",
        "multiplex": args.layout == "multiplex",
        "budget_fraction": args.budget_fraction,
        "num_hashes": 1,
        "hash_threshold": 5,
        "embedding_dim": 32,
        "output_dim": 32,
        "device": args.device,
    })
    settings = DATASETS[args.dataset]["static"]
    data = settings["load_data"](settings["data_dir"])
    dataset = settings["dataset_class"](data, split="train")
    specs = dataset.feature_specs
    tables = build_tables(specs, config)
    names = [
        name for table in tables if EMBEDDING_METHODS[table["method"]].requires_hashes
        for name in table["features"]
    ]
    embedder = FeatureEmbedder(tables, dataset.load_feature_hashes(names))
    dims = {
        owner: {name: dim for name, dim in embedder.output_dims.items() if specs[name]["owner"] == owner}
        for owner in ("user", "item")
    }
    model, _ = build_static_model(embedder, dims["user"], dims["item"], config)
    catalog_features = {
        name: torch.as_tensor(values, dtype=torch.int64, device=args.device)
        for name, values in data["item_features"].items() if name in dims["item"]
    }
    full_train_size = len(dataset)
    if args.max_interactions is not None:
        dataset = Subset(dataset, range(args.max_interactions))
    loader = create_dataloader(
        dataset, batch_size=args.batch_size, device=args.device,
        num_workers=args.num_workers, shuffle=False, drop_last=False,
    )
    num_items = specs["item_idx"]["cardinality"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = args.output_dir or PROJECT_ROOT / "experiment_logs/theory/bpr_gradients" / args.dataset / stamp
    output.mkdir(parents=True, exist_ok=False)
    print(f"Artifacts: {output.resolve()}", flush=True)

    snapshots = ["initial", "checkpoint"] if args.checkpoint else ["initial"]
    for snapshot in snapshots:
        checkpoint_info = None
        if snapshot == "checkpoint":
            checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
            model.load_state_dict(checkpoint["model"])
            checkpoint_info = {
                "path": str(args.checkpoint.resolve()),
                "epoch": checkpoint.get("epoch"),
                "global_step": checkpoint.get("global_step"),
            }
            del checkpoint
        print(f"Snapshot: {snapshot}; {len(dataset)} interactions x {num_items} items", flush=True)
        weights, features = feature_layout(model, specs)
        exact = accumulate_bpr_gradients(model, loader, catalog_features, num_items, weights, features)
        report = {
            "dataset": args.dataset,
            "snapshot": snapshot,
            "checkpoint": checkpoint_info,
            "seed": args.seed,
            "dtype": "float32",
            "layout": args.layout,
            "budget_fraction": args.budget_fraction,
            "hash_threshold": 5,
            "num_hashes": 1,
            "batch_size": args.batch_size,
            "full_train_interactions": full_train_size,
            "num_items": num_items,
            "tables": tables,
            "exact": {key: value for key, value in exact.items() if key not in ("gradients", "value_gradients")},
            "decomposition": decomposition_report(exact["gradients"], exact["value_gradients"], features),
            "monte_carlo": [],
        }
        report["peak_with_decomposition_gib"] = torch.cuda.max_memory_allocated(args.device) / 2**30
        report_path = output / f"{snapshot}.json"
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({
            "snapshot": snapshot, "exact": report["exact"],
            "tables": report["decomposition"]["tables"],
            "peak_with_decomposition_gib": report["peak_with_decomposition_gib"],
        }), flush=True)
        if not report["decomposition"]["passed"]:
            raise RuntimeError(f"Analytical/autograd gradient mismatch: {report_path}")
        del exact["value_gradients"]

        for count in args.mc_samples:
            for repeat in range(args.mc_repeats):
                sampling_seed = args.seed + 1000 + repeat
                torch.manual_seed(sampling_seed)
                sampled = accumulate_bpr_gradients(
                    model, loader, catalog_features, num_items, weights, features,
                    num_negatives=count,
                )
                errors = []
                for expected, estimate in zip(exact["gradients"], sampled["gradients"]):
                    error = gradient_error(expected, estimate)
                    error.pop("passed")  # Monte Carlo error is measured, not tested against FP32 tolerance.
                    errors.append(error)
                row = {
                    "num_negatives": count, "repeat": repeat, "sampling_seed": sampling_seed,
                    **{key: value for key, value in sampled.items() if key not in ("gradients", "value_gradients")},
                    "tables": errors,
                }
                report["monte_carlo"].append(row)
                report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                print(json.dumps({"monte_carlo": row}), flush=True)
                del sampled
        del exact
    print("CHECK COMPLETE", flush=True)


if __name__ == "__main__":
    main()
