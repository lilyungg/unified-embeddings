import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from runs.retrieval.datasets import DATASETS
from runs.retrieval.utils import PROJECT_ROOT, build_static_model, build_tables, resolve_config
from src.analysis.bpr_gradients import accumulate_bpr_gradients, decomposition_report, feature_layout
from src.analysis.reader_diagnostics import reader_diagnostics
from src.data.dataloader import create_dataloader
from src.data.item_features import gather_item_features
from src.embeddings import EMBEDDING_METHODS, FeatureEmbedder
from src.utils import seed_everything


def main(argv=None):
    parser = argparse.ArgumentParser(description="Probe fixed H=1 BPR checkpoints with exact catalog negatives.")
    parser.add_argument("--dataset", choices=("ml1m", "steam", "yambda"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--layout", choices=("multiplex", "per_feature", "collisionless"), default="multiplex")
    parser.add_argument("--budget-fraction", type=float, default=0.1)
    parser.add_argument("--probe-seed", type=int, default=2026)
    parser.add_argument("--probe-sizes", type=int, nargs="+", default=[4096, 8192, 16384])
    parser.add_argument("--receiver-indices", type=Path, help="Fixed train interaction IDs for readout weighting.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    seed_everything(args.probe_seed)
    config = resolve_config({
        "dataset": args.dataset, "model": "static", "loss": "bpr",
        "approach": "collisionless" if args.layout == "collisionless" else "hashing_trick",
        "multiplex": args.layout == "multiplex", "budget_fraction": args.budget_fraction,
        "num_hashes": 1, "device": args.device,
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
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(checkpoint["model"])
    checkpoint_info = {
        "path": str(args.checkpoint.resolve()),
        "epoch": checkpoint.get("epoch"), "global_step": checkpoint.get("global_step"),
    }
    del checkpoint
    catalog_features = {
        name: torch.as_tensor(values, dtype=torch.int64, device=args.device)
        for name, values in data["item_features"].items() if name in dims["item"]
    }
    weights, features = feature_layout(model, specs)
    sizes = sorted(args.probe_sizes)
    indices = np.random.choice(len(dataset), sizes[-1], replace=False)
    receiver_indices = (
        np.load(args.receiver_indices, allow_pickle=False)
        if args.receiver_indices is not None else indices
    )
    users = data["interactions"]["user_idx"][receiver_indices]
    targets = torch.as_tensor(data["interactions"]["item_idx"][receiver_indices], device=args.device)
    receiver_inputs = gather_item_features(targets, catalog_features)
    receiver_inputs["user_idx"] = torch.as_tensor(users, device=args.device)
    receiver_inputs.update({
        name: torch.as_tensor(values[users], device=args.device)
        for name, values in data["user_features"].items()
    })
    # A supplied population also stays fixed when the gradient-probe seed changes.
    counts = {
        name: torch.bincount(receiver_inputs[name], minlength=len(feature["rows"]))
        for name, feature in features.items()
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = args.output_dir or PROJECT_ROOT / "experiment_logs/theory/reader_diagnostics" / args.dataset / stamp
    output.mkdir(parents=True, exist_ok=False)
    np.save(output / "probe_interaction_idx.npy", indices, allow_pickle=False)
    np.save(output / "receiver_interaction_idx.npy", receiver_indices, allow_pickle=False)
    report = {
        "dataset": args.dataset, "checkpoint": checkpoint_info,
        "training_loss": "bpr", "probe_loss": "expected_uniform_bpr",
        "dtype": "float32", "probe_seed": args.probe_seed,
        "sampling": "uniform train interactions without replacement",
        "receiver_weighting": "fixed receiver population, user and positive-item occurrences",
        "receiver_indices_source": str(args.receiver_indices.resolve()) if args.receiver_indices is not None else "probe",
        "num_receiver_interactions": len(receiver_indices),
        "control": "exact isotropic expected squared readout, matched row-update norms",
        "layout": args.layout, "budget_fraction": args.budget_fraction,
        "tables": tables, "batch_size": args.batch_size,
        "full_train_interactions": len(dataset), "num_items": specs["item_idx"]["cardinality"],
        "snapshots": [],
    }
    gradient_sums = [torch.zeros_like(weight) for weight in weights]
    value_sums = {
        name: weights[feature["table"]].new_zeros((len(feature["rows"]), feature["reader"].shape[1]))
        for name, feature in features.items()
    }
    print(f"Artifacts: {output.resolve()}", flush=True)
    start, loss_sum = 0, 0.0
    started = time.perf_counter()
    for end in sizes:
        loader = create_dataloader(
            Subset(dataset, indices[start:end].tolist()), batch_size=args.batch_size,
            device=args.device, num_workers=args.num_workers, shuffle=False, drop_last=False,
        )
        block = accumulate_bpr_gradients(
            model, loader, catalog_features, report["num_items"], weights, features,
        )
        count = end - start
        for total, gradient in zip(gradient_sums, block["gradients"]):
            total.add_(gradient, alpha=count)
        for name, total in value_sums.items():
            total.add_(block["value_gradients"][name], alpha=count)
        loss_sum += count * block["loss"]
        del loader, block
        gradients = [total / end for total in gradient_sums]
        values = {name: total / end for name, total in value_sums.items()}
        check = decomposition_report(gradients, values, features)
        diagnostic = reader_diagnostics(values, weights, features, counts)
        snapshot = {
            "num_interactions": end, "loss": loss_sum / end,
            "decomposition": check, "features": diagnostic,
            "elapsed_s": time.perf_counter() - started,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(args.device) / 2**30,
        }
        report["snapshots"].append(snapshot)
        (output / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({
            "num_interactions": end, "gradient_check": check["passed"],
            "relative_rms": {
                name: {kind: effect["relative_rms"] for kind, effect in row["components"].items()}
                for name, row in diagnostic.items()
            },
        }), flush=True)
        if not check["passed"]:
            raise RuntimeError(f"Analytical/autograd gradient mismatch: {output / 'result.json'}")
        del gradients, values
        start = end
    print("CHECK COMPLETE", flush=True)


if __name__ == "__main__":
    main()
