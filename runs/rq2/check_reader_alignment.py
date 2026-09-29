import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from runs.retrieval.datasets import DATASETS
from runs.retrieval.utils import PROJECT_ROOT, build_static_model, build_tables, resolve_config
from src.analysis.reader_alignment import compare_reader_alignment
from src.embeddings import FeatureEmbedder
from src.utils import seed_everything


def main(argv=None):
    parser = argparse.ArgumentParser(description="Initial/checkpoint reader controls for static H=1 HT.")
    parser.add_argument("--dataset", choices=("ml1m", "steam", "yambda"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--budget-fraction", type=float, required=True)
    parser.add_argument("--layout", choices=("multiplex", "per_feature"), default="multiplex")
    parser.add_argument("--reference-report", type=Path, help="Match its reader pairs, including separate-table geometry.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    pairs = None
    if args.reference_report is not None:
        reference = json.loads(args.reference_report.read_text())
        assert (reference["dataset"], reference["budget_fraction"], reference["seed"], reference["num_hashes"]) == (
            args.dataset, args.budget_fraction, args.seed, 1,
        )
        pairs = [(pair["target"], pair["source"]) for pair in reference["pairs"]]
    torch.set_num_threads(4)
    config = resolve_config({
        "dataset": args.dataset, "model": "static", "approach": "hashing_trick", "loss": "bpr",
        "multiplex": args.layout == "multiplex", "budget_fraction": args.budget_fraction,
        "num_hashes": 1, "seed": args.seed, "device": args.device,
    })
    metadata_path = DATASETS[args.dataset]["static"]["data_dir"] / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    specs = {
        name: {**spec, "owner": owner}
        for owner, group in metadata["features"].items() for name, spec in group.items()
    }
    tables = build_tables(specs, config)
    seed_everything(args.seed)
    # Hash contents consume no RNG and are never looked up in this reader-only check.
    hashes = {name: torch.zeros(1, 1, dtype=torch.int64) for name in config["features"]}
    embedder = FeatureEmbedder(tables, hashes)
    dims = {
        owner: {name: dim for name, dim in embedder.output_dims.items() if specs[name]["owner"] == owner}
        for owner in ("user", "item")
    }
    model, _ = build_static_model(embedder, dims["user"], dims["item"], config)
    initial = {
        name: reader.weight.detach().clone()
        for tower in model.values() for name, reader in tower.encoder.readers.items()
    }
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    # Wrong table sizes/order would invalidate the reconstructed RNG sequence.
    for name, weight in model.state_dict().items():
        assert weight.shape == checkpoint["model"][name].shape, name
    features = {
        name: {"owner": specs[name]["owner"], "table": index}
        for index, table in enumerate(tables) for name in table["features"]
    }
    trained = {
        name: checkpoint["model"][f"{feature['owner']}.encoder.readers.{name}.weight"].to(args.device)
        for name, feature in features.items()
    }
    report = {
        "dataset": args.dataset, "layout": args.layout, "budget_fraction": args.budget_fraction,
        "training_loss": "bpr", "num_hashes": 1, "seed": args.seed,
        "checkpoint": str(args.checkpoint.resolve()),
        "epoch": checkpoint.get("epoch"), "global_step": checkpoint.get("global_step"),
        "initialization": "seeded reconstruction with current builders; no saved initial checkpoint",
        "torch_version": str(torch.__version__), "dtype": "float32", "tables": tables,
        "control": "all source output basis directions, source bank normalized to unit Frobenius norm",
        "reference_report": str(args.reference_report.resolve()) if args.reference_report is not None else None,
        "pair_selection": "reference pairs; separate tables are coordinate-only controls" if pairs is not None else "shared tables only",
        "coupling": "sqrt(d) * ||A B^T||_F / (||A||_F * ||B||_F)",
        **compare_reader_alignment(initial, trained, features, pairs=pairs),
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = args.output_dir or PROJECT_ROOT / "experiment_logs/theory/reader_alignment" / args.dataset / stamp
    output.mkdir(parents=True, exist_ok=False)
    torch.save({
        "initial": {name: weight.cpu() for name, weight in initial.items()},
        "trained": {name: weight.cpu() for name, weight in trained.items()},
    }, output / "readers.pt")
    (output / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Artifacts: {output.resolve()}", flush=True)
    for pair in report["pairs"]:
        if pair["target"] < pair["source"]:
            print(
                f"{pair['target']} / {pair['source']}: "
                f"{pair['coupling_initial']:.3f} -> {pair['coupling_trained']:.3f}",
                flush=True,
            )
    return report


if __name__ == "__main__":
    main()
