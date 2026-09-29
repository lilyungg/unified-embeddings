import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from runs.retrieval.datasets import DATASETS
from runs.rq2.paired_hashing import build_split_tables
from runs.retrieval.utils import build_static_model, build_tables, resolve_config
from src.analysis.frozen_reader_training import (
    bpr_forward, geometry_report, lookup_inputs, namespace_diagnostics,
    paired_probe, set_reader_geometry,
)
from src.embeddings import FeatureEmbedder
from src.training.evaluation import evaluate_retrieval
from src.utils import seed_everything


def build_pair(specs, hashes, config, rank, geometry):
    tables = build_tables(specs, config)
    shared_features = [name for table in tables if table["method"] == "hashing_trick" for name in table["features"]]
    encoder = FeatureEmbedder(tables, hashes).cuda()
    dims = {owner: {name: dim for name, dim in encoder.output_dims.items() if specs[name]["owner"] == owner} for owner in ("user", "item")}
    model, _ = build_static_model(encoder, dims["user"], dims["item"], config)
    set_reader_geometry(model, specs, shared_features, rank, geometry)
    separate = copy.deepcopy(model)
    split_tables = build_split_tables(tables, [(name,) for name in config["features"]])
    split_encoder = FeatureEmbedder(split_tables, hashes).cuda()
    sources = {name: embedder for embedder in encoder.embedders for name in embedder.features}
    with torch.no_grad():
        for embedder in split_encoder.embedders:
            name = next(iter(embedder.features))
            embedder.table.weight.copy_(sources[name].table.weight)
    separate["user"].feature_encoder.embedders = split_encoder.embedders
    return {"multiplex": model, "extended_per_feature": separate}, tables, split_tables, shared_features


def run(args):
    config = resolve_config({
        "dataset": args.dataset, "model": "static", "approach": "hashing_trick",
        "multiplex": True, "num_hashes": 1, "budget_fraction": args.budget_fraction,
        "embedding_dim": 64, "output_dim": 32, "device": "cuda", "seed": args.seed,
    })
    torch.set_num_threads(4)
    seed_everything(args.seed)
    setup = DATASETS[args.dataset]["static"]
    data = setup["load_data"](setup["data_dir"])
    train = setup["dataset_class"](data, "train")
    validation = setup["dataset_class"](data, "validation")
    specs = train.feature_specs
    hashes = train.load_feature_hashes(config["features"])
    models, tables, split_tables, shared_features = build_pair(specs, hashes, config, args.rank, args.geometry)
    geometry = geometry_report(models["multiplex"], specs, shared_features)
    assert geometry["bound_satisfied"]
    assert all(reader["rank"] == args.rank for reader in geometry["readers"].values())
    gpu = lambda array: torch.as_tensor(array, dtype=torch.int64, device="cuda")
    interactions = {name: gpu(data["interactions"][name]) for name in ("user_idx", "item_idx")}
    user_features = {name: gpu(values) for name, values in data["user_features"].items()}
    catalog = {name: gpu(values) for name, values in data["item_features"].items()}
    num_items = specs["item_idx"]["cardinality"]
    seed_everything(args.seed + 1000)
    probe_train_ids = torch.randint(len(train), (args.probe_size,), device="cuda")
    validation_ids = gpu(np.asarray(validation.interaction_indices))
    probe_val_ids = validation_ids[torch.randperm(len(validation_ids), device="cuda")[:args.probe_size]]
    probe_train = lookup_inputs(probe_train_ids, torch.randint(num_items, probe_train_ids.shape, device="cuda"), interactions, user_features, catalog)
    probe_val = lookup_inputs(probe_val_ids, torch.randint(num_items, probe_val_ids.shape, device="cuda"), interactions, user_features, catalog)
    optimizers = {
        name: torch.optim.SGD(model["user"].feature_encoder.parameters(), lr=args.lr)
        for name, model in models.items()
    }
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    settings = {
        **vars(args), "output_dir": str(output), "config": config, "tables": tables,
        "split_tables": split_tables, "geometry": geometry, "train_interactions": len(train),
        "validation_probe_interactions": len(probe_val_ids),
        "lookup_parameters": {name: sum(p.numel() for p in model["user"].feature_encoder.parameters()) for name, model in models.items()},
    }
    (output / "config.json").write_text(json.dumps(settings, indent=2) + "\n")
    torch.save({"train": probe_train_ids.cpu(), "validation": probe_val_ids.cpu(), "train_negatives": probe_train[1]["item_idx"][:, 1].cpu(), "validation_negatives": probe_val[1]["item_idx"][:, 1].cpu()}, output / "probes.pt")
    torch.save({owner: models["multiplex"][owner].encoder.state_dict() for owner in ("user", "item")}, output / "readers.pt")
    seed_everything(args.seed + 2000)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    loss_totals = torch.zeros(2, device="cuda")
    window_start = 0
    last_record = None
    with (output / "trajectory.jsonl").open("w", buffering=1) as file:
        for step in range(args.steps + 1):
            if step % args.log_every == 0 or step == args.steps:
                record = {
                    "step": step,
                    "train_probe": paired_probe(models, *probe_train, specs),
                    "validation_probe": paired_probe(models, *probe_val, specs),
                    "interference": namespace_diagnostics(models["multiplex"], *probe_train, specs, shared_features, args.lr, check_gradient=step == 0),
                }
                if step == 0:
                    assert record["train_probe"]["margin_gap_max"] == 0.0
                    assert record["interference"]["autograd_relative_error"] < 1e-3
                else:
                    record["train_window_loss"] = (loss_totals / (step - window_start)).tolist()
                record["elapsed_s"] = time.perf_counter() - started
                file.write(json.dumps(record, allow_nan=False) + "\n")
                print(json.dumps({"step": step, "elapsed_s": record["elapsed_s"], **record["validation_probe"]}), flush=True)
                loss_totals.zero_()
                window_start = step
                last_record = record
            if step == args.steps:
                break
            # Uniform interaction sampling and negatives are shared by both branches.
            indices = torch.randint(len(train), (args.batch_size,), device="cuda")
            negatives = torch.randint(num_items, indices.shape, device="cuda")
            queries, items = lookup_inputs(indices, negatives, interactions, user_features, catalog)
            for index, (name, model) in enumerate(models.items()):
                optimizers[name].zero_grad(set_to_none=True)
                loss, _, _, _ = bpr_forward(model, queries, items)
                loss.backward()
                optimizers[name].step()
                loss_totals[index] += loss.detach()
    retrieval = {}
    if args.eval_queries:
        query_ids = probe_val_ids[:args.eval_queries]
        batches = []
        for ids in query_ids.split(128):
            queries, items = lookup_inputs(ids, torch.zeros_like(ids), interactions, user_features, catalog)
            batches.append({**queries, "target_item_idx": items["item_idx"][:, 0]})
        for name, model in models.items():
            model.eval()
            retrieval[name] = evaluate_retrieval(model, batches, encode_query=lambda model, batch: model["user"](batch), catalog_features=catalog, num_items=num_items, ks=(100, 1000))
        retrieval["num_validation_queries"] = len(query_ids)
    torch.cuda.synchronize()
    torch.save({name: model.state_dict() for name, model in models.items()}, output / "checkpoint.pt")
    result = {
        "settings": settings, "final": last_record, "validation_retrieval_probe": retrieval,
        "elapsed_s": time.perf_counter() - started,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"COMPLETE {output}: {result['elapsed_s']:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Paired SGD trajectories under frozen reader subspaces.")
    parser.add_argument("--dataset", choices=("ml1m", "beauty", "steam"), required=True)
    parser.add_argument("--rank", type=int, choices=(8, 12, 13, 15, 16, 17, 32), required=True)
    parser.add_argument("--geometry", choices=("packed", "aligned"), default="packed")
    parser.add_argument("--budget-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=10.0)
    parser.add_argument("--probe-size", type=int, default=4096)
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--eval-queries", type=int, default=1024)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
