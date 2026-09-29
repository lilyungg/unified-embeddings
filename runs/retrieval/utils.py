import json
import math
from datetime import datetime, timezone
from fractions import Fraction
from functools import partial
from pathlib import Path

import numpy as np
import torch

from runs.retrieval.datasets import DATASETS
from src.data.collate import sequential_train_collate
from src.data.dataloader import create_dataloader
from src.embeddings import EMBEDDING_METHODS, FeatureEmbedder
from src.losses.regularisation import FeatureRegularisationLoss
from src.models import LinearFeatureEncoder, Tower
from src.models.sequential_tower import SequentialTower
from src.training import (
    ConsoleLoggerCallback,
    MetricsLoggerCallback,
    ModelCheckpointCallback,
    Trainer,
)
from src.training.evaluation import encode_sequential_query, evaluate_retrieval
from src.training.retrieval import compute_retrieval_batch
from src.utils import seed_everything


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = {
    "dataset": "ml1m",
    "model": "static",
    "approach": "collisionless",
    "loss": "bpr",
    "multiplex": False,
    "seed": 42,
    "device": "cuda",
    "embedding_dim": 32,
    "output_dim": 32,
    "embedding_lr": 3e-4,
    "reader_lr": 3e-4,
    "embedding_l2": 0.0,
    "num_sampled_negatives": 256,
    "num_steps": 100_000,
    "train_batch_size": 1024,
    "eval_batch_size": 128,
    "num_workers": 4,
    "num_threads": 4,
    "patience": 5,
    "log_every_n_steps": 100,
    "ks": (10, 100, 1000),
}

MODEL_DEFAULTS = {
    "static": {},
    "sequential": {
        "train_batch_size": 256,
        "log_every_n_steps": 20,
        "max_len": 200,
        "num_layers": 2,
        "num_heads": 2,
        "dim_feedforward": 128,
        "dropout": 0.1,
    },
}

HASHING_DEFAULTS = {
    "budget_fraction": 1.0,
    "num_hashes": 1,
    "hash_threshold": 5,
    "min_table_rows": 5,
}

APPROACH_DEFAULTS = {
    "collisionless": {},
    "hashing_trick": {**HASHING_DEFAULTS},
    "hash_embedding": {**HASHING_DEFAULTS, "num_hashes": 2},
    "pq": {**HASHING_DEFAULTS, "num_hashes": 2},
    "qr": {**HASHING_DEFAULTS, "num_hashes": 2},
    "hashed_net": {**HASHING_DEFAULTS},
    "robe_z": {**HASHING_DEFAULTS, "block_size": 8},
}


def resolve_config(overrides):
    """Apply explicit overrides last, preserving feature order."""
    selection = {**DEFAULT_CONFIG, **overrides}
    dataset = DATASETS[selection["dataset"]][selection["model"]]
    return {
        **DEFAULT_CONFIG,
        **MODEL_DEFAULTS[selection["model"]],
        **APPROACH_DEFAULTS[selection["approach"]],
        "features": dataset["features"],
        **overrides,
    }


def build_collisionless_tables(specs, config):
    return [
        {
            "method": "collisionless",
            "num_rows": specs[name]["cardinality"],
            "embedding_dim": config["embedding_dim"],
            "features": [name],
        }
        for name in config["features"]
    ]


def embedding_budget(specs, config):
    cardinality = sum(specs[name]["cardinality"] for name in config["features"])
    # Keep the existing whole-row budgets; decimal fractions avoid float ceil artifacts.
    rows = math.ceil(cardinality * Fraction(str(config["budget_fraction"])))
    return rows * config["embedding_dim"]


def _allocate_rows(cardinalities, num_rows, minimum):
    """Proportional rows with a floor, then largest remainder; ties use feature names."""
    pending = dict(cardinalities)
    rows = {}
    remaining = num_rows
    while pending:
        total = sum(pending.values())
        small = [
            name for name, cardinality in pending.items()
            if remaining * cardinality < minimum * total
        ]
        if not small:
            break
        for name in small:
            rows[name] = minimum
            remaining -= minimum
            pending.pop(name)

    if pending:
        numerators = {name: remaining * cardinality for name, cardinality in pending.items()}
        rows.update({name: numerator // total for name, numerator in numerators.items()})
        residual = remaining - sum(rows[name] for name in pending)
        order = sorted(pending, key=lambda name: (-(numerators[name] % total), name))
        for name in order[:residual]:
            rows[name] += 1
    return rows


def _qr_table_rows(cardinality, num_rows, num_components):
    """Minimize squared row sizes; ties use tuple order. Return None if infeasible."""
    # Reachable mixed-radix tables use at most N + K - 1 rows in total.
    if num_rows < num_components or num_rows > cardinality + num_components - 1:
        return None
    if num_components == 1:
        return (num_rows,) if num_rows == cardinality else None

    # Balanced sizes maximize capacity for a fixed row sum, even before reachability.
    rows, extra = divmod(num_rows, num_components)
    if rows ** (num_components - extra) * (rows + 1) ** extra < cardinality:
        return None

    best = None
    for first in range(1, min(cardinality, num_rows - num_components + 1) + 1):
        # Quotients form another dense vocabulary; first <= cardinality reaches every row.
        remaining = (cardinality + first - 1) // first
        tail = _qr_table_rows(remaining, num_rows - first, num_components - 1)
        if tail is None:
            continue
        candidate = (first, *tail)
        key = (sum(size ** 2 for size in candidate), candidate)
        if best is None or key < best:
            best = key
    return None if best is None else best[1]


def build_hashing_tables(specs, config, *, method, multiplex):
    """Match compressed main lookup budgets; HE importance weights are additional."""
    if method == "robe_z":
        feature_ids = {name: i for i, name in enumerate(sorted(specs))}

    cardinalities = {
        name: specs[name]["cardinality"]
        for name in config["features"]
    }
    threshold = config["hash_threshold"]
    exact = {
        name: cardinality for name, cardinality in cardinalities.items()
        if threshold is not None and cardinality <= threshold
    }
    hashed = {
        name: cardinality for name, cardinality in cardinalities.items()
        if name not in exact
    }
    dim = config["embedding_dim"]
    num_hashes = config["num_hashes"]
    minimum = 0 if method == "qr" else config["min_table_rows"]
    if method == "pq":
        # PQ splits each group's rows across K component tables.
        minimum *= num_hashes
    remaining = embedding_budget(specs, config) - sum(exact.values()) * dim
    minimum_rows = len(hashed) * minimum
    # Both layouts must fit the agreed per-feature minimum without increasing the budget.
    if remaining < minimum_rows * dim:
        raise ValueError("Embedding budget cannot fit exact features and min_table_rows.")

    tables = [
        {
            "method": "collisionless",
            "num_rows": cardinality,
            "embedding_dim": dim,
            "features": [name],
        }
        for name, cardinality in exact.items()
    ]
    if not hashed:
        if method == "qr" and remaining != 0:
            raise ValueError("QR budget must equal the protected tables when no QR features remain.")
        return tables

    num_rows = remaining // dim
    if multiplex:
        groups = [(list(hashed), num_rows)]
    else:
        allocations = _allocate_rows(hashed, num_rows, minimum)
        groups = [([name], allocations[name]) for name in hashed]

    for features, rows in groups:
        table = {"method": method, "embedding_dim": dim}
        if method in ("hashing_trick", "hashed_net", "robe_z"):
            table["num_rows"] = rows
            table["features"] = {name: num_hashes for name in features}
            if method == "robe_z":
                table.update(feature_ids=feature_ids, block_size=config["block_size"])
        elif method == "pq":
            table.update(num_rows=rows, features=features, num_hashes=num_hashes)
        elif method == "qr":
            group = {name: cardinalities[name] for name in features}
            cardinality = sum(group.values())
            table_rows = _qr_table_rows(cardinality, rows, num_hashes)
            if table_rows is None:
                raise ValueError(
                    f"QR group {features}: no reachable allocation for "
                    f"cardinality={cardinality}, rows={rows}, components={num_hashes}."
                )
            table.update(features=group, table_rows=list(table_rows))
        else:
            table.update(
                component_rows=rows, importance_rows=rows,
                features=features, num_hashes=num_hashes,
            )
        tables.append(table)

    if not multiplex:
        # Preserve configured feature order, including the RNG order during initialization.
        order = {name: index for index, name in enumerate(cardinalities)}
        tables.sort(key=lambda table: order[next(iter(table["features"]))])
    return tables


def build_tables(specs, config):
    if config["approach"] == "collisionless":
        return build_collisionless_tables(specs, config)
    return build_hashing_tables(
        specs, config, method=config["approach"], multiplex=config["multiplex"],
    )


def build_static_model(feature_encoder, user_dims, item_dims, config):
    model = torch.nn.ModuleDict({
        "user": Tower(feature_encoder, LinearFeatureEncoder(user_dims, config["output_dim"])),
        "item": Tower(feature_encoder, LinearFeatureEncoder(item_dims, config["output_dim"])),
    }).to(config["device"])
    reader_parameters = [
        parameter for tower in model.values() for parameter in tower.encoder.parameters()
    ]
    return model, reader_parameters


def build_sequential_model(feature_encoder, user_dims, item_dims, catalog_features, config):
    item_tower = Tower(feature_encoder, LinearFeatureEncoder(item_dims, config["output_dim"]))
    user_tower = None
    if user_dims:
        user_tower = Tower(feature_encoder, LinearFeatureEncoder(user_dims, config["output_dim"]))
    query_tower = SequentialTower(
        item_tower, catalog_features,
        d_model=config["output_dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        dim_feedforward=config["dim_feedforward"],
        dropout=config["dropout"],
        max_len=config["max_len"],
        user_tower=user_tower,
    )
    model = torch.nn.ModuleDict({"user": query_tower, "item": item_tower}).to(config["device"])
    reader_parameters = [*item_tower.encoder.parameters(), *query_tower.encoder.parameters()]
    if user_tower is not None:
        reader_parameters.extend(user_tower.encoder.parameters())
    return model, reader_parameters


def run(config, output_dir=None, *, prepare_embeddings=None, callbacks=()):
    config = resolve_config(config)
    device = config["device"]
    seed_everything(config["seed"])
    torch.set_num_threads(config["num_threads"])

    sequential = config["model"] == "sequential"
    dataset_settings = DATASETS[config["dataset"]][config["model"]]
    data = dataset_settings["load_data"](dataset_settings["data_dir"])
    dataset_kwargs = {"max_len": config["max_len"]} if sequential else {}
    datasets = {
        split: dataset_settings["dataset_class"](data, split=split, **dataset_kwargs)
        for split in ("train", "validation", "test")
    }
    loaders = {}
    for split, dataset in datasets.items():
        is_train = split == "train"
        loaders[split] = create_dataloader(
            dataset,
            batch_size=config["train_batch_size"] if is_train else config["eval_batch_size"],
            device=device,
            num_workers=config["num_workers"],
            collate_fn=sequential_train_collate if sequential and is_train else None,
            shuffle=is_train,
            drop_last=is_train,
        )

    specs = datasets["train"].feature_specs
    train_end = data["metadata"]["split"]["train_end"]
    train_seen = {}
    for name in ("user_idx", "item_idx"):
        seen = np.zeros(specs[name]["cardinality"], dtype=bool)
        seen[data["interactions"][name][:train_end]] = True
        train_seen[name] = torch.as_tensor(seen, device=device)

    tables = build_tables(specs, config)
    hashed_features = [
        name for table in tables
        if EMBEDDING_METHODS[table["method"]].requires_hashes
        for name in table["features"]
    ]
    feature_hashes = datasets["train"].load_feature_hashes(hashed_features)
    feature_encoder = FeatureEmbedder(tables, feature_hashes)
    user_dims = {
        name: dim for name, dim in feature_encoder.output_dims.items()
        if specs[name]["owner"] == "user"
    }
    item_dims = {
        name: dim for name, dim in feature_encoder.output_dims.items()
        if specs[name]["owner"] == "item"
    }
    catalog_features = {
        name: torch.as_tensor(values, dtype=torch.int64, device=device)
        for name, values in data["item_features"].items() if name in item_dims
    }
    if sequential:
        model, reader_parameters = build_sequential_model(
            feature_encoder, user_dims, item_dims, catalog_features, config,
        )
        encode_query = encode_sequential_query
    else:
        model, reader_parameters = build_static_model(feature_encoder, user_dims, item_dims, config)
        encode_query = lambda model, batch: model["user"](batch)

    if prepare_embeddings is not None:
        tables = prepare_embeddings(feature_encoder, tables, feature_hashes)

    # Shared feature/history/candidate parameters enter each optimizer only once.
    embedding_parameters = list(feature_encoder.parameters())
    embedding_count = sum(p.numel() for p in embedding_parameters)
    importance_count = sum(
        table["importance_rows"] * table["num_hashes"]
        for table in tables if table["method"] == "hash_embedding"
    )
    main_count = embedding_count - importance_count
    print(
        f"Parameters: main_lookup={main_count}, importance={importance_count}, "
        f"readers={sum(p.numel() for p in reader_parameters)}",
        flush=True,
    )
    if config["approach"] != "collisionless":
        budget = embedding_budget(specs, config)
        if any(table["method"] == "qr" for table in tables) and main_count != budget:
            raise ValueError(f"QR main lookup parameters {main_count} do not match budget {budget}.")
        baseline = sum(specs[name]["cardinality"] for name in config["features"]) * config["embedding_dim"]
        budget_label = "reference" if prepare_embeddings is not None else "target"
        unused = "" if prepare_embeddings is not None else f", unused={budget - main_count}"
        print(
            f"Main lookup budget: {budget_label}={budget}{unused}, "
            f"fraction={config['budget_fraction']}, actual_fraction={main_count / baseline:.8f}",
            flush=True,
        )
    optimizers = {
        "embeddings": torch.optim.SparseAdam(embedding_parameters, lr=config["embedding_lr"]),
        "readers": torch.optim.AdamW(reader_parameters, lr=config["reader_lr"]),
    }
    regularisation = None
    if config["embedding_l2"] != 0:
        feature_weights = {
            name: 1.0 / table["features"][name] if table["method"] == "hashing_trick" else 1.0
            for table in tables
            for name in table["features"]
        }
        regularisation = FeatureRegularisationLoss(feature_weights)
    metric_settings = {
        "catalog_features": catalog_features,
        "num_items": specs["item_idx"]["cardinality"],
        "ks": config["ks"],
    }
    compute_batch = partial(
        compute_retrieval_batch,
        objective=config["loss"],
        num_sampled_negatives=config["num_sampled_negatives"],
        regularisation=regularisation,
        embedding_l2=config["embedding_l2"],
        **metric_settings,
    )
    evaluate = partial(
        evaluate_retrieval, encode_query=encode_query,
        seen_users=train_seen["user_idx"], seen_items=train_seen["item_idx"],
        **metric_settings,
    )

    method = ("multiplex_" if config["multiplex"] else "") + config["approach"]
    output_root = (
        PROJECT_ROOT / "experiment_logs" / config["dataset"] / "retrieval"
        / config["model"] / f"{method}_{config['loss']}"
    )
    return run_training(
        config, model, optimizers, loaders, compute_batch, evaluate,
        output_root, output_dir, callbacks=callbacks,
    )


def run_training(
    config, model, optimizers, loaders, compute_batch, evaluate,
    output_root, output_dir=None, *, callbacks=(),
):
    if output_dir is None:
        if "budget_fraction" in config:
            settings = f"budget_{config['budget_fraction']}_h_{config['num_hashes']}"
            output_root = Path(output_root) / settings
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        output_dir = Path(output_root) / f"seed_{config['seed']}_{timestamp}"
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    print(f"Artifacts: {output_dir}", flush=True)

    with (output_dir / "metrics.jsonl").open("w", encoding="utf-8") as metrics_file:
        def log_metrics(metrics, *, step):
            metrics_file.write(json.dumps({"step": step, **metrics}, allow_nan=False) + "\n")
            metrics_file.flush()

        trainer = Trainer(
            model, optimizers, compute_batch, evaluate,
            loaders["train"], loaders["validation"], config["num_steps"],
            monitor="recall@1000",
            log_every_n_steps=config["log_every_n_steps"],
            patience=config["patience"],
            test_dataloader=loaders["test"],
            callbacks=[
                ConsoleLoggerCallback(),
                MetricsLoggerCallback(log_metrics),
                *callbacks,
                ModelCheckpointCallback(output_dir / "best.pt", restore_best=True),
            ],
        )
        result = trainer.train()

    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    return result
