import json
from pathlib import Path

import torch
from torch.utils.data import default_collate

from src.data.item_features import gather_item_features
from src.training.callbacks import TrainerCallback
from .bpr_gradients import feature_layout
from .frozen_reader_training import bpr_forward
from .reader_diagnostics import readout_effect


@torch.no_grad()
def reader_spectra(features):
    """Distinguish numerical rank, spectral concentration, and physically shared spaces."""
    readers, bases, dominant = {}, {}, {}
    for name, feature in features.items():
        weight = feature["reader"]
        _, singular, vh = torch.linalg.svd(weight, full_matrices=False)
        tolerance = max(weight.shape) * torch.finfo(weight.dtype).eps * singular[0]
        rank = int((singular > tolerance).sum().item())
        energy = singular.square()
        cumulative = energy.cumsum(0) / energy.sum()
        energy_dims = {
            str(level): int(torch.searchsorted(cumulative, level).item()) + 1
            for level in (0.95, 0.99)
        }
        q, _ = torch.linalg.qr(vh[:rank].T)
        bases[name] = q
        q, _ = torch.linalg.qr(vh[:energy_dims["0.99"]].T)
        dominant[name] = q
        readers[name] = {
            "owner": feature["owner"], "table": feature["table"],
            "shape": list(weight.shape), "rank": rank, "rank_tolerance": tolerance.item(),
            "singular_values": singular.tolist(), "frobenius_norm": energy.sum().sqrt().item(),
            "stable_rank": (energy.sum() / energy[0]).item(), "energy_dimensions": energy_dims,
            "relative_threshold_ranks": {
                str(tol): int((singular > tol * singular[0]).sum().item())
                for tol in (1e-3, 1e-4, 1e-5)
            },
        }
    pairs = {}
    names = list(features)
    for index, name in enumerate(names):
        for other in names[index + 1:]:
            a, b = features[name]["reader"], features[other]["reader"]
            pairs[f"{name}/{other}"] = {
                "shared_table": features[name]["table"] == features[other]["table"],
                "overlap": (bases[name].T @ bases[other]).square().sum().item(),
                "dominant_99_overlap": (dominant[name].T @ dominant[other]).square().sum().item(),
                "normalized_coupling": (a.shape[1]**0.5 * (a @ b.T).norm() / (a.norm() * b.norm())).item(),
            }
    groups = {}
    for table in sorted({feature["table"] for feature in features.values()}):
        group = [name for name in names if features[name]["table"] == table]
        width = features[group[0]]["reader"].shape[1]
        rank_sum = sum(readers[name]["rank"] for name in group)
        overlap = sum(
            (bases[name].T @ bases[other]).square().sum().item()
            for index, name in enumerate(group) for other in group[index + 1:]
        )
        groups[str(table)] = {
            "features": group, "rank_sum": rank_sum, "embedding_dim": width,
            "overlap": overlap, "lower_bound": max(0.0, 0.5 * (rank_sum**2 / width - rank_sum)),
        }
    return {"readers": readers, "pairs": pairs, "physical_table_groups": groups}


@torch.no_grad()
def bpr_probe_contributions(model, queries, items, weights, features):
    """Aggregate one-negative mean-BPR gradients at actual table rows."""
    loss, query, item, margins = bpr_forward(model, queries, items)
    rho = torch.sigmoid(margins) / len(margins)
    query_gradient = rho[:, None] * (item["embedding"][:, 1] - item["embedding"][:, 0])
    item_gradient = torch.stack((-rho, rho), dim=1)[..., None] * query["embedding"][:, None]
    contributions, counts = {}, {}
    for name, feature in features.items():
        inputs, gradient = (queries, query_gradient) if feature["owner"] == "user" else (items, item_gradient)
        reader = feature["reader"]
        rows = feature["rows"][inputs[name]].flatten()
        part = torch.zeros_like(weights[feature["table"]])
        part.index_add_(0, rows, (gradient @ reader).reshape(-1, reader.shape[1]))
        count = part.new_zeros(len(part))
        count.index_add_(0, rows, part.new_ones(len(rows)))
        contributions[name], counts[name] = part, count
    return loss.item(), contributions, counts


@torch.no_grad()
def probe_readouts(model, queries, items, weights, features):
    """Probe gradient sensitivity, not a decomposition of nonlinear Adam updates."""
    loss, contributions, counts = bpr_probe_contributions(model, queries, items, weights, features)
    report = {}
    for name, feature in features.items():
        components = {
            "same_feature": contributions[name],
            "inter_feature": torch.zeros_like(contributions[name]),
            "cross_tower": torch.zeros_like(contributions[name]),
        }
        for other, source in features.items():
            if other == name or source["table"] != feature["table"]:
                continue
            kind = "inter_feature" if source["owner"] == feature["owner"] else "cross_tower"
            components[kind] += contributions[other]
        report[name] = {
            kind: readout_effect(gradient, feature["reader"], counts[name])
            for kind, gradient in components.items()
        }
    return {"loss": loss, "features": report}


class ReaderDynamicsCallback(TrainerCallback):
    """Save reader trajectories and fixed BPR probes without consuming training RNG."""

    def __init__(self, output_dir, *, seed, layout, config, every_steps=1000, probe_size=4096):
        self.output_dir = Path(output_dir) / "reader_dynamics"
        self.seed = seed
        self.layout = layout
        self.config = config
        self.every_steps = every_steps
        self.probe_size = probe_size
        self.last_step = 0
        self.best_step = 0

    def on_train_start(self, trainer):
        self.output_dir.mkdir(parents=True, exist_ok=False)
        dataset = trainer.train_dataloader.dataset
        data = dataset.data
        self.weights, self.features = feature_layout(trainer.model, dataset.feature_specs)
        device = self.weights[0].device
        catalog = {
            name: torch.as_tensor(values, device=device)
            for name, values in data["item_features"].items()
        }
        # Probe construction must not shift DataLoader shuffles or sampled train negatives.
        with torch.random.fork_rng(devices=[device]):
            torch.manual_seed(self.seed + 1000)
            indices = torch.randint(len(dataset), (self.probe_size,))
            batch = default_collate([dataset[index] for index in indices.tolist()])
            self.queries = {name: value.to(device) for name, value in batch.items()}
            negatives = torch.randint(
                dataset.feature_specs["item_idx"]["cardinality"], (self.probe_size,), device=device,
            )
        self.items = gather_item_features(
            torch.stack((self.queries["target_item_idx"], negatives), dim=1), catalog,
        )
        torch.save({
            "interaction_idx": batch["interaction_idx"], "negative_item_idx": negatives.cpu(),
        }, self.output_dir / "probe.pt")
        settings = {
            "layout": self.layout, "config": self.config, "every_steps": self.every_steps,
            "probe_size": self.probe_size, "probe_seed": self.seed + 1000,
            "optimizers": {
                name: {
                    "class": type(optimizer).__name__,
                    "groups": [
                        {key: value for key, value in group.items() if key != "params"}
                        for group in optimizer.param_groups
                    ],
                }
                for name, optimizer in trainer.optimizers.items()
            },
            "lookup_parameters": sum(weight.numel() for weight in self.weights),
        }
        (self.output_dir / "config.json").write_text(json.dumps(settings, indent=2) + "\n")
        self.record(trainer, "initial")

    def record(self, trainer, phase, metrics=None):
        step = self.best_step if phase == "selected" else trainer.global_step
        record = {
            "step": step, "training_step": trainer.global_step, "phase": phase,
            "geometry": reader_spectra(self.features),
            "gradient_probe": probe_readouts(
                trainer.model, self.queries, self.items, self.weights, self.features,
            ),
            "metrics": metrics,
        }
        with (self.output_dir / "trajectory.jsonl").open("a") as file:
            file.write(json.dumps(record, allow_nan=False) + "\n")
        torch.save(
            {name: feature["reader"].cpu() for name, feature in self.features.items()},
            self.output_dir / f"readers_{step:06d}_{phase}.pt",
        )
        self.last_step = trainer.global_step

    def on_metrics(self, trainer, split, metrics, step):
        if split == "train" and step - self.last_step >= self.every_steps:
            self.record(trainer, "train", metrics)
        elif split == "test":
            self.record(trainer, "selected", metrics)

    def on_validation_end(self, trainer, epoch, metrics, is_best):
        if is_best:
            self.best_step = trainer.global_step
        self.record(trainer, "validation", {**metrics, "epoch": epoch, "is_best": is_best})

    def on_train_end(self, trainer):
        # This callback precedes ModelCheckpointCallback's restore_best hook.
        self.record(trainer, "last")
        torch.save({"model": trainer.model.state_dict(), "global_step": trainer.global_step}, self.output_dir / "last.pt")
