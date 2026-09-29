import math

import torch
from torch.nn import functional as F

from src.data.item_features import gather_item_features


@torch.no_grad()
def set_reader_geometry(model, specs, shared_features, rank, geometry):
    """Freeze equal-Frobenius-norm readers with controlled, densely rotated row spaces."""
    readers = {
        name: model[specs[name]["owner"]].encoder.readers[name]
        for name in model["user"].feature_encoder.output_dims
    }
    first = next(iter(readers.values())).weight
    k, d = first.shape
    basis, _ = torch.linalg.qr(torch.randn(d, d, device=first.device))
    for name, reader in readers.items():
        output_basis, _ = torch.linalg.qr(torch.randn(k, k, device=first.device))
        index = shared_features.index(name) if name in shared_features else 0
        start = index * rank if geometry == "packed" else 0
        columns = (torch.arange(rank, device=first.device) + start) % d
        reader.weight.copy_(
            math.sqrt(k / rank) * output_basis[:, :rank] @ basis[:, columns].T
        )
        reader.requires_grad_(False)


@torch.no_grad()
def geometry_report(model, specs, shared_features):
    readers, bases = {}, {}
    for name in shared_features:
        weight = model[specs[name]["owner"]].encoder.readers[name].weight
        _, singular, vh = torch.linalg.svd(weight, full_matrices=False)
        tolerance = max(weight.shape) * torch.finfo(weight.dtype).eps * singular.max()
        rank = int((singular > tolerance).sum().item())
        basis, _ = torch.linalg.qr(vh[:rank].T)
        bases[name] = basis.T
        readers[name] = {
            "rank": rank, "singular_values": singular.tolist(),
            "frobenius_norm": weight.norm().item(), "rank_tolerance": tolerance.item(),
        }
    pairs = {}
    for index, name in enumerate(shared_features):
        for other in shared_features[index + 1:]:
            pairs[f"{name}/{other}"] = (bases[name] @ bases[other].T).square().sum().item()
    rank_sum = sum(reader["rank"] for reader in readers.values())
    width = next(iter(bases.values())).shape[1]
    lower_bound = max(0.0, 0.5 * (rank_sum**2 / width - rank_sum))
    overlap = sum(pairs.values())
    return {
        "readers": readers, "pair_overlap": pairs, "rank_sum": rank_sum,
        "embedding_dim": width, "overlap": overlap, "lower_bound": lower_bound,
        "bound_satisfied": overlap + 1e-4 * max(1.0, lower_bound) >= lower_bound,
        "orthogonal": overlap < 1e-6,
    }


def bpr_forward(model, queries, items):
    """items contains [positive, one uniform negative] for each interaction."""
    query = model["user"](queries)
    item = model["item"](items)
    margins = (query["embedding"] * (item["embedding"][:, 1] - item["embedding"][:, 0])).sum(-1)
    return F.softplus(margins).mean(), query, item, margins


def lookup_inputs(indices, negatives, interactions, user_features, catalog_features):
    users = interactions["user_idx"][indices]
    queries = {"user_idx": users}
    queries.update({name: values[users] for name, values in user_features.items()})
    targets = interactions["item_idx"][indices]
    items = gather_item_features(torch.stack((targets, negatives), dim=1), catalog_features)
    return queries, items


def namespace_diagnostics(model, queries, items, specs, shared_features, learning_rate, check_gradient=False):
    """Actual BPR namespace writes, measured in every shared reader's output space."""
    encoder = model["user"].feature_encoder
    shared = next(embedder for embedder in encoder.embedders if len(embedder.features) > 1)
    loss, query, item, margins = bpr_forward(model, queries, items)
    reference = None
    if check_gradient:
        reference = torch.autograd.grad(loss, shared.table.weight)[0].coalesce().to_dense()
    with torch.no_grad():
        rho = torch.sigmoid(margins) / len(margins)
        query_gradient = rho[:, None] * (item["embedding"][:, 1] - item["embedding"][:, 0])
        item_gradient = torch.stack((-rho, rho), dim=1)[..., None] * query["embedding"][:, None]
        parts = {}
        for feature_index, name in enumerate(shared.features):
            owner = specs[name]["owner"]
            reader = model[owner].encoder.readers[name].weight
            inputs, gradient = (queries, query_gradient) if owner == "user" else (items, item_gradient)
            rows = getattr(shared, f"hashes_{feature_index}")[inputs[name], 0] % shared.table.num_embeddings
            contribution = torch.zeros_like(shared.table.weight)
            contribution.index_add_(0, rows.flatten(), (gradient @ reader).reshape(-1, reader.shape[1]))
            parts[name] = contribution
        total = sum(parts.values())
        result = {"probe_loss": loss.item(), "gradient_l2": total.norm().item(), "features": {}}
        if reference is not None:
            error = (total - reference).norm() / reference.norm().clamp_min(1e-30)
            result["autograd_relative_error"] = error.item()
        for name in shared_features:
            owner = specs[name]["owner"]
            reader = model[owner].encoder.readers[name].weight
            own = parts[name] @ reader.T
            foreign = total - parts[name]
            inter = sum((part for other, part in parts.items() if other != name and specs[other]["owner"] == owner), torch.zeros_like(total))
            cross = foreign - inter
            result["features"][name] = {
                "own_readout_update_l2": learning_rate * own.norm().item(),
                "foreign_readout_update_l2": learning_rate * (foreign @ reader.T).norm().item(),
                "inter_readout_update_l2": learning_rate * (inter @ reader.T).norm().item(),
                "cross_readout_update_l2": learning_rate * (cross @ reader.T).norm().item(),
                "foreign_gain": ((foreign @ reader.T).norm() / foreign.norm().clamp_min(1e-30)).item(),
                "foreign_to_own": ((foreign @ reader.T).norm() / own.norm().clamp_min(1e-30)).item(),
            }
        return result


@torch.no_grad()
def paired_probe(models, queries, items, specs):
    outputs = {name: bpr_forward(model, queries, items) for name, model in models.items()}
    global_output, separate_output = outputs["multiplex"], outputs["extended_per_feature"]
    gap_sq, reference_sq = 0.0, 0.0
    for name in models["multiplex"]["user"].feature_encoder.output_dims:
        owner = specs[name]["owner"]
        index = 1 if owner == "user" else 2
        reader = models["multiplex"][owner].encoder.readers[name]
        a = reader(global_output[index][name])
        b = reader(separate_output[index][name])
        gap_sq = gap_sq + (a - b).square().sum()
        reference_sq = reference_sq + b.square().sum()
    return {
        "multiplex_loss": global_output[0].item(),
        "extended_per_feature_loss": separate_output[0].item(),
        "readout_gap_relative": (gap_sq / reference_sq.clamp_min(1e-30)).sqrt().item(),
        "margin_gap_rms": (global_output[3] - separate_output[3]).square().mean().sqrt().item(),
        "margin_gap_max": (global_output[3] - separate_output[3]).abs().max().item(),
    }
