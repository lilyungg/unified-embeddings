import time

import torch
from torch.nn import functional as F

from src.data.item_features import gather_item_features
from src.embeddings.hashing_trick import HashingTrickEmbedding


def feature_layout(model, specs):
    """Describe H=1 HT/exact lookups; table identity is part of each address."""
    weights = []
    features = {}
    for table_index, embedder in enumerate(model["user"].feature_encoder.embedders):
        weight = embedder.table.weight
        weights.append(weight)
        for feature_index, name in enumerate(embedder.features):
            if isinstance(embedder, HashingTrickEmbedding):
                if embedder.features[name] != 1:
                    raise ValueError("The gradient decomposition currently supports H=1 only.")
                rows = getattr(embedder, f"hashes_{feature_index}")[:, 0] % len(weight)
            else:
                rows = torch.arange(specs[name]["cardinality"], device=weight.device)
            owner = specs[name]["owner"]
            features[name] = {
                "owner": owner,
                "table": table_index,
                "rows": rows,
                "reader": model[owner].encoder.readers[name].weight.detach(),
            }
    return weights, features


def gradient_error(reference, actual, rtol=1e-3, atol=1e-10):
    """Compare whole-gradient L2 norms, not elementwise relative errors near zero."""
    difference = actual - reference
    reference_norm = torch.linalg.vector_norm(reference).item()
    error_norm = torch.linalg.vector_norm(difference).item()
    return {
        "reference_l2": reference_norm,
        "error_l2": error_norm,
        "relative_l2": error_norm / reference_norm if reference_norm else None,
        "max_abs_error": difference.abs().max().item(),
        "passed": error_norm <= atol + rtol * reference_norm,
    }


def accumulate_bpr_gradients(
    model, dataloader, catalog_features, num_items, weights, features,
    *, num_negatives=None,
):
    """Fixed-weight mean BPR gradient over all interactions; None enumerates negatives.

    Exact mode also accumulates independently derived contributions per feature value.
    Sampled mode draws K uniform negatives per interaction, without exclusion.
    """
    device = weights[0].device
    num_interactions = len(dataloader.dataset)
    exact = num_negatives is None
    negatives_per_query = num_items if exact else num_negatives
    denominator = num_interactions * negatives_per_query
    gradients = [torch.zeros_like(weight) for weight in weights]
    value_gradients = {}
    if exact:
        for name, feature in features.items():
            weight = weights[feature["table"]]
            value_gradients[name] = weight.new_zeros((len(feature["rows"]), weight.shape[1]))
    catalog = gather_item_features(torch.arange(num_items, device=device), catalog_features)
    total_loss = weights[0].new_zeros(())
    processed = 0
    model.eval()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()

    for step, batch in enumerate(dataloader, start=1):
        targets = batch["target_item_idx"]
        query = model["user"](batch)["embedding"]
        if exact:
            items = model["item"](catalog)["embedding"]
            scores = query @ items.T
            margins = scores - scores.gather(1, targets[:, None])
        else:
            negative_ids = torch.randint(num_items, (len(targets), num_negatives), device=device)
            positives = model["item"](gather_item_features(targets, catalog_features))["embedding"]
            negatives = model["item"](gather_item_features(negative_ids, catalog_features))["embedding"]
            margins = (query[:, None] * negatives).sum(-1) - (query * positives).sum(-1)[:, None]
        loss = F.softplus(margins).sum() / denominator
        batch_gradients = torch.autograd.grad(loss, weights)

        with torch.no_grad():
            total_loss += loss.detach()
            # Coalesce each batch, then scatter into fixed-size buffers: no growing COO graph.
            for accumulated, gradient in zip(gradients, batch_gradients):
                gradient = gradient.coalesce()
                accumulated.index_add_(0, gradient.indices()[0], gradient.values())

            if exact:
                rho = torch.sigmoid(margins.detach()) / denominator
                rho_sum = rho.sum(1, keepdim=True)
                query_gradient = rho @ items.detach() - rho_sum * items.detach()[targets]
                item_gradient = rho.T @ query.detach()
                item_gradient.index_add_(0, targets, -rho_sum * query.detach())
                for name, feature in features.items():
                    inputs, upstream = (
                        (batch, query_gradient) if feature["owner"] == "user"
                        else (catalog, item_gradient)
                    )
                    value_gradients[name].index_add_(
                        0, inputs[name], upstream @ feature["reader"],
                    )

        processed += len(targets)
        if step % 100 == 0 or processed == num_interactions:
            torch.cuda.synchronize(device)
            mode = "exact" if exact else f"sampled K={num_negatives}"
            print(f"{mode}: {processed}/{num_interactions}, {time.perf_counter() - started:.1f}s", flush=True)

    torch.cuda.synchronize(device)
    return {
        "gradients": gradients,
        "value_gradients": value_gradients,
        "loss": total_loss.item(),
        "num_interactions": processed,
        "num_pairs": processed * negatives_per_query,
        "elapsed_s": time.perf_counter() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
    }


@torch.no_grad()
def decomposition_report(reference, value_gradients, features):
    """Split each value's shared-row gradient into true/intra/inter/cross-tower terms."""
    by_feature = {}
    analytical = [torch.zeros_like(gradient) for gradient in reference]
    for name, feature in features.items():
        table = feature["table"]
        contribution = torch.zeros_like(reference[table])
        contribution.index_add_(0, feature["rows"], value_gradients[name])
        by_feature[name] = contribution
        analytical[table] += contribution

    table_reports = [
        {"table": index, **gradient_error(expected, actual)}
        for index, (expected, actual) in enumerate(zip(reference, analytical))
    ]
    feature_reports = {}
    for name, feature in features.items():
        table, rows = feature["table"], feature["rows"]
        inter = torch.zeros_like(reference[table])
        cross = torch.zeros_like(reference[table])
        for other_name, other in features.items():
            if other_name == name or other["table"] != table:
                continue
            destination = inter if other["owner"] == feature["owner"] else cross
            destination += by_feature[other_name]
        components = {
            "true": value_gradients[name],
            "intra": by_feature[name][rows] - value_gradients[name],
            "inter": inter[rows],
            "cross_tower": cross[rows],
        }
        reconstruction = sum(components.values())
        feature_reports[name] = {
            "owner": feature["owner"],
            "table": table,
            "num_values": len(rows),
            "component_l2": {
                component: torch.linalg.vector_norm(values).item()
                for component, values in components.items()
            },
            **gradient_error(reference[table][rows], reconstruction),
        }

    return {
        "passed": all(row["passed"] for row in [*table_reports, *feature_reports.values()]),
        "rtol_l2": 1e-3,
        "atol_l2": 1e-10,
        "tables": table_reports,
        "features": feature_reports,
    }
