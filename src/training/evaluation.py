import math

import torch

from src.data.item_features import gather_item_features
from src.metrics.retrieval import compute_retrieval_metrics


def encode_sequential_query(model, batch):
    """Return [B, D] embeddings at the last real history position; histories are nonempty."""
    output = model["user"](batch)
    embedding = output["embedding"]
    rows = torch.arange(embedding.shape[0], device=embedding.device)
    last_positions = batch["history_length"] - 1
    return {"embedding": embedding[rows, last_positions]}


@torch.inference_mode()
def evaluate_retrieval(
    model,
    dataloader,
    *,
    encode_query,
    catalog_features,
    num_items,
    ks=(10, 100, 1000),
    seen_users=None,
    seen_items=None,
):
    """Return full-catalog Recall/NDCG means over evaluation interactions.

    Encode the entire catalog once in dense item_idx order, then score each batch.
    encode_query returns {'embedding': [B, D]} without using current targets.
    The caller sets eval mode. Accumulate float64 sums and return Python scalars.
    Optional seen_users/items are train-seen bool arrays on the batch device.
    Group means use their own counts; empty groups report only count=0.
    """
    ks = tuple(ks)

    item_idx = torch.arange(num_items, device=next(model["item"].parameters()).device)
    catalog_embeddings = model["item"](
        gather_item_features(item_idx, catalog_features)
    )["embedding"]

    totals = {}
    num_interactions = 0
    for batch in dataloader:
        targets = batch["target_item_idx"]
        query_embedding = encode_query(model, batch)["embedding"]
        groups = None
        if seen_users is not None:
            warm_user = seen_users[batch["user_idx"]]
            warm_item = seen_items[targets]
            groups = {
                "warm_user/warm_item": warm_user & warm_item,
                "warm_user/cold_item": warm_user & ~warm_item,
                "cold_user/warm_item": ~warm_user & warm_item,
                "cold_user/cold_item": ~warm_user & ~warm_item,
            }
        batch_sums = compute_retrieval_metrics(
            query_embedding, catalog_embeddings, targets,
            ks=ks, groups=groups,
        )
        for name, value in batch_sums.items():
            if name not in totals:
                totals[name] = value
            else:
                totals[name] += value
        num_interactions += targets.numel()

    metrics = {
        name: int(value.item()) for name, value in totals.items()
        if name.endswith("/num_interactions")
    }
    for name, value in totals.items():
        if name.endswith("/num_interactions"):
            continue
        group = name.rpartition("/")[0]
        count = metrics[f"{group}/num_interactions"] if group else num_interactions
        if count:
            metrics[name] = (value / count).item()
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError("Non-finite retrieval scores; metrics would be invalid")
    return metrics
