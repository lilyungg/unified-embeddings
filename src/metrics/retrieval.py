import torch


def retrieval_metric_sums(top_item_idx, target_item_idx, ks=(10, 100, 1000), *, groups=None):
    """Sum single-target Recall/NDCG in float64 over interactions.

    Rankings [B, K] or shared [K] list dense item indices by descending score;
    targets are [B]. Supply at least min(max(ks), catalog size) ranked items.
    NumPy arrays and tensors are accepted. Divide sums by the interaction count.
    Optional groups map names to boolean masks [B]; return group sums and counts.
    """
    top_item_idx = torch.as_tensor(top_item_idx)
    target_item_idx = torch.as_tensor(target_item_idx, device=top_item_idx.device)
    if top_item_idx.ndim == 1:
        top_item_idx = top_item_idx.unsqueeze(0)

    found, positions = (top_item_idx == target_item_idx[:, None]).max(dim=-1)
    discounts = (positions.to(torch.float64) + 2).log2().reciprocal()
    metrics = {}
    if groups:
        group_masks = torch.stack([
            torch.as_tensor(mask, device=target_item_idx.device)
            for mask in groups.values()
        ])
        counts = group_masks.sum(dim=1, dtype=torch.float64)
        for index, name in enumerate(groups):
            metrics[f"{name}/num_interactions"] = counts[index]

    for k in ks:
        hit = found & (positions < k)
        ndcg = hit * discounts
        metrics[f"recall@{k}"] = hit.sum(dtype=torch.float64)
        metrics[f"ndcg@{k}"] = ndcg.sum()
        if groups:
            group_recall = (group_masks & hit[None, :]).sum(dim=1, dtype=torch.float64)
            group_ndcg = (group_masks * ndcg[None, :]).sum(dim=1)
            for index, name in enumerate(groups):
                metrics[f"{name}/recall@{k}"] = group_recall[index]
                metrics[f"{name}/ndcg@{k}"] = group_ndcg[index]
    return metrics


@torch.inference_mode()
def compute_retrieval_metrics(
    query_embedding,
    catalog_embeddings,
    target_item_idx,
    *,
    ks=(10, 100, 1000),
    groups=None,
):
    """Return device float64 Recall/NDCG sums for single-target queries [Q, D].

    Catalog row i represents dense item_idx i. Rank by score only using topk;
    ties have no secondary priority. Non-finite scores produce NaN metrics.
    Optional group metrics reuse the same scores and top-K; counts stay finite.
    """
    ks = tuple(ks)
    num_items = catalog_embeddings.shape[0]

    k = min(max(ks), num_items)
    scores = query_embedding @ catalog_embeddings.T
    finite_scores = scores.isfinite().all()
    top_item_idx = scores.topk(k, dim=-1).indices
    sums = retrieval_metric_sums(top_item_idx, target_item_idx, ks, groups=groups)
    return {
        name: value if name.endswith("/num_interactions") else value.masked_fill(~finite_scores, torch.nan)
        for name, value in sums.items()
    }
