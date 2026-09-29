import torch

from src.data.item_features import gather_item_features
from src.losses.retrieval import bpr_loss, full_ce_loss, sampled_ce_loss
from src.metrics.retrieval import compute_retrieval_metrics


def compute_retrieval_batch(
    model,
    batch,
    *,
    catalog_features,
    num_items,
    regularisation,
    embedding_l2,
    objective="bpr",
    num_sampled_negatives=256,
    ks=(10, 100, 1000),
):
    """Return BPR/sampled CE/full CE loss and detached logging scalars.

    BPR draws one negative per target; sampled CE shares a pool across queries.
    Both sample uniformly with replacement, without filtering. Zero L2 skips R.
    Train Recall/NDCG reuse training queries and re-encode the catalog in eval mode.
    """
    query_tower = model["user"]
    output = query_tower(batch)
    if embedding_l2 == 0:
        output = {"embedding": output["embedding"]}
    pairs = query_tower.prepare_training_pairs(output, batch)
    # Keep target/interaction IDs out of feature regularisation.
    query_output = {name: pairs[name] for name in output}
    positive_item_idx = pairs["target_item_idx"]
    query_embedding = query_output["embedding"]
    item_tower = model["item"]

    if objective == "full_ce":
        item_idx = torch.arange(num_items, device=positive_item_idx.device)
        catalog_output = item_tower(gather_item_features(item_idx, catalog_features))
        logits = query_embedding @ catalog_output["embedding"].T
        downstream_loss = full_ce_loss(logits, positive_item_idx)
    else:
        negative_shape = (
            positive_item_idx.shape if objective == "bpr" else (num_sampled_negatives,)
        )
        negative_item_idx = torch.randint(
            num_items, negative_shape, device=positive_item_idx.device,
        )
        positive_output = item_tower(
            gather_item_features(positive_item_idx, catalog_features)
        )
        negative_output = item_tower(
            gather_item_features(negative_item_idx, catalog_features)
        )
        positive_scores = (query_embedding * positive_output["embedding"]).sum(dim=-1)

        if objective == "bpr":
            negative_scores = (query_embedding * negative_output["embedding"]).sum(dim=-1)
            downstream_loss = bpr_loss(positive_scores, negative_scores)
        elif objective == "sampled_ce":
            negative_scores = query_embedding @ negative_output["embedding"].T
            logits = torch.cat([positive_scores[:, None], negative_scores], dim=1)
            downstream_loss = sampled_ce_loss(logits)

    regularisation_loss = downstream_loss.new_zeros(())
    loss = downstream_loss
    if embedding_l2 != 0:
        if objective == "full_ce":
            catalog_penalties = regularisation.per_example(catalog_output)
            regularisation_loss = (
                regularisation(query_output)
                + catalog_penalties[positive_item_idx].mean()
                + catalog_penalties.mean()
            )
        else:
            regularisation_loss = (
                regularisation(query_output)
                + regularisation(positive_output)
                + regularisation(negative_output)
            )
        loss = downstream_loss + embedding_l2 * regularisation_loss

    result = {
        "loss": loss,
        "downstream_loss": downstream_loss.detach(),
        "regularisation_loss": regularisation_loss.detach(),
    }
    if objective == "bpr":
        result["pairwise_accuracy"] = (positive_scores > negative_scores).float().mean().detach()

    previous_modes = [(module, module.training) for module in item_tower.modules()]
    try:
        item_tower.eval()
        with torch.inference_mode():
            item_idx = torch.arange(num_items, device=positive_item_idx.device)
            catalog_embeddings = item_tower(
                gather_item_features(item_idx, catalog_features)
            )["embedding"]

            metric_sums = compute_retrieval_metrics(
                query_embedding, catalog_embeddings, positive_item_idx,
                ks=ks,
            )
            result.update({
                name: value / positive_item_idx.numel()
                for name, value in metric_sums.items()
            })
    finally:
        # Restore individual flags, including mixed modes and shared modules.
        for module, was_training in previous_modes:
            module.training = was_training

    return result
