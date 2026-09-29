import torch
from torch.nn import functional as F


def bpr_loss(positive_scores, negative_scores):
    """Mean pairwise logistic loss for paired positive/negative scores [B]."""
    return F.softplus(negative_scores - positive_scores).mean()


def sampled_ce_loss(logits):
    """Mean CE over logits [B, 1 + K], with the positive in column zero."""
    targets = torch.zeros(logits.shape[0], dtype=torch.int64, device=logits.device)
    return F.cross_entropy(logits, targets)


def full_ce_loss(logits, target_item_idx):
    """Mean CE over logits [B, num_items], with columns ordered by item_idx."""
    return F.cross_entropy(logits, target_item_idx)
