import torch
from torch.utils.data import default_collate


def sequential_train_collate(samples):
    batch = default_collate(samples)
    positions = torch.arange(
        batch["input_item_idx"].shape[1], dtype=torch.int64, device="cpu"
    )
    valid = positions[None, :] < batch["history_length"][:, None]
    batch["valid_positions"] = valid.flatten().nonzero(as_tuple=True)[0]
    return batch
