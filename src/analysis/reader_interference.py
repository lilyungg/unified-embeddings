import torch


@torch.no_grad()
def measure_readout_interference(target_reader, source_reader, embedding, step_size):
    """Probe every source-output basis direction; readers have orthonormal rows."""
    source = source_reader.weight
    target = target_reader.weight
    directions = torch.eye(source.shape[0], dtype=source.dtype, device=source.device)
    updates = -step_size * (directions @ source)

    before = target_reader(embedding)
    after = target_reader(embedding[None, :] + updates)
    measured = after - before
    cross = target @ source.T
    predicted = -step_size * (directions @ cross.T)
    update_norm = torch.linalg.vector_norm(updates)

    return {
        "update_l2": update_norm.item(),
        "readout_change_l2": torch.linalg.vector_norm(measured).item(),
        "max_abs_readout_change": measured.abs().max().item(),
        "measured_gain": (torch.linalg.vector_norm(measured) / update_norm).item(),
        "predicted_gain": (torch.linalg.vector_norm(predicted) / update_norm).item(),
        "formula_error_per_update": (
            torch.linalg.vector_norm(measured - predicted) / update_norm
        ).item(),
        "overlap_per_rank": (cross.square().sum() / source.shape[0]).item(),
    }
