import torch

from .reader_diagnostics import readout_effect


@torch.no_grad()
def reader_geometry(reader):
    singular_values = torch.linalg.svdvals(reader)
    threshold = max(reader.shape) * torch.finfo(reader.dtype).eps * singular_values[0]
    return {
        "rank": int((singular_values > threshold).sum().item()),
        "stable_rank": (singular_values.square().sum() / singular_values[0].square()).item(),
        "frobenius_norm": singular_values.norm().item(),
        "singular_values": singular_values.tolist(),
    }


@torch.no_grad()
def compare_reader_alignment(initial, trained, features, pairs=None):
    """Compare fixed source banks; explicit pairs may include coordinate-only controls."""
    states = {"initial": initial, "trained": trained}
    if pairs is None:
        pairs = [
            (target, source)
            for target, target_info in features.items()
            for source, source_info in features.items()
            if source != target and source_info["table"] == target_info["table"]
        ]
    comparisons = []
    for target, source in pairs:
        target_info, source_info = features[target], features[source]
        controls = {}
        for source_state, readers in states.items():
            # Each row is B^T z for a standard basis z; hold this bank fixed for both A states.
            bank = readers[source] / readers[source].norm()
            counts = bank.new_ones(len(bank))
            controls[source_state] = {
                target_state: readout_effect(bank, target_readers[target], counts)
                for target_state, target_readers in states.items()
            }
        comparisons.append({
            "target": target, "source": source,
            "kind": "inter_feature" if target_info["owner"] == source_info["owner"] else "cross_tower",
            "shared_table": source_info["table"] == target_info["table"],
            "coupling_initial": controls["initial"]["initial"]["relative_rms"],
            "coupling_trained": controls["trained"]["trained"]["relative_rms"],
            "fixed_source_controls": controls,
        })
    return {
        "readers": {
            name: {
                **feature,
                **{state: reader_geometry(readers[name]) for state, readers in states.items()},
            }
            for name, feature in features.items()
        },
        "pairs": comparisons,
    }
