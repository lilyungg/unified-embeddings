import math

import torch


@torch.no_grad()
def readout_effect(gradient, reader, counts):
    """Occurrence-weighted RMS sensitivity relative to isotropic updates of equal norm."""
    mass = counts.sum()
    gradient_energy = (counts * gradient.square().sum(-1)).sum() / mass
    readout_energy = (counts * (gradient @ reader.T).square().sum(-1)).sum() / mass
    # E[u u^T] = I/d for a uniform unit direction; no Monte Carlo control is needed.
    isotropic_energy = gradient_energy * reader.square().sum() / reader.shape[1]
    gradient_energy, readout_energy, isotropic_energy = (
        value.item() for value in (gradient_energy, readout_energy, isotropic_energy)
    )
    return {
        "gradient_rms": math.sqrt(gradient_energy),
        "readout_rms": math.sqrt(readout_energy),
        "isotropic_readout_rms": math.sqrt(isotropic_energy),
        "relative_rms": math.sqrt(readout_energy / isotropic_energy) if isotropic_energy else None,
    }


@torch.no_grad()
def reader_diagnostics(value_gradients, weights, features, value_counts):
    """Read out the accumulated gradient at actual shared addresses, not optimizer steps."""
    contributions = {}
    for name, feature in features.items():
        contribution = torch.zeros_like(weights[feature["table"]])
        contribution.index_add_(0, feature["rows"], value_gradients[name])
        contributions[name] = contribution

    report = {}
    for name, feature in features.items():
        reader = feature["reader"]
        singular_values = torch.linalg.svdvals(reader)
        rank_threshold = max(reader.shape) * torch.finfo(reader.dtype).eps * singular_values[0]
        template = weights[feature["table"]]
        counts = template.new_zeros(len(template))
        counts.index_add_(0, feature["rows"], value_counts[name].to(template.dtype))
        components = {
            "same_feature": contributions[name],
            "inter_feature": torch.zeros_like(template),
            "cross_tower": torch.zeros_like(template),
        }
        sources = {}
        zero = torch.zeros_like(template)
        for source_name, source in features.items():
            if source_name == name:
                continue
            kind = "inter_feature" if source["owner"] == feature["owner"] else "cross_tower"
            shared = source["table"] == feature["table"]
            gradient = contributions[source_name] if shared else zero
            if shared:
                components[kind] += gradient
            sources[source_name] = {
                "kind": kind, "shared_table": shared,
                **readout_effect(gradient, reader, counts),
            }
        report[name] = {
            "owner": feature["owner"],
            "table": feature["table"],
            "reader": {
                "shape": list(reader.shape),
                "rank": int((singular_values > rank_threshold).sum().item()),
                "stable_rank": (singular_values.square().sum() / singular_values[0].square()).item(),
                "frobenius_norm": torch.linalg.vector_norm(singular_values).item(),
                "spectral_norm": singular_values[0].item(),
            },
            "components": {
                kind: readout_effect(gradient, reader, counts)
                for kind, gradient in components.items()
            },
            "sources": sources,
        }
    return report
