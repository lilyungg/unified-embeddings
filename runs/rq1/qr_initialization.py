import math
import sys
from pathlib import Path

import torch


if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from runs.retrieval.train import parse_args
from runs.retrieval.utils import run


def match_product_variance(feature_encoder, tables, feature_hashes):
    """Match QR product variance to one uniform(-0.05, 0.05) lookup at initialization."""
    lookup_std = 0.05 / math.sqrt(3)
    with torch.no_grad():
        for settings, embedder in zip(tables, feature_encoder.embedders):
            if settings["method"] != "qr":
                continue
            components = len(embedder.tables)
            scale = lookup_std ** (1 / components - 1)
            for table in embedder.tables:
                table.weight.mul_(scale)
            print(
                f"QR init control: features={list(settings['features'])}, "
                f"components={components}, component_scale={scale:.8f}, "
                f"target_product_std={lookup_std:.8f}",
                flush=True,
            )
    return tables


def main(argv=None):
    config, output_dir = parse_args(argv)
    config = {"model": "static", "approach": "qr", "loss": "bpr", **config}
    return run(config, output_dir, prepare_embeddings=match_product_variance)


if __name__ == "__main__":
    main()
