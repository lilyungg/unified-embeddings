import json
from os.path import relpath
from pathlib import Path

import numpy as np


def prepare_sequential(static_dir, sequential_dir):
    static_dir = Path(static_dir)
    sequential_dir = Path(sequential_dir)
    interaction_dir = static_dir / "interactions"
    event_dir = sequential_dir / "events"

    metadata = json.loads((static_dir / "metadata.json").read_text(encoding="utf-8"))

    user_idx = np.load(interaction_dir / "user_idx.npy", allow_pickle=False)
    num_users = metadata["features"]["user"]["user_idx"]["cardinality"]

    # Static rows are chronological; stable grouping preserves order per user.
    interaction_order = np.argsort(user_idx, kind="stable").astype(np.int64, copy=False)
    counts = np.bincount(user_idx, minlength=num_users)
    user_offsets = np.empty(num_users + 1, dtype=np.int64)
    user_offsets[0] = 0
    np.cumsum(counts, out=user_offsets[1:])

    event_dir.mkdir(parents=True, exist_ok=True)
    np.save(sequential_dir / "user_offsets.npy", user_offsets, allow_pickle=False)
    np.save(event_dir / "interaction_idx.npy", interaction_order, allow_pickle=False)

    # Apply the same permutation to items and all declared interaction features.
    for name in ("item_idx", *metadata["features"]["interaction"]):
        values = np.load(interaction_dir / f"{name}.npy", allow_pickle=False)
        np.save(event_dir / f"{name}.npy", values[interaction_order], allow_pickle=False)

    # Boundaries are absolute exclusive offsets, not counts or a new split.
    for name in ("train_end", "validation_end"):
        global_end = metadata["split"][name]
        prefix_counts = np.bincount(user_idx[:global_end], minlength=num_users)
        user_ends = user_offsets[:-1] + prefix_counts
        np.save(sequential_dir / f"{name}.npy", user_ends, allow_pickle=False)

    sequential_metadata = {"static_dir": relpath(static_dir, sequential_dir)}
    (sequential_dir / "metadata.json").write_text(
        json.dumps(sequential_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
