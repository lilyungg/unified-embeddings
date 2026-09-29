import json
from pathlib import Path

import numpy as np
import pandas as pd

from .ml1m import HASH_MASK, HASH_SEEDS, HASH_SERIALIZATION, _hash_values


MISSING = "MISSING"


def _load_item_feature(mapping_path, item_ids, name):
    mapping = pd.read_parquet(mapping_path)
    mapping = mapping[mapping["item_id"].isin(item_ids)]
    # Numeric sorting makes the category independent of mapping row order.
    values = mapping.groupby("item_id")[f"{name}_id"].agg(
        lambda ids: json.dumps(sorted(ids.tolist()), separators=(",", ":"))
    )
    return values.reindex(item_ids).fillna(MISSING).to_numpy(dtype=object)


def prepare_yambda_retrieval(
    likes_path,
    artist_mapping_path,
    album_mapping_path,
    output_dir,
    *,
    split_ratios=(0.8, 0.1, 0.1),
):
    """Prepare all likes, including duplicate rows; is_organic is audit-only."""
    ratios = np.asarray(split_ratios, dtype=np.float64)
    interactions = pd.read_parquet(likes_path)

    # Ties retain source row order. Global row position becomes interaction_idx.
    interactions = interactions.sort_values("timestamp", kind="stable", ignore_index=True)
    user_ids, user_idx = np.unique(interactions["uid"].to_numpy(dtype=np.int64), return_inverse=True)
    item_ids, item_idx = np.unique(interactions["item_id"].to_numpy(dtype=np.int64), return_inverse=True)
    timestamps = interactions["timestamp"].to_numpy(dtype=np.int64)

    # Keep each timestamp group entirely within one temporal split.
    split = {}
    for name, fraction in zip(("train_end", "validation_end"), np.cumsum(ratios)[:2]):
        end = int(len(interactions) * fraction)
        if end > 0:
            end = int(np.searchsorted(timestamps, timestamps[end - 1], side="right"))
        split[name] = end
        split[f"{name}_timestamp"] = int(timestamps[end - 1]) if end > 0 else None

    output_dir = Path(output_dir)
    interaction_dir = output_dir / "interactions"
    interaction_dir.mkdir(parents=True, exist_ok=True)
    for name, values in (
        ("user_idx", user_idx),
        ("item_idx", item_idx),
        ("timestamp", timestamps),
        ("is_organic", interactions["is_organic"].to_numpy()),
    ):
        np.save(interaction_dir / f"{name}.npy", values.astype(np.int64, copy=False), allow_pickle=False)

    features = {"user": {}, "item": {}, "interaction": {}}
    for owner, raw_ids in (("user", user_ids), ("item", item_ids)):
        entity_dir = output_dir / f"{owner}s"
        entity_dir.mkdir(parents=True, exist_ok=True)
        name = f"{owner}_idx"
        np.save(entity_dir / "raw_id.npy", raw_ids, allow_pickle=False)
        np.save(entity_dir / "id_hashes.npy", _hash_values(raw_ids, f"{owner}.{name}"), allow_pickle=False)
        features[owner][name] = {"cardinality": len(raw_ids)}

    for name, mapping_path in (("artist", artist_mapping_path), ("album", album_mapping_path)):
        values = _load_item_feature(mapping_path, item_ids, name)
        raw_values, codes = np.unique(values, return_inverse=True)
        # Convert only the vocabulary, not every item's potentially long ID list.
        raw_values = raw_values.astype(str)
        vocabulary_dir = output_dir / "vocabularies" / "item" / name
        vocabulary_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "items" / f"{name}.npy", codes.astype(np.int64, copy=False), allow_pickle=False)
        np.save(vocabulary_dir / "raw_values.npy", raw_values, allow_pickle=False)
        np.save(vocabulary_dir / "hashes.npy", _hash_values(raw_values, f"item.{name}"), allow_pickle=False)
        features["item"][name] = {"cardinality": len(raw_values)}

    metadata = {
        "split": split,
        "features": features,
        "audit_fields": ["is_organic"],
        "hash": {
            "algorithm": "xxHash64",
            "serialization": HASH_SERIALIZATION,
            "seeds": list(HASH_SEEDS),
            "mask": HASH_MASK,
        },
        "preprocessing": {"split_ratios": ratios.tolist()},
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
