import ast
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from .ml1m import HASH_MASK, HASH_SEEDS, HASH_SERIALIZATION, _hash_values


MISSING = "MISSING"


def _load_interactions(reviews_path):
    rows = []
    with ZipFile(reviews_path) as archive:
        with archive.open("steam_new.json") as source:
            for line in source:
                record = ast.literal_eval(line.decode("utf-8"))
                user_id = record.get("user_id")
                if not user_id or not user_id.strip():
                    continue
                rows.append((int(user_id), int(record["product_id"]), record["date"]))

    interactions = pd.DataFrame(rows, columns=["user_id", "item_id", "timestamp"])
    # Dates have day precision; store midnight UTC as Unix seconds.
    interactions["timestamp"] = interactions["timestamp"].to_numpy(
        dtype="datetime64[s]",
    ).astype(np.int64)
    return interactions


def _load_item_features(metadata_path, item_ids):
    item_indices = {item_id: index for index, item_id in enumerate(item_ids)}
    features = {
        name: np.full(len(item_ids), MISSING, dtype=object)
        for name in ("developer", "publisher", "genres")
    }
    with Path(metadata_path).open(encoding="utf-8") as source:
        for line in source:
            record = ast.literal_eval(line)
            item_id = record.get("id")
            if not item_id:
                continue
            index = item_indices.get(int(item_id))
            if index is None:
                continue
            for name in ("developer", "publisher"):
                value = record.get(name)
                if value and value.strip():
                    features[name][index] = value
            genres = record.get("genres")
            if genres:
                features["genres"][index] = json.dumps(
                    sorted(genres), ensure_ascii=False, separators=(",", ":"),
                )
    return features


def prepare_steam_retrieval(
    reviews_path,
    metadata_path,
    output_dir,
    *,
    split_ratios=(0.8, 0.1, 0.1),
):
    """Prepare Steam reviews with known user_id; no other interaction filtering."""
    ratios = np.asarray(split_ratios, dtype=np.float64)
    interactions = _load_interactions(reviews_path)

    # Ties retain source row order. Global row position becomes interaction_idx.
    interactions = interactions.sort_values("timestamp", kind="stable", ignore_index=True)
    user_ids, user_idx = np.unique(interactions["user_id"].to_numpy(), return_inverse=True)
    item_ids, item_idx = np.unique(interactions["item_id"].to_numpy(), return_inverse=True)
    timestamps = interactions["timestamp"].to_numpy(dtype=np.int64)

    # Keep each date entirely within one temporal split.
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

    item_features = _load_item_features(metadata_path, item_ids)
    for name, values in item_features.items():
        # Keep MISSING in every feature vocabulary, even without observed gaps.
        raw_values = np.unique(np.append(values, MISSING))
        codes = np.searchsorted(raw_values, values).astype(np.int64)
        raw_values = raw_values.astype(str)
        vocabulary_dir = output_dir / "vocabularies" / "item" / name
        vocabulary_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "items" / f"{name}.npy", codes, allow_pickle=False)
        np.save(vocabulary_dir / "raw_values.npy", raw_values, allow_pickle=False)
        np.save(vocabulary_dir / "hashes.npy", _hash_values(raw_values, f"item.{name}"), allow_pickle=False)
        features["item"][name] = {"cardinality": len(raw_values)}

    metadata = {
        "split": split,
        "features": features,
        "audit_fields": [],
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
