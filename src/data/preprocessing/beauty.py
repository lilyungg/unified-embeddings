import ast
import json
import re
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from .ml1m import HASH_MASK, HASH_SEEDS, HASH_SERIALIZATION, _hash_values


MISSING = "MISSING"


def _categories_value(paths):
    """Encode the sorted collection of complete paths as one category."""
    paths = [path for path in paths or [] if path]
    if not paths:
        return MISSING
    return json.dumps(sorted(paths), ensure_ascii=False, separators=(",", ":"))


def _load_item_features(metadata_path, item_ids):
    item_indices = {item_id: index for index, item_id in enumerate(item_ids)}
    features = {
        name: np.full(len(item_ids), MISSING, dtype=object)
        for name in ("brand", "categories")
    }
    asin_pattern = re.compile(rb"""['"]asin['"]\s*:\s*['"]([^'"]+)""")

    with ZipFile(metadata_path) as archive:
        with archive.open("metadata.json") as source:
            for line in source:
                # Parse full records only for items in the interaction corpus.
                asin = asin_pattern.search(line).group(1).decode("utf-8")
                index = item_indices.get(asin)
                if index is None:
                    continue
                record = ast.literal_eval(line.decode("utf-8"))
                brand = record.get("brand")
                if brand and brand.strip():
                    features["brand"][index] = brand
                features["categories"][index] = _categories_value(record.get("categories"))

    return features


def prepare_beauty_retrieval(
    ratings_path,
    metadata_path,
    output_dir,
    *,
    split_ratios=(0.8, 0.1, 0.1),
):
    """Prepare full Beauty 2014 interactions and scalar item features; no filtering."""
    ratios = np.asarray(split_ratios, dtype=np.float64)
    interactions = pd.read_csv(
        ratings_path,
        names=["user_id", "item_id", "rating", "timestamp"],
        dtype={"user_id": str, "item_id": str, "rating": np.int64, "timestamp": np.int64},
        keep_default_na=False,
    )

    # Ties retain source row order. Global row position becomes interaction_idx.
    interactions = interactions.sort_values("timestamp", kind="stable", ignore_index=True)
    user_ids, user_idx = np.unique(interactions["user_id"].to_numpy(dtype=str), return_inverse=True)
    item_ids, item_idx = np.unique(interactions["item_id"].to_numpy(dtype=str), return_inverse=True)
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
        ("rating", interactions["rating"].to_numpy()),
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
        raw_values, codes = np.unique(values, return_inverse=True)
        # Convert only the vocabulary, not every item's potentially long paths.
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
        "audit_fields": ["rating"],
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
