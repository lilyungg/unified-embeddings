import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import xxhash


HASH_SEEDS = (0, 1, 2, 3)
HASH_MASK = (1 << 63) - 1
HASH_SERIALIZATION = "namespace_utf8_nul_typed_int64le_or_utf8_v1"


def _hash_values(raw_values, namespace):
    hashes = np.empty((len(raw_values), len(HASH_SEEDS)), dtype=np.int64)
    prefix = namespace.encode("utf-8") + b"\0"
    integer_values = raw_values.dtype.kind in "iu"

    for row, value in enumerate(raw_values):
        if integer_values:
            payload = b"i" + int(value).to_bytes(8, byteorder="little", signed=True)
        else:
            payload = b"s" + str(value).encode("utf-8")
        for component, seed in enumerate(HASH_SEEDS):
            hashes[row, component] = (
                xxhash.xxh64(prefix + payload, seed=seed).intdigest() & HASH_MASK
            )
    return hashes


def prepare_ml1m_retrieval(
    raw_path,
    output_dir,
    *,
    split_ratios=(0.8, 0.1, 0.1),
):
    ratios = np.asarray(split_ratios, dtype=np.float64)

    with ZipFile(raw_path) as archive:
        with archive.open("ml-1m/ratings.dat") as source:
            interactions = pd.read_csv(
                source,
                sep="::",
                engine="python",
                names=["user_id", "item_id", "rating", "timestamp"],
                dtype=np.int64,
            )
        with archive.open("ml-1m/users.dat") as source:
            users = pd.read_csv(
                source,
                sep="::",
                engine="python",
                names=["user_id", "gender", "age", "occupation", "zipcode"],
                dtype={
                    "user_id": np.int64,
                    "gender": str,
                    "age": np.int64,
                    "occupation": np.int64,
                    "zipcode": str,
                },
                keep_default_na=False,
            )

    # Ties retain source row order. Global row position becomes interaction_idx.
    interactions = interactions.sort_values("timestamp", kind="stable", ignore_index=True)
    user_ids, user_idx = np.unique(interactions["user_id"].to_numpy(), return_inverse=True)
    item_ids, item_idx = np.unique(interactions["item_id"].to_numpy(), return_inverse=True)
    timestamps = interactions["timestamp"].to_numpy(dtype=np.int64)

    # Ratios request row boundaries; move each boundary past its timestamp group.
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
        np.save(
            interaction_dir / f"{name}.npy",
            values.astype(np.int64, copy=False),
            allow_pickle=False,
        )

    features = {"user": {}, "item": {}, "interaction": {}}
    for owner, raw_ids in (("user", user_ids), ("item", item_ids)):
        entity_dir = output_dir / f"{owner}s"
        entity_dir.mkdir(parents=True, exist_ok=True)
        name = f"{owner}_idx"
        np.save(entity_dir / "raw_id.npy", raw_ids, allow_pickle=False)
        np.save(
            entity_dir / "id_hashes.npy",
            _hash_values(raw_ids, f"{owner}.{name}"),
            allow_pickle=False,
        )
        features[owner][name] = {"cardinality": len(raw_ids)}

    # The interaction corpus defines the universe; metadata only adds features.
    users = users.set_index("user_id").loc[user_ids]
    for name in ("gender", "age", "occupation", "zipcode"):
        dtype = str if name in ("gender", "zipcode") else np.int64
        raw_values, codes = np.unique(users[name].to_numpy(dtype=dtype), return_inverse=True)
        vocabulary_dir = output_dir / "vocabularies" / "user" / name
        vocabulary_dir.mkdir(parents=True, exist_ok=True)
        np.save(
            output_dir / "users" / f"{name}.npy",
            codes.astype(np.int64, copy=False),
            allow_pickle=False,
        )
        np.save(vocabulary_dir / "raw_values.npy", raw_values, allow_pickle=False)
        np.save(
            vocabulary_dir / "hashes.npy",
            _hash_values(raw_values, f"user.{name}"),
            allow_pickle=False,
        )
        features["user"][name] = {"cardinality": len(raw_values)}

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
