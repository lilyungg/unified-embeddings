import json
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset


def load_beauty_retrieval(static_dir):
    """Load arrays to share across train/validation/test Dataset instances."""
    static_dir = Path(static_dir)
    metadata = json.loads((static_dir / "metadata.json").read_text(encoding="utf-8"))
    features = metadata["features"]

    interactions = {
        name: np.load(static_dir / "interactions" / f"{name}.npy", allow_pickle=False)
        for name in ("user_idx", "item_idx", *features["interaction"])
    }
    user_features = {
        name: np.load(static_dir / "users" / f"{name}.npy", allow_pickle=False)
        for name in features["user"]
        if name != "user_idx"
    }
    item_features = {
        name: np.load(static_dir / "items" / f"{name}.npy", allow_pickle=False)
        for name in features["item"]
        if name != "item_idx"
    }

    return {
        "static_dir": static_dir,
        "metadata": metadata,
        "interactions": interactions,
        "user_features": user_features,
        "item_features": item_features,
    }


def load_beauty_sequential(sequential_dir):
    """Load shared static arrays and user-grouped event sequences."""
    sequential_dir = Path(sequential_dir)
    metadata = json.loads((sequential_dir / "metadata.json").read_text(encoding="utf-8"))
    data = load_beauty_retrieval((sequential_dir / metadata["static_dir"]).resolve())
    data["sequential"] = {
        name: np.load(sequential_dir / f"{name}.npy", allow_pickle=False)
        for name in ("user_offsets", "train_end", "validation_end")
    }
    data["sequential"]["events"] = {
        name: np.load(sequential_dir / "events" / f"{name}.npy", allow_pickle=False)
        for name in ("item_idx", "interaction_idx")
    }
    return data


def _feature_specs(data):
    return {
        name: {"owner": owner, "cardinality": description["cardinality"]}
        for owner, features in data["metadata"]["features"].items()
        for name, description in features.items()
    }


def _load_feature_hashes(data, feature_names):
    static_dir = data["static_dir"]
    specs = _feature_specs(data)
    hashes = {}
    for name in feature_names:
        owner = specs[name]["owner"]
        if owner in ("user", "item") and name == f"{owner}_idx":
            path = static_dir / f"{owner}s" / "id_hashes.npy"
        else:
            path = static_dir / "vocabularies" / owner / name / "hashes.npy"
        hashes[name] = np.load(path, allow_pickle=False)
    return hashes


class BeautyRetrievalDataset(Dataset):
    """Flat NumPy int64 samples: user_idx, target_item_idx, interaction_idx.

    Evaluation excludes each user's first event, matching sequential targets.
    Item features stay in the catalog, outside query samples.
    """

    def __init__(self, data, split="train"):
        self.data = data
        self.split = split
        boundaries = data["metadata"]["split"]
        train_end = boundaries["train_end"]
        validation_end = boundaries["validation_end"]
        num_interactions = len(data["interactions"]["item_idx"])
        self.start, self.end = {
            "train": (0, train_end),
            "validation": (train_end, validation_end),
            "test": (validation_end, num_interactions),
        }[split]
        self.interaction_indices = range(self.start, self.end)
        if split != "train":
            # Global row order is chronological; only the first event has no history.
            if "first_interaction_idx" not in data:
                _, data["first_interaction_idx"] = np.unique(
                    data["interactions"]["user_idx"], return_index=True,
                )
            self.interaction_indices = np.setdiff1d(
                np.arange(self.start, self.end, dtype=np.int64),
                data["first_interaction_idx"],
                assume_unique=True,
            )

    def __len__(self):
        return len(self.interaction_indices)

    def __getitem__(self, index):
        # Stop split-local iteration before it reaches the next split.
        if not 0 <= index < len(self):
            raise IndexError(index)

        interaction_idx = np.int64(self.interaction_indices[index])
        interactions = self.data["interactions"]
        user_idx = interactions["user_idx"][interaction_idx]
        sample = {
            "user_idx": user_idx,
            "target_item_idx": interactions["item_idx"][interaction_idx],
            "interaction_idx": interaction_idx,
        }
        for name, values in self.data["user_features"].items():
            sample[name] = values[user_idx]
        for name in self.data["metadata"]["features"]["interaction"]:
            sample[name] = interactions[name][interaction_idx]
        return sample

    @property
    def feature_specs(self):
        return _feature_specs(self.data)

    def load_feature_hashes(self, feature_names):
        return _load_feature_hashes(self.data, feature_names)


class BeautySequentialDataset(Dataset):
    """Per-user next-item train samples; per-interaction evaluation samples.

    Histories are right-padded to max_len; history_length marks real events.
    Target and interaction IDs are arrays in train and scalars in evaluation.
    Train needs two events; evaluation needs a nonempty past history.
    """

    def __init__(self, data, split="train", max_len=200):
        self.data = data
        self.split = split
        self.max_len = max_len
        sequential = data["sequential"]

        if split == "train":
            train_lengths = sequential["train_end"] - sequential["user_offsets"][:-1]
            self.user_indices = np.flatnonzero(train_lengths >= 2)
        else:
            boundaries = data["metadata"]["split"]
            num_interactions = len(sequential["events"]["interaction_idx"])
            self.start, self.end = {
                "validation": (boundaries["train_end"], boundaries["validation_end"]),
                "test": (boundaries["validation_end"], num_interactions),
            }[split]
            interaction_idx = sequential["events"]["interaction_idx"]
            positions = np.flatnonzero(
                (interaction_idx >= self.start) & (interaction_idx < self.end)
            )
            # Local evaluation index -> position in the user-grouped event array.
            self.target_positions = np.empty(self.end - self.start, dtype=np.int64)
            self.target_positions[interaction_idx[positions] - self.start] = positions
            user_idx = data["interactions"]["user_idx"][self.start:self.end]
            has_history = self.target_positions > sequential["user_offsets"][user_idx]
            # Excluded targets remain in the timeline as history for later events.
            self.target_positions = self.target_positions[has_history]

    def __len__(self):
        if self.split == "train":
            return len(self.user_indices)
        return len(self.target_positions)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)

        if self.split == "train":
            user_idx = self.user_indices[index]
            sample = self._train_sample(user_idx)
        else:
            target_position = self.target_positions[index]
            interaction_idx = self.data["sequential"]["events"]["interaction_idx"][target_position]
            user_idx = self.data["interactions"]["user_idx"][interaction_idx]
            sample = self._eval_sample(user_idx, target_position)

        sample["user_idx"] = user_idx
        for name, values in self.data["user_features"].items():
            sample[name] = values[user_idx]
        return sample

    def _pad(self, values):
        padded = np.zeros(self.max_len, dtype=np.int64)
        padded[:len(values)] = values
        return padded

    def _train_sample(self, user_idx):
        """Shift the last max_len + 1 train events into next-item pairs."""
        sequential = self.data["sequential"]
        events = sequential["events"]
        end = sequential["train_end"][user_idx]
        start = max(sequential["user_offsets"][user_idx], end - self.max_len - 1)
        return {
            "input_item_idx": self._pad(events["item_idx"][start:end - 1]),
            "target_item_idx": self._pad(events["item_idx"][start + 1:end]),
            "interaction_idx": self._pad(events["interaction_idx"][start + 1:end]),
            "history_length": np.int64(end - start - 1),
        }

    def _eval_sample(self, user_idx, target_position):
        """Include earlier train/validation/test events, but not the target."""
        sequential = self.data["sequential"]
        events = sequential["events"]
        start = max(sequential["user_offsets"][user_idx], target_position - self.max_len)
        return {
            "input_item_idx": self._pad(events["item_idx"][start:target_position]),
            "target_item_idx": events["item_idx"][target_position],
            "interaction_idx": events["interaction_idx"][target_position],
            "history_length": np.int64(target_position - start),
        }

    @property
    def feature_specs(self):
        return _feature_specs(self.data)

    def load_feature_hashes(self, feature_names):
        return _load_feature_hashes(self.data, feature_names)
