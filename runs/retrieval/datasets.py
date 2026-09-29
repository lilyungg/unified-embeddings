from pathlib import Path

from src.data.datasets.beauty import (
    BeautyRetrievalDataset,
    BeautySequentialDataset,
    load_beauty_retrieval,
    load_beauty_sequential,
)
from src.data.datasets.ml1m import (
    ML1MRetrievalDataset,
    ML1MSequentialDataset,
    load_ml1m_retrieval,
    load_ml1m_sequential,
)
from src.data.datasets.steam import (
    SteamRetrievalDataset,
    SteamSequentialDataset,
    load_steam_retrieval,
    load_steam_sequential,
)
from src.data.datasets.yambda import (
    YambdaRetrievalDataset,
    YambdaSequentialDataset,
    load_yambda_retrieval,
    load_yambda_sequential,
)


PREPROCESSED_DIR = Path(__file__).resolve().parents[2] / "data/preprocessed"

DATASETS = {
    "ml1m": {
        "static": {
            "data_dir": PREPROCESSED_DIR / "ml1m/retrieval/static",
            "load_data": load_ml1m_retrieval,
            "dataset_class": ML1MRetrievalDataset,
            "features": ("user_idx", "gender", "age", "occupation", "zipcode", "item_idx"),
        },
        "sequential": {
            "data_dir": PREPROCESSED_DIR / "ml1m/retrieval/sequential",
            "load_data": load_ml1m_sequential,
            "dataset_class": ML1MSequentialDataset,
            "features": ("item_idx",),
        },
    },
    "beauty": {
        "static": {
            "data_dir": PREPROCESSED_DIR / "beauty/retrieval/static",
            "load_data": load_beauty_retrieval,
            "dataset_class": BeautyRetrievalDataset,
            "features": ("user_idx", "item_idx", "brand", "categories"),
        },
        "sequential": {
            "data_dir": PREPROCESSED_DIR / "beauty/retrieval/sequential",
            "load_data": load_beauty_sequential,
            "dataset_class": BeautySequentialDataset,
            "features": ("item_idx", "brand", "categories"),
        },
    },
    "yambda": {
        "static": {
            "data_dir": PREPROCESSED_DIR / "yambda/retrieval/static",
            "load_data": load_yambda_retrieval,
            "dataset_class": YambdaRetrievalDataset,
            "features": ("user_idx", "item_idx", "artist", "album"),
        },
        "sequential": {
            "data_dir": PREPROCESSED_DIR / "yambda/retrieval/sequential",
            "load_data": load_yambda_sequential,
            "dataset_class": YambdaSequentialDataset,
            "features": ("item_idx", "artist", "album"),
        },
    },
    "steam": {
        "static": {
            "data_dir": PREPROCESSED_DIR / "steam/retrieval/static",
            "load_data": load_steam_retrieval,
            "dataset_class": SteamRetrievalDataset,
            "features": ("user_idx", "item_idx", "developer", "publisher", "genres"),
        },
        "sequential": {
            "data_dir": PREPROCESSED_DIR / "steam/retrieval/sequential",
            "load_data": load_steam_sequential,
            "dataset_class": SteamSequentialDataset,
            "features": ("item_idx", "developer", "publisher", "genres"),
        },
    },
}
