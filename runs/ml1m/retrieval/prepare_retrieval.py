import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing.ml1m import prepare_ml1m_retrieval
from src.data.preprocessing.sequential import prepare_sequential


RAW_PATH = PROJECT_ROOT / "data/raw/ml1m/ml-1m.zip"
RETRIEVAL_DIR = PROJECT_ROOT / "data/preprocessed/ml1m/retrieval"
STATIC_DIR = RETRIEVAL_DIR / "static"
SEQUENTIAL_DIR = RETRIEVAL_DIR / "sequential"

SPLIT_RATIOS = (0.8, 0.1, 0.1)
REBUILD_STATIC = False
REBUILD_SEQUENTIAL = False
PREPARE_SEQUENTIAL = False


def main():
    build_static = REBUILD_STATIC or not (STATIC_DIR / "metadata.json").exists()
    if build_static:
        prepare_ml1m_retrieval(RAW_PATH, STATIC_DIR, split_ratios=SPLIT_RATIOS)
        print(f"Static artifact: {STATIC_DIR}", flush=True)
    else:
        print(f"Using existing static artifact: {STATIC_DIR}", flush=True)

    if PREPARE_SEQUENTIAL:
        if build_static or REBUILD_SEQUENTIAL or not (SEQUENTIAL_DIR / "metadata.json").exists():
            prepare_sequential(STATIC_DIR, SEQUENTIAL_DIR)
            print(f"Sequential artifact: {SEQUENTIAL_DIR}", flush=True)
        else:
            print(f"Using existing sequential artifact: {SEQUENTIAL_DIR}", flush=True)


if __name__ == "__main__":
    main()
