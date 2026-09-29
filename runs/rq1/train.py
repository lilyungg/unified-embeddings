import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from runs.retrieval.train import parse_args
from runs.retrieval.utils import run


def main(argv=None):
    config, output_dir = parse_args(argv, model="static", loss="bpr")
    return run(config, output_dir=output_dir)


if __name__ == "__main__":
    main()
