import argparse
import datetime
import json
import pathlib
import torch

from data      import load_movielens, load_avazu, load_criteo
from benchmark import run_dataset, table_report


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ml1m",     default="ml-1m",
                   help="path to ml-1m dir (only read when movielens is not skipped)")
    p.add_argument("--ml-labels", default="wang", choices=["wang", "ge3"],
                   help="movielens labels: wang = Wang et al. 2021 (1-2 -> 0, "
                        "4-5 -> 1, 3s removed, as the paper's pipeline); "
                        "ge3 = rating >= 3 -> 1 (paper's literal description)")
    p.add_argument("--avazu",    default=None,
                   help="local Avazu file: raw train.gz/csv or prepared .parquet "
                        "from prepare_data.py (default: prepared parquet if "
                        "present, else HuggingFace reczoo/Avazu_x4)")
    p.add_argument("--criteo",   default=None,
                   help="prepared Criteo .parquet from prepare_data.py "
                        "(default: prepared parquet if present, else "
                        "HuggingFace reczoo/Criteo_x4)")
    p.add_argument("--skip",     nargs="*", default=[], choices=["movielens", "avazu", "criteo"])
    p.add_argument("--epochs",   type=int,   default=30)
    p.add_argument("--patience", type=int,   default=5,
                   help="early stopping patience on val AUC; 0 disables early stopping")
    p.add_argument("--batch",    type=int,   default=None,
                   help="batch size (default per dataset: movielens 128, "
                        "avazu/criteo 512 — see DATASET_CFG)")
    p.add_argument("--lr",       type=float, default=None,
                   help="learning rate (default per dataset: movielens 1e-3, "
                        "avazu/criteo 2e-4)")
    p.add_argument("--cross",    type=int,   default=None,
                   help="override number of cross layers (default: paper value per dataset)")
    p.add_argument("--dnn",      default=None,
                   help="override DNN widths, comma-separated, e.g. 256,128 "
                        "(default: paper value per dataset)")
    p.add_argument("--only",     default=None,
                   help="run only experiments whose name contains this substring")
    p.add_argument("--with-mlp", action="store_true",
                   help="also train the plain-MLP arm (off by default; DCN only, "
                        "the paper has no MLP arm)")
    p.add_argument("--workers",  type=int, default=4,
                   help="DataLoader workers (raise to 8-16 on many-core servers)")
    p.add_argument("--dropout",  type=float, default=0.0,
                   help="dropout in the DCN DNN stack (paper has none)")
    p.add_argument("--wd",       type=float, default=1e-5,
                   help="Adam weight decay")
    p.add_argument("--runs",     type=int,   default=1,
                   help="independent training runs per experiment (paper uses 5)")
    p.add_argument("--seed",     type=int,   default=42)
    p.add_argument("--paper-protocol", action="store_true",
                   help="evaluate test AUC every epoch and report the best over "
                        "epochs (Tsang & Ahle protocol used by the paper); "
                        "combine with --patience 0 to disable early stopping")
    p.add_argument("--bn",       action="store_true",
                   help="enable BatchNorm in the DCN DNN stack (off by default; "
                        "the paper has none and it costs ~0.5-1 AUC points)")
    p.add_argument("--budgets",  nargs="+",  type=float, default=[1.0],
                   help="memory budget multipliers for emb_levels, e.g. 1.0 0.5 0.1")
    p.add_argument("--fast",     action="store_true")
    p.add_argument("--out",      default="experiment_logs")
    p.add_argument("--tb",       default="runs",
                   help="TensorBoard log dir; empty string disables logging")
    p.add_argument("--dry-run",  action="store_true",
                   help="print embedding table sizes per budget and exit (no training)")
    return p.parse_args()


def print_table_report(name: str, report: dict):
    print(f"\n{name}: total_vocab={report['total_vocab']:,}  d={report['emb_dim']}  "
          f"budget={report['budget']}  emb_levels={report['emb_levels']:,}")
    for method, t in report["tables"].items():
        print(f"  {method:<14} {t['rows']:>9,} rows  {t['size_mb']:>9.4f} MB  "
              f"{t['rows_over_vocab']:>6.3f}x vocab")


def process_dataset(name: str, loaded: tuple, args, device, run_ts: str,
                    out_dir: pathlib.Path) -> dict:
    df, dense, labels, tr, va, te = loaded

    if args.dry_run:
        for b in args.budgets:
            print_table_report(name, table_report(name, df, budget=b))
        return {}

    max_budget = max(args.budgets)
    dnn_dims = tuple(int(d) for d in args.dnn.split(",")) if args.dnn else None
    by_budget = {}
    for b in args.budgets:
        by_budget[str(b)] = run_dataset(
            name, df, labels, tr, va, te,
            device     = device,
            batch_size = args.batch,
            lr         = args.lr,
            max_epochs = args.epochs,
            patience   = args.patience,
            weight_decay = args.wd,
            budget     = b,
            include_collisionless = (b == max_budget),
            tb_dir     = f"{args.tb}/{run_ts}" if args.tb else None,
            num_cross  = args.cross,
            dnn_dims   = dnn_dims,
            dropout    = args.dropout,
            use_bn     = args.bn,
            only       = args.only,
            n_runs     = args.runs,
            seed       = args.seed,
            eval_test_epochs = args.paper_protocol,
            dense      = dense,
            with_mlp   = args.with_mlp,
            num_workers = args.workers,
        )

    (out_dir / f"{run_ts}_{name}.json").write_text(
        json.dumps({"config": vars(args), "budgets": by_budget}, indent=2))
    return by_budget


def main():
    args    = parse_args()
    device  = get_device()
    n_rows  = 1_000_000 if args.fast else None
    run_ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.dry_run:
        print(f"device={device}  budgets={args.budgets}  run={run_ts}", flush=True)

    all_results = {}

    if "movielens" not in args.skip:
        all_results["movielens"] = process_dataset(
            "movielens", load_movielens(args.ml1m, label_mode=args.ml_labels),
            args, device, run_ts, out_dir)

    if "avazu" not in args.skip:
        all_results["avazu"] = process_dataset(
            "avazu", load_avazu(path=args.avazu, n_rows=n_rows), args, device, run_ts, out_dir)

    if "criteo" not in args.skip:
        all_results["criteo"] = process_dataset(
            "criteo", load_criteo(n_rows=n_rows, path=args.criteo),
            args, device, run_ts, out_dir)

    if not args.dry_run:
        (out_dir / f"{run_ts}_summary.json").write_text(
            json.dumps({"config": vars(args), "results": all_results}, indent=2))


if __name__ == "__main__":
    main()
