"""Print a markdown table comparing a run's results to the paper (Table 1).

Usage: python report.py experiment_logs/<ts>_<dataset>.json
"""
import json
import sys

# Paper AUCs (Table 1, /100). Keyed by dataset -> budget -> method.
# Budget 1.0 corresponds to the repo's base emb_levels, which map to these
# paper columns: movielens 1.6MB, avazu 3.24MB, criteo 12.5MB.
PAPER = {
    "movielens": {
        "col_names": {1.0: "1.6MB", 0.5: "791kB", 0.1: "158kB"},
        "Collisionless": {1.0: 0.8872},
        "Non-multiplex": {1.0: 0.8537, 0.5: 0.8300, 0.1: 0.7707},
        "Multiplex":     {1.0: 0.8774, 0.5: 0.8693, 0.1: 0.8200},
    },
    "avazu": {
        "col_names": {10.0: "32.4MB", 1.0: "3.24MB", 0.1: "324kB"},
        "Collisionless": {10.0: 0.7735},
        "Non-multiplex": {1.0: 0.7671, 0.1: 0.7510},
        "Multiplex":     {1.0: 0.7718, 0.1: 0.7686},
    },
    "criteo": {
        "col_names": {2.0: "25MB", 1.0: "12.5MB", 0.2: "2.5MB"},
        "Collisionless": {2.0: 0.8070},
        "Non-multiplex": {1.0: 0.7998, 0.2: 0.7944},
        "Multiplex":     {1.0: 0.8047, 0.2: 0.8049},
    },
}


def main(path: str):
    data    = json.load(open(path))
    dataset = next((d for d in PAPER if d in path), "movielens")
    paper   = PAPER[dataset]

    print(f"### {dataset} (config: {json.dumps(data['config'])})\n")
    print("| Budget | Table size | Experiment | AUC | Paper (DCN) | Diff |")
    print("|---|---|---|---|---|---|")
    for b_str, experiments in data["budgets"].items():
        b = float(b_str)
        col = paper["col_names"].get(b, f"{b}x")
        for exp_name, r in experiments.items():
            method = exp_name.split(" + ")[0]
            ref    = paper.get(method, {}).get(b)
            is_dcn = exp_name.endswith("DCN")
            ref_s  = f"{ref:.4f}" if ref is not None and is_dcn else "—"
            diff_s = f"{r['auc'] - ref:+.4f}" if ref is not None and is_dcn else "—"
            print(f"| {b}x | {r['table']['size_mb']:.2f} MB ({col}) | {exp_name} "
                  f"| {r['auc']:.4f} | {ref_s} | {diff_s} |")
    print()


if __name__ == "__main__":
    main(sys.argv[1])
