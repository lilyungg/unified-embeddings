"""Generate replication plots from a movielens run JSON.

Usage: python plots.py experiment_logs/<ts>_movielens.json [outdir]

Outputs (default outdir plots/):
  tradeoff.png  parameter-accuracy tradeoff, ours vs paper (Table 1 row)
  norms.png     embedding squared-norm growth vs the O(N/M) theory (Fig. 2)
  curves.png    val AUC training curves per memory budget
"""
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from report import PAPER

# validated categorical palette (fixed slot order) + chart chrome
COLORS  = {"Non-multiplex": "#2a78d6", "Multiplex": "#1baf7a", "Collisionless": "#eda100"}
INK     = "#0b0b0b"
MUTED   = "#898781"
GRID    = "#e1e0d9"
BASE    = "#c3c2b7"
SURFACE = "#ffffff"

plt.rcParams.update({
    "font.family": "sans-serif",
    "text.color": INK, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": BASE, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.titlecolor": INK,
})


def method_of(exp: str) -> str:
    return exp.split(" + ")[0]


def load(path: str):
    data = json.load(open(path))
    budgets = sorted((float(b) for b in data["budgets"]), reverse=True)
    return data, budgets


def dcn_cell(data, b: float, method: str):
    return data["budgets"][str(b)].get(f"{method} + DCN")


def plot_tradeoff(data, budgets, paper, out):
    fig, ax = plt.subplots(figsize=(7, 4.6), layout="constrained")
    for method in ["Non-multiplex", "Multiplex"]:
        xs, ys, errs, pxs, pys = [], [], [], [], []
        for b in budgets:
            cell = dcn_cell(data, b, method)
            if cell is None:
                continue
            xs.append(cell["table"]["size_mb"]); ys.append(cell["auc"])
            errs.append(cell.get("auc_std", 0.0))
            ref = paper.get(method, {}).get(b)
            if ref:
                pxs.append(cell["table"]["size_mb"]); pys.append(ref)
        c = COLORS[method]
        ax.errorbar(xs, ys, yerr=errs if any(errs) else None, color=c, lw=2,
                    marker="o", ms=7, capsize=3, label=f"{method} (ours)")
        ax.plot(pxs, pys, color=c, lw=2, ls="--", marker="o", ms=7,
                mfc="none", label=f"{method} (paper)")
        ax.annotate(method, (xs[0], ys[0]), xytext=(8, -3),
                    textcoords="offset points", color=INK, fontsize=9)

    cl = dcn_cell(data, max(budgets), "Collisionless")
    if cl:
        c = COLORS["Collisionless"]
        x = cl["table"]["size_mb"]
        ax.plot([x], [cl["auc"]], color=c, marker="*", ms=14, ls="none",
                label="Collisionless (ours)")
        ref = paper.get("Collisionless", {}).get(max(budgets))
        if ref:
            ax.plot([x], [ref], color=c, marker="*", ms=14, mfc="none",
                    ls="none", label="Collisionless (paper)")

    ax.set_xscale("log")
    ax.set_xlabel("embedding table size, MB (log)")
    ax.set_ylabel("test AUC")
    ax.set_title("Parameter–accuracy tradeoff — MovieLens-1M, ours (solid) vs paper (dashed)")
    ax.legend(fontsize=8, frameon=False, loc="lower right", ncols=2)
    fig.savefig(out / "tradeoff.png", dpi=160)
    plt.close(fig)


def plot_norms(data, budgets, out):
    base = max(budgets)
    ref_cell = dcn_cell(data, base, "Multiplex")
    base_l2 = ref_cell["runs"][0]["emb_l2_final"]
    xs, ys = [], []
    for b in budgets:
        cell = dcn_cell(data, b, "Multiplex")
        xs.append(base / b)
        ys.append(cell["runs"][0]["emb_l2_final"] ** 2 / base_l2 ** 2)

    fig, ax = plt.subplots(figsize=(6, 4.2), layout="constrained")
    ax.plot(xs, xs, color=MUTED, lw=2, ls="--", label="theory  O(N/M)")
    ax.plot(xs, ys, color=COLORS["Multiplex"], lw=2, marker="o", ms=8,
            label="measured (Multiplex + DCN)")
    for x, y in zip(xs, ys):
        ax.annotate(f"×{y:.2f}", (x, y), xytext=(6, -12),
                    textcoords="offset points", color=INK, fontsize=9)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(xs, [f"{x:g}×" for x in xs])
    ax.set_xlabel("compression  N/M  (log)")
    ax.set_ylabel("‖e‖² growth vs full budget (log)")
    ax.set_title("Embedding norms grow as O(N/M) — paper Fig. 2 prediction")
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    fig.savefig(out / "norms.png", dpi=160)
    plt.close(fig)


def plot_curves(data, budgets, out):
    fig, axes = plt.subplots(1, len(budgets), figsize=(11, 3.8), sharey=True,
                             layout="constrained")
    for ax, b in zip(axes, budgets):
        for method in ["Non-multiplex", "Multiplex", "Collisionless"]:
            cell = dcn_cell(data, b, method)
            if cell is None:
                continue
            hist = cell["histories"][0]
            ax.plot([h["epoch"] for h in hist], [h["val_auc"] for h in hist],
                    color=COLORS[method], lw=2, label=method)
        ax.set_title(f"budget {b}×")
        ax.set_xlabel("epoch")
    axes[0].set_ylabel("val AUC")
    axes[0].legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle("Training curves (DCN), val AUC by memory budget", color=INK)
    fig.savefig(out / "curves.png", dpi=160)
    plt.close(fig)


def main():
    path = sys.argv[1]
    out = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "plots")
    out.mkdir(parents=True, exist_ok=True)
    data, budgets = load(path)
    dataset = next((d for d in PAPER if d in path), "movielens")
    plot_tradeoff(data, budgets, PAPER[dataset], out)
    plot_norms(data, budgets, out)
    plot_curves(data, budgets, out)
    print(f"saved: {out}/tradeoff.png, {out}/norms.png, {out}/curves.png")


if __name__ == "__main__":
    main()
