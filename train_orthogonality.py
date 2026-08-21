import datetime
import json
import pathlib
from argparse import ArgumentParser

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset_utils import EmbDataset, load_movielens
from embeddings import UnifiedEmbedding, prehash
from eval_utils import evaluate_auc
from models import SingleLayer
from utils import get_device, load_config


def theta_mean_angle_deg(theta: torch.Tensor) -> float:
    with torch.no_grad():
        w = theta / theta.norm(dim=1, keepdim=True)
        cos = (w @ w.T).clamp(-1.0, 1.0)
        iu = torch.triu_indices(len(w), len(w), offset=1)
        return torch.rad2deg(torch.acos(cos[iu[0], iu[1]])).mean().item()


def emb_l2_used(emb: UnifiedEmbedding, used_ids: np.ndarray) -> float:
    with torch.no_grad():
        w = emb.embedding.weight[torch.from_numpy(used_ids)]
        return w.norm(dim=1).mean().item()


def run_budget(budget: float, df, labels, tr, va, device, cfg) -> dict:
    M    = max(1, round(cfg.emb_levels * budget))
    d    = cfg.emb_dim
    cols = df.columns

    hash_data = np.concatenate(
        [prehash(df[c].to_numpy(), (0,), M, feature_id=c) for c in cols], axis=1)
    used_ids = np.unique(hash_data)

    tr_l = DataLoader(EmbDataset(hash_data[tr], labels[tr]),
                      batch_size=cfg.batch, shuffle=True)
    va_l = DataLoader(EmbDataset(hash_data[va], labels[va]), batch_size=cfg.batch)

    model = SingleLayer(M, len(cols), d).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    crit = nn.BCEWithLogitsLoss()

    history = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for x, xd, y in tr_l:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            crit(model(x), y).backward()
            opt.step()
        row = {
            'epoch':     epoch,
            'angle_deg': round(theta_mean_angle_deg(model.theta), 2),
            'emb_l2':    round(emb_l2_used(model.emb, used_ids), 4),
            'val_auc':   round(evaluate_auc(model, va_l, device), 4),
        }
        history.append(row)
        print(f"  b={budget} M={M:,} epoch {epoch}: angle {row['angle_deg']:.1f}deg  "
              f"emb_l2 {row['emb_l2']:.3f}  val_auc {row['val_auc']:.4f}", flush=True)

    final = history[-1]
    return {
        'budget': budget, 'rows': M, 'size_mb': round(M * d * 4 / 1e6, 4),
        'used_rows': len(used_ids),
        'angle_deg': final['angle_deg'], 'emb_l2': final['emb_l2'],
        'emb_l2_sq': round(final['emb_l2'] ** 2, 4),
        'val_auc': final['val_auc'], 'history': history,
    }


def plot(results: list, out: pathlib.Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    INK, MUTED, GRID, BLUE, AQUA = '#0b0b0b', '#898781', '#e1e0d9', '#2a78d6', '#1baf7a'
    plt.rcParams.update({
        'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED,
        'axes.edgecolor': '#c3c2b7', 'axes.grid': True, 'grid.color': GRID,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.titlesize': 10, 'axes.titlecolor': INK, 'figure.facecolor': 'white',
    })
    rows   = [r['rows'] for r in results]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')

    ax1.plot(rows, [r['emb_l2'] for r in results], color=AQUA, lw=2, marker='o', ms=7)
    ax1.set_xscale('log')
    ax1.set_xlabel('table rows M (log)')
    ax1.set_ylabel('mean embedding L2 norm (used rows)')
    ax1.set_title('Norms, single-layer @10 epochs (flat — cf. O(N/M) in the DCN runs)')

    ax2.plot(rows, [r['angle_deg'] for r in results], color=BLUE, lw=2, marker='o', ms=7)
    ax2.axhline(90, color=MUTED, lw=1.5, ls='--')
    ax2.annotate('orthogonal (90°)', (rows[len(rows) // 2], 90), xytext=(0, 6),
                 textcoords='offset points', color=MUTED, fontsize=9, ha='center')
    ax2.set_xscale('log')
    ax2.set_xlabel('table rows M (log)')
    ax2.set_ylabel('mean pairwise angle of θ_t, degrees')
    ax2.set_title('θ_t orthogonalize; stronger for small M (init: 0°)')

    fig.suptitle('Single-layer model on MovieLens — paper Fig. 2 predictions',
                 color=INK, fontsize=11)
    fig.savefig(out, dpi=160)
    print(f'saved {out}', flush=True)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    cfg = load_config(parser.parse_args().config)

    device = get_device()
    df, _, labels, tr, va, te = load_movielens(cfg.ml1m)
    print(f'device={device}  budgets={list(cfg.budgets)}', flush=True)

    results = [run_budget(b, df, labels, tr, va, device, cfg)
               for b in sorted(cfg.budgets, reverse=True)]

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = pathlib.Path(cfg.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f'{ts}_orthogonality.json').write_text(
        json.dumps({'config': vars(cfg), 'results': results}, indent=2))

    pathlib.Path('plots').mkdir(exist_ok=True)
    plot(results, pathlib.Path('plots/orthogonality.png'))

    print('\nbudget    rows      angle°   emb_l2   val_auc')
    for r in results:
        print(f"{r['budget']:>6}  {r['rows']:>8,}  {r['angle_deg']:>7.1f}  "
              f"{r['emb_l2']:>7.3f}  {r['val_auc']:.4f}")


if __name__ == '__main__':
    main()
