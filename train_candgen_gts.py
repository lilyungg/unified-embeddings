import datetime
import json
import pathlib
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F

from dataset_utils import load_vklsvd_gts, load_yambda_gts
from embeddings import prehash
from eval_utils import evaluate_gts
from models import TwoTowerFeat
from utils import get_device, load_config


CKPT_DIR = pathlib.Path('checkpoints')


def encode_all(data: dict, method: str, budget: float, base_levels: int,
               probes: int = 1):
    cols = data['q_static'] + data['ev_cols'] + data['item_cols']
    sizes = {c['name']: len(c['vocab']) for c in cols}
    total = sum(sizes.values())
    levels = max(1, round((base_levels or total) * budget))

    vc, offset = {}, 0
    for c in cols:
        n, name = len(c['vocab']), c['name']
        ids = np.arange(n)
        if method == 'Multiplex':
            vc[name] = prehash(ids, tuple(range(probes)), levels * probes,
                               feature_id=name)
        elif method == 'Non-multiplex':
            lev = max(1, round(n / total * levels))
            vc[name] = prehash(ids, (0,), lev, feature_id=name) + offset
            offset += lev
        else:
            vc[name] = ids.reshape(-1, 1) + offset
            offset += n
    n_rows = levels * probes if method == 'Multiplex' else offset

    code = lambda c: vc[c['name']][c['inv']].astype(np.int64)
    QS = np.concatenate([code(c) for c in data['q_static']], axis=1)
    EV = [code(c) for c in data['ev_cols']]
    EVAL = [vc[c['name']][c['eval_idx']].astype(np.int64) for c in data['ev_cols']]
    IC = np.concatenate([code(c) for c in data['item_cols']], axis=1)
    return QS, EV, EVAL, IC, n_rows, total


def run_experiment(method, budget, data, device, cfg, seed=42) -> dict:
    probes = cfg.probes if method == 'Multiplex' else 1
    QS, EV, EVAL, IC, n_rows, total_vocab = encode_all(
        data, method, budget, cfg.base_levels, probes)
    used = torch.tensor(np.unique(np.concatenate(
        [QS.ravel(), IC.ravel()] + [e.ravel() for e in EV])), device=device)

    torch.manual_seed(seed)
    n_q = len(data['q_static']) + len(data['ev_cols'])
    n_i = len(data['item_cols'])
    model = TwoTowerFeat(n_rows, cfg.emb_dim, cfg.k, n_q, n_i,
                         probes=probes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    QSt = torch.tensor(QS, device=device)
    ICt = torch.tensor(IC, device=device)
    EVt = [torch.tensor(e, device=device) for e in EV]
    ev_u = torch.tensor(data['ev_u'], device=device)
    ev_i = torch.tensor(data['ev_i'], device=device)
    Qe = torch.cat([QSt] + [torch.as_tensor(np.tile(v, (len(QSt), 1)),
                                            device=device) for v in EVAL], dim=1)

    counts = np.bincount(data['ev_i'], minlength=data['n_catalog']).astype(np.float64)
    logq = torch.tensor(np.log((counts + 1) / (counts.sum() + data['n_catalog'])),
                        device=device, dtype=torch.float32)

    size_mb = n_rows * (cfg.emb_dim // probes) * 4 / 1e6
    print(f"[{cfg.tag} b={budget}] {method}: table {n_rows:,} rows / "
          f"{size_mb:.3f} MB ({n_rows/total_vocab:.3f}x vocab)", flush=True)

    tb, tb_tag = None, f'{cfg.ts}_gts_{cfg.tag}/b{budget}'
    if cfg.tb:
        from torch.utils.tensorboard import SummaryWriter
        name = method + (f'_seed{seed}' if cfg.runs > 1 else '')
        tb = SummaryWriter(f'{cfg.tb}/{cfg.ts}_gts_{cfg.tag}/b{budget}/{name}')

    n = len(ev_u)
    best, best_state, bad, history = -1.0, None, 0, []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        cap = cfg.batches_per_epoch or (n + cfg.batch - 1) // cfg.batch
        tot, nb = 0.0, 0
        for step in range(min(cap, (n + cfg.batch - 1) // cfg.batch)):
            idx = perm[step * cfg.batch:(step + 1) * cfg.batch]
            qc = torch.cat([QSt[ev_u[idx]]] + [e[idx] for e in EVt], dim=1)
            u = model.user(qc)
            items_b = ev_i[idx]
            if cfg.loss == 'full':
                loss = F.cross_entropy(u @ model.items(ICt).T, items_b)
            else:
                logits = u @ model.items(ICt[items_b]).T - logq[items_b]
                same = items_b.unsqueeze(0) == items_b.unsqueeze(1)
                same.fill_diagonal_(False)
                logits = logits.masked_fill(same, float('-inf'))
                loss = F.cross_entropy(logits,
                                       torch.arange(len(items_b), device=device))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1

        val = evaluate_gts(model, Qe, ICt, data['val'], data['n_catalog'], device)
        ov, _ = model.proj_overlap()
        with torch.no_grad():
            l2 = model.emb.weight[used].norm(dim=1).mean().item()
        history.append({'epoch': epoch, 'loss': round(tot / nb, 4),
                        **{f'val_{k}': v for k, v in val.items()},
                        'proj_overlap': round(ov, 4), 'emb_l2': round(l2, 4)})
        if tb:
            tb.add_scalar(f'ndcg100_val/{tb_tag}', val['ndcg100'], epoch)
            tb.add_scalar(f'recall100_val/{tb_tag}', val['recall100'], epoch)
        print(f"    epoch {epoch:>2}: loss {tot/nb:.4f}  "
              f"val_ndcg100 {val['ndcg100']:.4f}  val_recall100 {val['recall100']:.4f}"
              f"  overlap {ov:.3f}  emb_l2 {l2:.3f}", flush=True)
        if val['ndcg100'] > best:
            best, bad = val['ndcg100'], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if cfg.patience > 0 and bad >= cfg.patience:
            break

    model.load_state_dict(best_state)
    if tb:
        tb.close()
    CKPT_DIR.mkdir(exist_ok=True)
    torch.save({'state_dict': best_state, 'arm': 'candgen_gts', 'method': method,
                'budget': budget, 'seed': seed, 'probes': probes},
               CKPT_DIR / f"candgen_gts-{cfg.tag}-{method.replace(' ', '')}"
                          f"-b{budget}-seed{seed}.pt")
    test = evaluate_gts(model, Qe, ICt, data['test'], data['n_catalog'], device)
    ov_mean, ov_max = model.proj_overlap()
    rk = model.ranks()
    with torch.no_grad():
        l2 = model.emb.weight[used].norm(dim=1).mean().item()
    res = {'dataset': cfg.tag, 'method': method, 'budget': budget,
           'rows': n_rows, 'size_mb': round(size_mb, 4),
           'rows_over_vocab': round(n_rows / total_vocab, 4), 'seed': seed,
           **{f'test_{k}': v for k, v in test.items()},
           'best_val_ndcg100': best, 'proj_overlap_mean': round(ov_mean, 4),
           'proj_overlap_max': round(ov_max, 4),
           'ranks': [round(r, 3) for r in rk],
           'rank_sum': round(float(np.sum(rk)), 3), 'rank_budget': cfg.emb_dim,
           'emb_l2_final': round(l2, 4),
           'epochs_run': len(history), 'history': history}
    print(f"[{cfg.tag} b={budget}] {method}: test recall@100 {test['recall100']:.4f} "
          f"ndcg@100 {test['ndcg100']:.4f}  recall@10 {test['recall10']:.4f}  "
          f"overlap {ov_mean:.3f}", flush=True)
    return res


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    cfg = load_config(parser.parse_args().config)
    cfg.ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    device = get_device()
    if cfg.dataset == 'vklsvd':
        cfg.tag = f'vklsvd_{cfg.subset}_{cfg.positive}'
        data = load_vklsvd_gts(cfg.subset, cfg.positive)
    else:
        cfg.tag = f'{cfg.dataset}_{cfg.interaction}'
        data = load_yambda_gts(cfg.dataset.split('_')[1], cfg.interaction)
    feats = [c['name'] for c in data['q_static'] + data['ev_cols'] + data['item_cols']]
    print(f"device={device}  {cfg.tag}: {data['n_users']:,} users, "
          f"{data['n_catalog']:,} catalog items, {len(data['ev_u']):,} events, "
          f"val/test users {len(data['val']):,}/{len(data['test']):,}\n"
          f"features: {feats}", flush=True)

    results, mx = [], max(cfg.budgets)
    for b in sorted(cfg.budgets, reverse=True):
        methods = ['Non-multiplex', 'Multiplex'] + (['Collisionless'] if b == mx else [])
        if cfg.only:
            exact = [m for m in methods if cfg.only.lower() == m.lower()]
            methods = exact or [m for m in methods if cfg.only.lower() in m.lower()]
        for m in methods:
            for r in range(cfg.runs):
                results.append(run_experiment(m, b, data, device, cfg,
                                              seed=cfg.seed + r))

    out = pathlib.Path(cfg.out); out.mkdir(parents=True, exist_ok=True)
    (out / f'{cfg.ts}_gts_{cfg.tag}.json').write_text(
        json.dumps({'config': vars(cfg), 'results': results}, indent=2))
    print(f'\nsaved {cfg.out}/{cfg.ts}_gts_{cfg.tag}.json')
    print(f"\n{'method':<16}{'budget':>7}{'rows':>10}{'r@10':>8}{'r@100':>8}"
          f"{'n@100':>8}{'cov@100':>9}{'overlap':>9}{'emb_l2':>8}")
    for r in results:
        print(f"{r['method']:<16}{r['budget']:>7}{r['rows']:>10,}"
              f"{r['test_recall10']:>8.4f}{r['test_recall100']:>8.4f}"
              f"{r['test_ndcg100']:>8.4f}{r['test_coverage100']:>9.4f}"
              f"{r['proj_overlap_mean']:>9.3f}{r['emb_l2_final']:>8.3f}")


if __name__ == '__main__':
    main()
