import datetime
import json
import pathlib
from argparse import ArgumentParser

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

from dataset_utils import load as load_retrieval
from dataset_utils import load_movielens
from embeddings import prehash
from eval_utils import evaluate_candgen
from models import TwoTower
from utils import get_device, load_config


ML_USER_COLS = ['user_id', 'gender', 'age', 'occupation', 'zip']
CKPT_DIR = pathlib.Path('checkpoints')


def build_data(cfg) -> dict:
    if cfg.dataset == 'movielens':
        df, _, labels, _, _, _ = load_movielens(cfg.ml1m)
        df = df.with_columns(pl.Series('label', labels)).filter(pl.col('label') == 1)
        users  = sorted(df['user_id'].unique().to_list(), key=str)
        movies = sorted(df['movie_id'].unique().to_list(), key=str)
        u_idx = {u: i for i, u in enumerate(users)}
        m_idx = {m: i for i, m in enumerate(movies)}
        du = df.unique(subset=['user_id'], keep='first').sort('user_id')
        user_vals = {c: du[c].to_numpy() for c in ML_USER_COLS}
        item_vals = np.array(movies)
        pairs = np.array([[u_idx[u], m_idx[m]] for u, m in
                          zip(df['user_id'].to_list(), df['movie_id'].to_list())])
        rng = np.random.default_rng(cfg.seed)
        perm = rng.permutation(len(pairs))
        t, v = int(0.8 * len(pairs)), int(0.9 * len(pairs))
        tr, va, te = pairs[perm[:t]], pairs[perm[t:v]], pairs[perm[v:]]
        n_users, n_items = len(users), len(movies)
        print(f'movielens: {len(pairs):,} positive pairs, {n_users:,} users, '
              f'{n_items:,} movies; splits {len(tr):,}/{len(va):,}/{len(te):,}',
              flush=True)
    else:
        d = load_retrieval(cfg.dataset)
        user_vals = {'user_id': d['users']}
        item_vals = d['items']
        tr, va, te = d['train'], d['val'], d['test']
        n_users, n_items = len(d['users']), len(d['items'])

    seen_tr    = _csr(tr, n_users)
    seen_trval = _csr(np.concatenate([tr, va]), n_users)
    return {'user_vals': user_vals, 'item_vals': item_vals,
            'user_cols': list(user_vals), 'n_users': n_users, 'n_items': n_items,
            'train': tr, 'val': va, 'test': te,
            'seen_tr': seen_tr, 'seen_trval': seen_trval}


def _csr(pairs: np.ndarray, n_users: int):
    order = np.argsort(pairs[:, 0], kind='stable')
    p = pairs[order]
    offs = np.searchsorted(p[:, 0], np.arange(n_users + 1))
    return p, offs


def encode_tables(data: dict, method: str, budget: float, base_levels: int,
                  probes: int = 1):
    cols = data['user_cols'] + ['item_id']
    vals = {**data['user_vals'], 'item_id': data['item_vals']}
    vocabs = {c: np.unique(vals[c]) for c in cols}
    sizes  = {c: len(vocabs[c]) for c in cols}
    total  = sum(sizes.values())
    levels = max(1, round(base_levels * budget))

    codes, offset = {}, 0
    for c in cols:
        v = vals[c]
        if method == 'Multiplex':
            codes[c] = prehash(v, tuple(range(probes)), levels * probes, feature_id=c)
        elif method == 'Non-multiplex':
            lev = max(1, round(sizes[c] / total * levels))
            codes[c] = prehash(v, (0,), lev, feature_id=c) + offset
            offset += lev
        else:
            lut = {x: i for i, x in enumerate(vocabs[c])}
            codes[c] = np.array([lut[x] for x in v],
                                dtype=np.int64).reshape(-1, 1) + offset
            offset += sizes[c]
    n_rows = levels * probes if method == 'Multiplex' else offset

    user_codes = np.concatenate([codes[c] for c in data['user_cols']], axis=1)
    return user_codes.astype(np.int64), codes['item_id'].astype(np.int64), n_rows, total


def emb_l2_used(model, used) -> float:
    with torch.no_grad():
        return model.emb.weight[used].norm(dim=1).mean().item()


def build_model(cfg, data, method, budget, seed):
    probes = cfg.probes if method == 'Multiplex' else 1
    user_codes, item_codes, n_rows, total_vocab = encode_tables(
        data, method, budget, cfg.base_levels or data['total_vocab'], probes)
    torch.manual_seed(seed)
    model = TwoTower(n_rows, cfg.emb_dim, cfg.k, len(data['user_cols']),
                     worst_init=cfg.worst_init, seed=seed,
                     score=cfg.score, temp=cfg.temp, probes=probes)
    return model, user_codes, item_codes, n_rows, total_vocab, probes


def run_experiment(method, budget, data, device, cfg, ts, seed=42) -> dict:
    model, user_codes, item_codes, n_rows, total_vocab, probes = build_model(
        cfg, data, method, budget, seed)
    model = model.to(device)
    used = torch.tensor(np.unique(np.concatenate(
        [user_codes.ravel(), item_codes.ravel()])), device=device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    uc = torch.tensor(user_codes, device=device)
    ic = torch.tensor(item_codes, device=device)
    tr = data['train']
    tr_u = torch.tensor(tr[:, 0], device=device)
    tr_i = torch.tensor(tr[:, 1], device=device)

    counts = np.bincount(tr[:, 1], minlength=data['n_items']).astype(np.float64)
    logq = torch.tensor(np.log((counts + 1) / (counts.sum() + data['n_items'])),
                        device=device, dtype=torch.float32)

    size_mb = n_rows * (cfg.emb_dim // probes) * 4 / 1e6
    print(f'[{cfg.dataset} b={budget}] {method}: table {n_rows:,} rows / '
          f'{size_mb:.3f} MB ({n_rows/total_vocab:.3f}x vocab)', flush=True)

    best, best_state, no_improve, history = -1.0, None, 0, []
    n = len(tr_u)
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        total_loss, n_batches = 0.0, 0
        for s in range(0, n, cfg.batch):
            idx = perm[s:s + cfg.batch]
            u = model.user(uc[tr_u[idx]])
            if cfg.loss == 'full':
                loss = F.cross_entropy(u @ model.items(ic).T, tr_i[idx])
            else:
                items_b = tr_i[idx]
                logits = u @ model.items(ic[items_b]).T - logq[items_b]
                same = items_b.unsqueeze(0) == items_b.unsqueeze(1)
                same.fill_diagonal_(False)
                logits = logits.masked_fill(same, float('-inf'))
                loss = F.cross_entropy(logits, torch.arange(len(items_b), device=device))
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item(); n_batches += 1

        val = evaluate_candgen(model, uc, ic, data['val'], data['seen_tr'], device)
        ov_mean, _ = model.proj_overlap()
        row = {'epoch': epoch, 'loss': round(total_loss / n_batches, 4),
               **{f'val_{k}': v for k, v in val.items()},
               'proj_overlap': round(ov_mean, 4),
               'emb_l2': round(emb_l2_used(model, used), 4)}
        history.append(row)
        print(f"    epoch {epoch:>2}: loss {row['loss']:.4f}  "
              f"val_hr10 {val['hr10']:.4f}  overlap {ov_mean:.3f}  "
              f"emb_l2 {row['emb_l2']:.3f}", flush=True)

        if val['hr10'] > best:
            best, no_improve = val['hr10'], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
        if cfg.patience > 0 and no_improve >= cfg.patience:
            break

    model.load_state_dict(best_state)
    CKPT_DIR.mkdir(exist_ok=True)
    torch.save({'state_dict': best_state, 'arm': 'candgen', 'method': method,
                'budget': budget, 'seed': seed, 'probes': probes},
               CKPT_DIR / f"candgen-{cfg.dataset}-{method.replace(' ', '')}"
                          f"-b{budget}-seed{seed}.pt")
    test = evaluate_candgen(model, uc, ic, data['test'], data['seen_trval'], device)
    ov_mean, ov_max = model.proj_overlap()
    rk = model.ranks()
    res = {'dataset': cfg.dataset, 'method': method, 'budget': budget,
           'rows': n_rows, 'size_mb': round(size_mb, 4),
           'rows_over_vocab': round(n_rows / total_vocab, 4), 'seed': seed,
           **{f'test_{k}': v for k, v in test.items()},
           'best_val_hr10': best, 'proj_overlap_mean': round(ov_mean, 4),
           'proj_overlap_max': round(ov_max, 4),
           'ranks': [round(r, 3) for r in rk],
           'rank_sum': round(float(np.sum(rk)), 3), 'rank_budget': cfg.emb_dim,
           'emb_l2_final': round(emb_l2_used(model, used), 4),
           'epochs_run': len(history), 'history': history}
    print(f"[{cfg.dataset} b={budget}] {method}: test HR@10 {test['hr10']:.4f} "
          f"NDCG@10 {test['ndcg10']:.4f}  overlap {ov_mean:.3f}  "
          f"rank_sum {res['rank_sum']:.1f}/{cfg.emb_dim}", flush=True)
    return res


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    cfg = load_config(parser.parse_args().config)

    device = get_device()
    print(f'device={device}  dataset={cfg.dataset}  budgets={list(cfg.budgets)}', flush=True)
    data = build_data(cfg)
    cols = data['user_cols'] + ['item_id']
    vals = {**data['user_vals'], 'item_id': data['item_vals']}
    data['total_vocab'] = sum(len(np.unique(vals[c])) for c in cols)

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    results = []
    max_budget = max(cfg.budgets)
    for b in sorted(cfg.budgets, reverse=True):
        methods = ['Non-multiplex', 'Multiplex']
        if b == max_budget:
            methods.append('Collisionless')
        if cfg.only:
            exact = [m for m in methods if cfg.only.lower() == m.lower()]
            methods = exact or [m for m in methods if cfg.only.lower() in m.lower()]
        for method in methods:
            for r in range(cfg.runs):
                results.append(run_experiment(method, b, data, device, cfg, ts,
                                              seed=cfg.seed + r))

    out = pathlib.Path(cfg.out); out.mkdir(parents=True, exist_ok=True)
    (out / f'{ts}_candgen_{cfg.dataset}.json').write_text(
        json.dumps({'config': vars(cfg), 'results': results}, indent=2))
    print(f'\nsaved {cfg.out}/{ts}_candgen_{cfg.dataset}.json')
    print(f"\n{'method':<16}{'budget':>7}{'rows':>9}{'HR@10':>8}{'NDCG@10':>9}"
          f"{'overlap':>9}{'ranks':>8}{'emb_l2':>8}")
    for r in results:
        print(f"{r['method']:<16}{r['budget']:>7}{r['rows']:>9,}"
              f"{r['test_hr10']:>8.4f}{r['test_ndcg10']:>9.4f}"
              f"{r['proj_overlap_mean']:>9.3f}{r['rank_sum']:>8.1f}"
              f"{r['emb_l2_final']:>8.3f}")


if __name__ == '__main__':
    main()
