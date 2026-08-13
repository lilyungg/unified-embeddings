import argparse
import datetime
import json
import pathlib

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from torch import nn

from retrieval_data import YAMBDA, _fetch, fetch_vklsvd
from run import get_device
from ue import prehash

EMB_DIM = 30
KS = (10, 50, 100)
TOPN = 100
LISTEN_THRESHOLD = 50


def _col(name, values):
    vocab, inv = np.unique(values, return_inverse=True)
    return {'name': name, 'vocab': vocab, 'inv': inv.astype(np.int64)}


def _targets(us, items, u_map, cat_map):
    out = {}
    for u, i in zip(us, items):
        if u in u_map:
            out.setdefault(u_map[u], []).append(cat_map.get(i, -1))
    return {u: (len(v), np.array([x for x in v if x >= 0], dtype=np.int64))
            for u, v in out.items()}


def load_yambda_gts(size: str, interaction: str) -> dict:
    key = f'yambda_{size}' if interaction == 'likes' else f'yambda_{size}_{interaction}'
    cols = ['uid', 'item_id', 'timestamp']
    if interaction == 'multi':
        cols += ['event_type', 'played_ratio_pct', 'track_length_seconds']
    elif interaction == 'listens':
        cols += ['played_ratio_pct', 'track_length_seconds']
    df = pl.read_parquet(_fetch(key, ext='parquet'), columns=cols)
    if interaction == 'listens':
        df = df.filter(pl.col('played_ratio_pct') >= LISTEN_THRESHOLD)
    elif interaction == 'multi':
        df = df.filter((pl.col('event_type') != 'listen')
                       | (pl.col('played_ratio_pct') >= LISTEN_THRESHOLD))

    last, day, gap = YAMBDA['LAST'], YAMBDA['DAY'], YAMBDA['GAP']
    test_ts = last - day
    tr = df.filter(pl.col('timestamp') < test_ts - gap - day - gap)
    va = df.filter((pl.col('timestamp') >= test_ts - day - gap)
                   & (pl.col('timestamp') < test_ts - gap))
    te = df.filter(pl.col('timestamp') >= test_ts)
    if interaction == 'multi':
        va, te = (s.filter(pl.col('event_type') == 'like') for s in (va, te))

    users = np.unique(tr['uid'].to_numpy())
    catalog = np.unique(tr['item_id'].to_numpy())
    u_map = {u: i for i, u in enumerate(users)}
    cat_map = {x: i for i, x in enumerate(catalog)}
    ev_u = np.array([u_map[u] for u in tr['uid'].to_numpy()], dtype=np.int64)
    ev_i = np.array([cat_map[x] for x in tr['item_id'].to_numpy()], dtype=np.int64)

    item_cols = [_col('item_id', catalog)]
    if interaction != 'likes':
        tl = (tr.group_by('item_id').agg(pl.col('track_length_seconds').max())
                .drop_nulls())
        lut = dict(zip(tl['item_id'].to_list(), tl['track_length_seconds'].to_list()))
        buck = np.array([int(np.log2(1 + lut.get(x, 0))) if x in lut else -1
                         for x in catalog], dtype=np.int64)
        item_cols.append(_col('tl_bucket', buck))

    ev_cols = []
    if interaction == 'multi':
        et = tr['event_type'].cast(pl.Utf8).to_numpy()
        c = _col('event_type', et)
        c['eval_idx'] = int(np.searchsorted(c['vocab'], 'like'))
        ev_cols.append(c)

    return {'q_static': [_col('user_id', users)], 'ev_cols': ev_cols,
            'item_cols': item_cols, 'ev_u': ev_u, 'ev_i': ev_i,
            'n_users': len(users), 'n_catalog': len(catalog),
            'val': _targets(va['uid'].to_numpy(), va['item_id'].to_numpy(), u_map, cat_map),
            'test': _targets(te['uid'].to_numpy(), te['item_id'].to_numpy(), u_map, cat_map)}


def load_vklsvd_gts(subset: str, positive: str) -> dict:
    base = 'interactions' if subset == 'whole' else f'subsamples/{subset}'
    need = ['user_id', 'item_id', 'like'] + (['timespent'] if positive == 'watch' else [])

    items_meta = pl.read_parquet(fetch_vklsvd('metadata/items_metadata.parquet'),
                                 columns=['item_id', 'author_id', 'duration'])

    def positives(rel):
        df = pl.read_parquet(fetch_vklsvd(rel), columns=need)
        if positive == 'like':
            df = df.filter(pl.col('like'))
        else:
            df = (df.join(items_meta.select('item_id', 'duration'), on='item_id', how='left')
                    .filter((pl.col('duration') > 0)
                            & (pl.col('timespent') >= 0.5 * pl.col('duration'))))
        return df.select('user_id', 'item_id')

    tr = pl.concat([positives(f'{base}/train/week_{i:02}.parquet') for i in range(25)])
    va = positives(f'{base}/validation/week_25.parquet')
    te = positives(f'{base}/test/week_26.parquet')

    users = np.unique(tr['user_id'].to_numpy())
    catalog = np.unique(tr['item_id'].to_numpy())
    u_map = {u: i for i, u in enumerate(users)}
    cat_map = {x: i for i, x in enumerate(catalog)}
    ev_u = np.array([u_map[u] for u in tr['user_id'].to_numpy()], dtype=np.int64)
    ev_i = np.array([cat_map[x] for x in tr['item_id'].to_numpy()], dtype=np.int64)

    um = (pl.DataFrame({'user_id': users})
          .join(pl.read_parquet(fetch_vklsvd('metadata/users_metadata.parquet'),
                                columns=['user_id', 'age', 'gender', 'geo']),
                on='user_id', how='left').fill_null(0))
    im = (pl.DataFrame({'item_id': catalog})
          .join(items_meta, on='item_id', how='left').fill_null(0))

    return {'q_static': [_col('user_id', users),
                         _col('age', um['age'].to_numpy()),
                         _col('gender', um['gender'].to_numpy()),
                         _col('geo', um['geo'].to_numpy())],
            'ev_cols': [],
            'item_cols': [_col('item_id', catalog),
                          _col('author_id', im['author_id'].to_numpy()),
                          _col('duration', im['duration'].to_numpy())],
            'ev_u': ev_u, 'ev_i': ev_i,
            'n_users': len(users), 'n_catalog': len(catalog),
            'val': _targets(va['user_id'].to_numpy(), va['item_id'].to_numpy(), u_map, cat_map),
            'test': _targets(te['user_id'].to_numpy(), te['item_id'].to_numpy(), u_map, cat_map)}


def encode_all(data: dict, method: str, budget: float, base_levels: int):
    cols = data['q_static'] + data['ev_cols'] + data['item_cols']
    sizes = {c['name']: len(c['vocab']) for c in cols}
    total = sum(sizes.values())
    levels = max(1, round((base_levels or total) * budget))

    vc, offset = {}, 0
    for c in cols:
        n, name = len(c['vocab']), c['name']
        ids = np.arange(n)
        if method == 'Multiplex':
            vc[name] = prehash(ids, (0,), levels, feature_id=name)[:, 0]
        elif method == 'Non-multiplex':
            lev = max(1, round(n / total * levels))
            vc[name] = prehash(ids, (0,), lev, feature_id=name)[:, 0] + offset
            offset += lev
        else:
            vc[name] = ids + offset
            offset += n
    n_rows = levels if method == 'Multiplex' else offset

    code = lambda c: vc[c['name']][c['inv']].astype(np.int64)
    QS = np.stack([code(c) for c in data['q_static']], axis=1)
    EV = [vc[c['name']][c['inv']].astype(np.int64) for c in data['ev_cols']]
    EVAL = [int(vc[c['name']][c['eval_idx']]) for c in data['ev_cols']]
    IC = np.stack([code(c) for c in data['item_cols']], axis=1)
    return QS, EV, EVAL, IC, n_rows, total


class TwoTowerFeat(nn.Module):

    def __init__(self, n_rows, d, k, n_q, n_i) -> None:
        super().__init__()
        self.emb = nn.Embedding(n_rows, d)
        nn.init.uniform_(self.emb.weight, -0.05, 0.05)
        self.user_proj = nn.Linear(n_q * d, k, bias=False)
        self.item_proj = nn.Linear(n_i * d, k, bias=False)
        self.d, self.n_q, self.n_i = d, n_q, n_i

    def user(self, codes):
        return self.user_proj(self.emb(codes).reshape(len(codes), -1))

    def items(self, codes):
        return self.item_proj(self.emb(codes).reshape(len(codes), -1))

    def blocks(self):
        bs = [self.user_proj.weight[:, t * self.d:(t + 1) * self.d]
              for t in range(self.n_q)]
        bs += [self.item_proj.weight[:, t * self.d:(t + 1) * self.d]
               for t in range(self.n_i)]
        return bs

    def proj_overlap(self):
        bs = self.blocks()
        with torch.no_grad():
            vals = [(bs[i] @ bs[j].T).norm().item()
                    / (bs[i].norm().item() * bs[j].norm().item() + 1e-12)
                    for i in range(len(bs)) for j in range(i + 1, len(bs))]
        return (float(np.mean(vals)), float(np.max(vals))) if vals else (0.0, 0.0)

    def ranks(self):
        with torch.no_grad():
            rs = []
            for A in self.blocks():
                s = torch.linalg.svdvals(A.float().cpu())
                rs.append(((s ** 2).sum() / (s[0] ** 2 + 1e-12)).item())
        return rs


def evaluate_gts(model, Q, IC, targets, n_catalog, device, chunk=256) -> dict:
    model.eval()
    users = sorted(targets)
    disc = 1.0 / np.log2(np.arange(2, TOPN + 2))
    idcg = np.concatenate([[0.0], np.cumsum(disc)])
    rec = dict.fromkeys(KS, 0.0)
    ndcg = dict.fromkeys(KS, 0.0)
    hr = dict.fromkeys(KS, 0)
    cover = {k: [] for k in KS}
    with torch.no_grad():
        V = model.items(IC)
        for lo in range(0, len(users), chunk):
            us = users[lo:lo + chunk]
            top = (model.user(Q[us]) @ V.T).topk(TOPN, dim=1).indices
            for k in KS:
                cover[k].append(top[:, :k].reshape(-1))
            top = top.cpu().numpy()
            for r, u in enumerate(us):
                total, pos = targets[u]
                hit = np.isin(top[r], pos)
                for k in KS:
                    rec[k] += hit[:k].sum() / min(total, k)
                    ndcg[k] += (hit[:k] * disc[:k]).sum() / idcg[min(total, k)]
                    hr[k] += bool(hit[:k].any())
    n = len(users)
    out = {f'recall{k}': round(rec[k] / n, 4) for k in KS}
    out.update({f'ndcg{k}': round(ndcg[k] / n, 4) for k in KS})
    out.update({f'hr{k}': round(hr[k] / n, 4) for k in KS})
    out.update({f'coverage{k}':
                round(torch.cat(cover[k]).unique().numel() / n_catalog, 4)
                for k in KS})
    return out


def run_experiment(method, budget, data, device, args, seed=42) -> dict:
    QS, EV, EVAL, IC, n_rows, total_vocab = encode_all(
        data, method, budget, args.base_levels)
    used = torch.tensor(np.unique(np.concatenate(
        [QS.ravel(), IC.ravel()] + [e for e in EV])), device=device)

    torch.manual_seed(seed)
    n_q = QS.shape[1] + len(EV)
    model = TwoTowerFeat(n_rows, EMB_DIM, args.k, n_q, IC.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    QSt = torch.tensor(QS, device=device)
    ICt = torch.tensor(IC, device=device)
    EVt = [torch.tensor(e, device=device) for e in EV]
    ev_u = torch.tensor(data['ev_u'], device=device)
    ev_i = torch.tensor(data['ev_i'], device=device)
    Qe = torch.cat([QSt] + [torch.full((len(QSt), 1), v, dtype=torch.long,
                                       device=device) for v in EVAL], dim=1)

    counts = np.bincount(data['ev_i'], minlength=data['n_catalog']).astype(np.float64)
    logq = torch.tensor(np.log((counts + 1) / (counts.sum() + data['n_catalog'])),
                        device=device, dtype=torch.float32)

    size_mb = n_rows * EMB_DIM * 4 / 1e6
    print(f"[{args.tag} b={budget}] {method}: table {n_rows:,} rows / "
          f"{size_mb:.3f} MB ({n_rows/total_vocab:.3f}x vocab)", flush=True)

    tb, tb_tag = None, f'{args.ts}_gts_{args.tag}/b{budget}'
    if args.tb:
        from torch.utils.tensorboard import SummaryWriter
        name = method + (f'_seed{seed}' if args.runs > 1 else '')
        tb = SummaryWriter(f'{args.tb}/{args.ts}_gts_{args.tag}/b{budget}/{name}')

    n = len(ev_u)
    best, best_state, bad, history = -1.0, None, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        cap = args.batches_per_epoch or (n + args.batch - 1) // args.batch
        tot, nb = 0.0, 0
        for step in range(min(cap, (n + args.batch - 1) // args.batch)):
            idx = perm[step * args.batch:(step + 1) * args.batch]
            qc = torch.cat([QSt[ev_u[idx]]]
                           + [e[idx].unsqueeze(1) for e in EVt], dim=1)
            u = model.user(qc)
            items_b = ev_i[idx]
            if args.loss == 'full':
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
        if args.patience > 0 and bad >= args.patience:
            break

    model.load_state_dict(best_state)
    if tb:
        tb.close()
    test = evaluate_gts(model, Qe, ICt, data['test'], data['n_catalog'], device)
    ov_mean, ov_max = model.proj_overlap()
    rk = model.ranks()
    with torch.no_grad():
        l2 = model.emb.weight[used].norm(dim=1).mean().item()
    res = {'dataset': args.tag, 'method': method, 'budget': budget,
           'rows': n_rows, 'size_mb': round(size_mb, 4),
           'rows_over_vocab': round(n_rows / total_vocab, 4), 'seed': seed,
           **{f'test_{k}': v for k, v in test.items()},
           'best_val_ndcg100': best, 'proj_overlap_mean': round(ov_mean, 4),
           'proj_overlap_max': round(ov_max, 4),
           'ranks': [round(r, 3) for r in rk],
           'rank_sum': round(float(np.sum(rk)), 3), 'rank_budget': EMB_DIM,
           'emb_l2_final': round(l2, 4),
           'epochs_run': len(history), 'history': history}
    print(f"[{args.tag} b={budget}] {method}: test recall@100 {test['recall100']:.4f} "
          f"ndcg@100 {test['ndcg100']:.4f}  recall@10 {test['recall10']:.4f}  "
          f"overlap {ov_mean:.3f}", flush=True)
    return res


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='vklsvd',
                   choices=['vklsvd', 'yambda_50m', 'yambda_500m'])
    p.add_argument('--subset', default='up0.001_ip0.001',
                   help='VK-LSVD subsample dir (e.g. ur0.01_ip0.01, whole)')
    p.add_argument('--positive', default='like', choices=['like', 'watch'],
                   help='VK-LSVD positive: explicit like or timespent>=0.5*duration')
    p.add_argument('--interaction', default='multi',
                   choices=['likes', 'listens', 'multi'],
                   help='Yambda training events (targets are always likes for multi)')
    p.add_argument('--budgets', nargs='+', type=float, default=[1.0, 0.5, 0.1])
    p.add_argument('--base-levels', type=int, default=None)
    p.add_argument('--k',       type=int,   default=32)
    p.add_argument('--batch',   type=int,   default=4096)
    p.add_argument('--lr',      type=float, default=1e-3)
    p.add_argument('--epochs',  type=int,   default=30)
    p.add_argument('--patience', type=int,  default=5)
    p.add_argument('--batches-per-epoch', type=int, default=0,
                   help='cap optimizer steps per epoch (0 = full pass)')
    p.add_argument('--loss',    default='sampled', choices=['full', 'sampled'])
    p.add_argument('--runs',    type=int,   default=1)
    p.add_argument('--seed',    type=int,   default=42)
    p.add_argument('--only',    default=None)
    p.add_argument('--tb',      default='runs', help='TensorBoard dir ("" = off)')
    p.add_argument('--out',     default='experiment_logs')
    args = p.parse_args()
    args.ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    device = get_device()
    if args.dataset == 'vklsvd':
        args.tag = f'vklsvd_{args.subset}_{args.positive}'
        data = load_vklsvd_gts(args.subset, args.positive)
    else:
        args.tag = f'{args.dataset}_{args.interaction}'
        data = load_yambda_gts(args.dataset.split('_')[1], args.interaction)
    feats = [c['name'] for c in data['q_static'] + data['ev_cols'] + data['item_cols']]
    print(f"device={device}  {args.tag}: {data['n_users']:,} users, "
          f"{data['n_catalog']:,} catalog items, {len(data['ev_u']):,} events, "
          f"val/test users {len(data['val']):,}/{len(data['test']):,}\n"
          f"features: {feats}", flush=True)

    results, mx = [], max(args.budgets)
    for b in sorted(args.budgets, reverse=True):
        methods = ['Non-multiplex', 'Multiplex'] + (['Collisionless'] if b == mx else [])
        if args.only:
            exact = [m for m in methods if args.only.lower() == m.lower()]
            methods = exact or [m for m in methods if args.only.lower() in m.lower()]
        for m in methods:
            for r in range(args.runs):
                results.append(run_experiment(m, b, data, device, args,
                                              seed=args.seed + r))

    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / f'{args.ts}_gts_{args.tag}.json').write_text(
        json.dumps({'config': vars(args), 'results': results}, indent=2))
    print(f'\nsaved experiment_logs/{args.ts}_gts_{args.tag}.json')
    print(f"\n{'method':<16}{'budget':>7}{'rows':>10}{'r@10':>8}{'r@100':>8}"
          f"{'n@100':>8}{'cov@100':>9}{'overlap':>9}{'emb_l2':>8}")
    for r in results:
        print(f"{r['method']:<16}{r['budget']:>7}{r['rows']:>10,}"
              f"{r['test_recall10']:>8.4f}{r['test_recall100']:>8.4f}"
              f"{r['test_ndcg100']:>8.4f}{r['test_coverage100']:>9.4f}"
              f"{r['proj_overlap_mean']:>9.3f}{r['emb_l2_final']:>8.3f}")


if __name__ == '__main__':
    main()
