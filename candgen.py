import argparse
import datetime
import json
import pathlib

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from data import load_movielens
from run import get_device
from torch import nn
from ue import prehash


EMB_DIM = 30
ML_USER_COLS = ['user_id', 'gender', 'age', 'occupation', 'zip']


def build_data(args) -> dict:
    if args.dataset == 'movielens':
        df, _, labels, _, _, _ = load_movielens(args.ml1m)
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
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(len(pairs))
        t, v = int(0.8 * len(pairs)), int(0.9 * len(pairs))
        tr, va, te = pairs[perm[:t]], pairs[perm[t:v]], pairs[perm[v:]]
        n_users, n_items = len(users), len(movies)
        print(f'movielens: {len(pairs):,} positive pairs, {n_users:,} users, '
              f'{n_items:,} movies; splits {len(tr):,}/{len(va):,}/{len(te):,}',
              flush=True)
    else:
        import retrieval_data
        d = retrieval_data.load(args.dataset)
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


def encode_tables(data: dict, method: str, budget: float, base_levels: int):
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
            codes[c] = prehash(v, (0,), levels, feature_id=c)[:, 0]
        elif method == 'Non-multiplex':
            lev = max(1, round(sizes[c] / total * levels))
            h = prehash(v, (0,), lev, feature_id=c)[:, 0]
            codes[c] = h + offset
            offset += lev
        else:
            lut = {x: i for i, x in enumerate(vocabs[c])}
            codes[c] = np.array([lut[x] for x in v], dtype=np.int64) + offset
            offset += sizes[c]
    n_rows = levels if method == 'Multiplex' else offset

    user_codes = np.stack([codes[c] for c in data['user_cols']], axis=1)
    return user_codes.astype(np.int64), codes['item_id'].astype(np.int64), n_rows, total


class TwoTower(nn.Module):
    def __init__(self, n_rows, d, k, n_user_feats, worst_init=False, seed=42,
                 score='dot', temp=0.05) -> None:
        super().__init__()
        self.emb = nn.Embedding(n_rows, d)
        nn.init.uniform_(self.emb.weight, -0.05, 0.05)
        self.user_proj = nn.Linear(n_user_feats * d, k, bias=False)
        self.item_proj = nn.Linear(d, k, bias=False)
        self.d, self.n_user_feats = d, n_user_feats
        self.score, self.temp = score, temp
        if worst_init:
            g = torch.Generator().manual_seed(seed)
            W0 = torch.randn(k, d, generator=g)
            W0 = W0 / W0.norm() * self.item_proj.weight.norm()
            with torch.no_grad():
                for t in range(n_user_feats):
                    self.user_proj.weight[:, t * d:(t + 1) * d] = W0
                self.item_proj.weight.copy_(W0)

    def blocks(self):
        W = self.user_proj.weight
        bs = [W[:, t * self.d:(t + 1) * self.d] for t in range(self.n_user_feats)]
        return bs + [self.item_proj.weight]

    def user(self, codes):
        e = self.emb(codes).reshape(len(codes), -1)
        u = self.user_proj(e)
        return F.normalize(u, dim=1) / self.temp if self.score == 'cosine' else u

    def items(self, codes):
        v = self.item_proj(self.emb(codes))
        return F.normalize(v, dim=1) if self.score == 'cosine' else v

    def proj_overlap(self):
        bs = self.blocks()
        with torch.no_grad():
            vals = [ (bs[i] @ bs[j].T).norm().item()
                     / (bs[i].norm().item() * bs[j].norm().item() + 1e-12)
                     for i in range(len(bs)) for j in range(i + 1, len(bs)) ]
        return (float(np.mean(vals)), float(np.max(vals))) if vals else (0.0, 0.0)

    def ranks(self):
        with torch.no_grad():
            rs = []
            for A in self.blocks():
                s = torch.linalg.svdvals(A.float().cpu())
                rs.append(((s ** 2).sum() / (s[0] ** 2 + 1e-12)).item())
        return rs


def evaluate(model, uc, ic, pairs, seen, device, ks=(10, 20, 50, 100), chunk=512) -> dict:
    model.eval()
    seen_pairs, seen_offs = seen
    max_k = max(ks)
    hits = dict.fromkeys(ks, 0)
    ndcg = dict.fromkeys(ks, 0.0)
    by_user = {}
    for u, i in pairs:
        by_user.setdefault(int(u), []).append(int(i))

    with torch.no_grad():
        V = model.items(ic)
        for lo in range(0, len(uc), chunk):
            hi = min(lo + chunk, len(uc))
            targets = [u for u in range(lo, hi) if u in by_user]
            if not targets:
                continue
            scores = model.user(uc[lo:hi]) @ V.T
            s, e = seen_offs[lo], seen_offs[hi]
            if e > s:
                sp = seen_pairs[s:e]
                scores[torch.as_tensor(sp[:, 0] - lo, device=device),
                       torch.as_tensor(sp[:, 1], device=device)] = float('-inf')
            top = scores.topk(max_k, dim=1).indices.cpu().numpy()
            for u in targets:
                row = top[u - lo]
                for i in by_user[u]:
                    pos = np.where(row == i)[0]
                    if len(pos) == 0:
                        continue
                    r = int(pos[0])
                    for k in ks:
                        if r < k:
                            hits[k] += 1
                            ndcg[k] += 1.0 / np.log2(r + 2)
    n = len(pairs)
    out = {f'hr{k}': round(hits[k] / n, 4) for k in ks}
    out.update({f'ndcg{k}': round(ndcg[k] / n, 4) for k in ks})
    return out


def emb_l2_used(model, used) -> float:
    with torch.no_grad():
        return model.emb.weight[used].norm(dim=1).mean().item()


def run_experiment(method, budget, data, device, args, seed=42) -> dict:
    user_codes, item_codes, n_rows, total_vocab = encode_tables(
        data, method, budget, args.base_levels or data['total_vocab'])
    used = torch.tensor(np.unique(np.concatenate(
        [user_codes.ravel(), item_codes])), device=device)

    torch.manual_seed(seed)
    model = TwoTower(n_rows, EMB_DIM, args.k, len(data['user_cols']),
                     worst_init=args.worst_init, seed=seed,
                     score=args.score, temp=args.temp).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    uc = torch.tensor(user_codes, device=device)
    ic = torch.tensor(item_codes, device=device)
    tr = data['train']
    tr_u = torch.tensor(tr[:, 0], device=device)
    tr_i = torch.tensor(tr[:, 1], device=device)

    counts = np.bincount(tr[:, 1], minlength=data['n_items']).astype(np.float64)
    logq = torch.tensor(np.log((counts + 1) / (counts.sum() + data['n_items'])),
                        device=device, dtype=torch.float32)

    size_mb = n_rows * EMB_DIM * 4 / 1e6
    print(f'[{args.dataset} b={budget}] {method}: table {n_rows:,} rows / '
          f'{size_mb:.3f} MB ({n_rows/total_vocab:.3f}x vocab)', flush=True)

    best, best_state, no_improve, history = -1.0, None, 0, []
    n = len(tr_u)
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        total_loss, n_batches = 0.0, 0
        for s in range(0, n, args.batch):
            idx = perm[s:s + args.batch]
            u = model.user(uc[tr_u[idx]])
            if args.loss == 'full':
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

        val = evaluate(model, uc, ic, data['val'], data['seen_tr'], device)
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
        if args.patience > 0 and no_improve >= args.patience:
            break

    model.load_state_dict(best_state)
    test = evaluate(model, uc, ic, data['test'], data['seen_trval'], device)
    ov_mean, ov_max = model.proj_overlap()
    rk = model.ranks()
    res = {'dataset': args.dataset, 'method': method, 'budget': budget,
           'rows': n_rows, 'size_mb': round(size_mb, 4),
           'rows_over_vocab': round(n_rows / total_vocab, 4), 'seed': seed,
           **{f'test_{k}': v for k, v in test.items()},
           'best_val_hr10': best, 'proj_overlap_mean': round(ov_mean, 4),
           'proj_overlap_max': round(ov_max, 4),
           'ranks': [round(r, 3) for r in rk],
           'rank_sum': round(float(np.sum(rk)), 3), 'rank_budget': EMB_DIM,
           'emb_l2_final': round(emb_l2_used(model, used), 4),
           'epochs_run': len(history), 'history': history}
    print(f"[{args.dataset} b={budget}] {method}: test HR@10 {test['hr10']:.4f} "
          f"NDCG@10 {test['ndcg10']:.4f}  overlap {ov_mean:.3f}  "
          f"rank_sum {res['rank_sum']:.1f}/{EMB_DIM}", flush=True)
    return res


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='movielens',
                   choices=['movielens', 'beauty', 'steam', 'gowalla',
                            'yambda_50m', 'yambda_500m'])
    p.add_argument('--ml1m',    default='ml-1m')
    p.add_argument('--budgets', nargs='+', type=float, default=[1.0, 0.5, 0.1])
    p.add_argument('--base-levels', type=int, default=None,
                   help='rows at budget 1.0 (default: total vocabulary)')
    p.add_argument('--k',       type=int,   default=32)
    p.add_argument('--batch',   type=int,   default=1024)
    p.add_argument('--lr',      type=float, default=1e-3)
    p.add_argument('--epochs',  type=int,   default=20)
    p.add_argument('--patience', type=int,  default=3)
    p.add_argument('--runs',    type=int,   default=1)
    p.add_argument('--seed',    type=int,   default=42)
    p.add_argument('--loss',    default='full', choices=['full', 'sampled'])
    p.add_argument('--score',   default='dot', choices=['dot', 'cosine'])
    p.add_argument('--temp',    type=float, default=0.05)
    p.add_argument('--only',    default=None)
    p.add_argument('--worst-init', action='store_true')
    p.add_argument('--out',     default='experiment_logs')
    args = p.parse_args()

    device = get_device()
    print(f'device={device}  dataset={args.dataset}  budgets={args.budgets}', flush=True)
    data = build_data(args)
    cols = data['user_cols'] + ['item_id']
    vals = {**data['user_vals'], 'item_id': data['item_vals']}
    data['total_vocab'] = sum(len(np.unique(vals[c])) for c in cols)

    results = []
    max_budget = max(args.budgets)
    for b in sorted(args.budgets, reverse=True):
        methods = ['Non-multiplex', 'Multiplex']
        if b == max_budget:
            methods.append('Collisionless')
        if args.only:
            exact = [m for m in methods if args.only.lower() == m.lower()]
            methods = exact or [m for m in methods if args.only.lower() in m.lower()]
        for method in methods:
            for r in range(args.runs):
                results.append(run_experiment(method, b, data, device, args,
                                              seed=args.seed + r))

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / f'{ts}_candgen_{args.dataset}.json').write_text(
        json.dumps({'config': vars(args), 'results': results}, indent=2))
    print(f'\nsaved experiment_logs/{ts}_candgen_{args.dataset}.json')
    print(f"\n{'method':<16}{'budget':>7}{'rows':>9}{'HR@10':>8}{'NDCG@10':>9}"
          f"{'overlap':>9}{'ranks':>8}{'emb_l2':>8}")
    for r in results:
        print(f"{r['method']:<16}{r['budget']:>7}{r['rows']:>9,}"
              f"{r['test_hr10']:>8.4f}{r['test_ndcg10']:>9.4f}"
              f"{r['proj_overlap_mean']:>9.3f}{r['rank_sum']:>8.1f}"
              f"{r['emb_l2_final']:>8.3f}")


if __name__ == '__main__':
    main()
