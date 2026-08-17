import argparse
import datetime
import json
import pathlib

import numpy as np
import retrieval_data
import torch
import torch.nn.functional as F
from run import get_device
from sasrec import GSASRec, MultiplexedEmbeddings


def build_sequences(d: dict, max_len: int):
    n_items = len(d['items'])
    pad = n_items + 1
    seqs = {}
    for u, i in d['train']:
        seqs.setdefault(int(u), []).append(int(i) + 1)
    val = {int(u): int(i) + 1 for u, i in d['val']}
    test = {int(u): int(i) + 1 for u, i in d['test']}

    users = sorted(seqs)
    train_mat = np.full((len(users), max_len + 1), pad, dtype=np.int64)
    for r, u in enumerate(users):
        s = seqs[u][-(max_len + 1):]
        train_mat[r, max_len + 1 - len(s):] = s
    return users, seqs, train_mat, val, test, pad, n_items


def eval_split(model, users, seqs, targets, extra, max_len, pad, n_items,
               device, ks=(10, 20), chunk=256) -> dict:
    model.eval()
    us = [u for u in users if u in targets]
    hits = dict.fromkeys(ks, 0)
    ndcg = dict.fromkeys(ks, 0.0)
    with torch.no_grad():
        W = model.get_output_embeddings().weight
        for lo in range(0, len(us), chunk):
            batch = us[lo:lo + chunk]
            X = np.full((len(batch), max_len), pad, dtype=np.int64)
            for r, u in enumerate(batch):
                s = seqs[u] + ([extra[u]] if extra and u in extra else [])
                s = s[-max_len:]
                X[r, max_len - len(s):] = s
            h, _ = model(torch.as_tensor(X, device=device))
            scores = h[:, -1, :] @ W.T
            scores[:, 0] = float('-inf')
            scores[:, n_items + 1:] = float('-inf')
            for r, u in enumerate(batch):
                seen = seqs[u] + ([extra[u]] if extra and u in extra else [])
                scores[r, torch.as_tensor(seen, device=device)] = float('-inf')
            top = scores.topk(max(ks), dim=1).indices.cpu().numpy()
            for r, u in enumerate(batch):
                pos = np.where(top[r] == targets[u])[0]
                if len(pos) == 0:
                    continue
                rank = int(pos[0])
                for k in ks:
                    if rank < k:
                        hits[k] += 1
                        ndcg[k] += 1.0 / np.log2(rank + 2)
    n = len(us)
    out = {f'hr{k}': round(hits[k] / n, 4) for k in ks}
    out.update({f'ndcg{k}': round(ndcg[k] / n, 4) for k in ks})
    return out


def run(method, budget, data, args, device, seed=42) -> dict:
    users, seqs, train_mat, val, test, pad, n_items = build_sequences(data, args.max_len)
    sizes = {'item_in': n_items + 2, 'position': args.max_len}
    if not args.tie_io:
        sizes['item_out'] = n_items + 2

    torch.manual_seed(seed)
    probes = args.probes if method == 'Multiplex' else 1
    tables = MultiplexedEmbeddings(sizes, args.emb_dim, method, budget,
                                   align_roles=args.align_roles,
                                   probes=probes, combine=args.combine).to(device)
    label = method + (' (aligned)' if method == 'Multiplex' and args.align_roles
                      and not args.tie_io else '')
    if probes > 1:
        label += f' ({probes}-probe {args.combine})'
    if args.tie_io:
        label += ' (tied)'
    model = GSASRec(num_items=n_items, sequence_length=args.max_len,
                    embedding_dim=args.emb_dim, num_heads=args.heads,
                    num_blocks=args.blocks, dropout_rate=args.dropout,
                    reuse_item_embeddings=args.tie_io, tables=tables).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    st = tables.stats()
    print(f"[{args.dataset} b={budget}] {label}: table {st['rows']:,} rows / "
          f"{st['size_mb']} MB ({st['rows_over_vocab']:.3f}x vocab)", flush=True)

    X = torch.as_tensor(train_mat, device=device)
    best, best_state, bad, history = -1.0, None, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(X), device=device)
        n_steps = min(args.batches_per_epoch, (len(X) + args.batch - 1) // args.batch)
        tot, nb = 0.0, 0
        for step in range(n_steps):
            s = step * args.batch
            b = X[perm[s:s + args.batch]]
            inp, labels = b[:, :-1], b[:, 1:]
            mask = (inp != pad)
            h, _ = model(inp)
            W = model.get_output_embeddings().weight
            loss = F.cross_entropy(h[mask] @ W.T, labels[mask])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1

        v = eval_split(model, users, seqs, val, None, args.max_len, pad,
                       n_items, device)
        history.append({'epoch': epoch, 'loss': round(tot / nb, 4),
                        **{f'val_{k}': x for k, x in v.items()},
                        'emb_l2': round(tables.emb_l2(), 4)})
        print(f"    epoch {epoch:>2}: loss {tot/nb:.4f}  val_ndcg10 {v['ndcg10']:.4f}"
              f"  val_hr10 {v['hr10']:.4f}  emb_l2 {tables.emb_l2():.3f}", flush=True)
        if v['ndcg10'] > best:
            best, bad = v['ndcg10'], 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
        if args.patience > 0 and bad >= args.patience:
            break

    model.load_state_dict(best_state)
    t = eval_split(model, users, seqs, test, val, args.max_len, pad, n_items, device)
    res = {'dataset': args.dataset, 'method': label, 'budget': budget,
           'align_roles': args.align_roles, 'tie_io': args.tie_io,
           'seed': seed, **st, **{f'test_{k}': v for k, v in t.items()},
           'best_val_ndcg10': best, 'emb_l2_final': round(tables.emb_l2(), 4),
           'n_params': sum(p.numel() for p in model.parameters()),
           'epochs_run': len(history), 'history': history}
    print(f"[{args.dataset} b={budget}] {label}: test NDCG@10 {t['ndcg10']:.4f} "
          f"HR@10 {t['hr10']:.4f}  NDCG@20 {t['ndcg20']:.4f}", flush=True)
    return res


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='ml1m',
                   choices=['ml1m', 'beauty', 'steam', 'gowalla', 'yambda_50m'])
    p.add_argument('--budgets', nargs='+', type=float, default=[1.0, 0.5, 0.1])
    p.add_argument('--emb-dim', type=int, default=128)
    p.add_argument('--max-len', type=int, default=200)
    p.add_argument('--heads',   type=int, default=1)
    p.add_argument('--blocks',  type=int, default=2)
    p.add_argument('--dropout', type=float, default=0.5)
    p.add_argument('--batch',   type=int, default=128)
    p.add_argument('--batches-per-epoch', type=int, default=100)
    p.add_argument('--lr',      type=float, default=1e-3)
    p.add_argument('--epochs',  type=int, default=300)
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--tie-io',  action='store_true',
                   help="tie input/output item embeddings (SASRec's default is untied)")
    p.add_argument('--no-align-roles', dest='align_roles', action='store_false',
                   help="salt item_in/item_out separately in Multiplex (untied rows) "
                        "instead of the paper's shared hash for shared vocabularies")
    p.add_argument('--probes', type=int, default=1,
                   help='hash lookups per value for Multiplex (same bytes: concat '
                        'halves row width and doubles rows; mean reuses full rows)')
    p.add_argument('--combine', default='concat', choices=['concat', 'mean'])
    p.add_argument('--tied-baseline', action='store_true', default=True,
                   help='also run the tied-collisionless baseline — the structured '
                        '2x compression that hashing must beat at 0.5x')
    p.add_argument('--no-tied-baseline', dest='tied_baseline', action='store_false')
    p.add_argument('--runs',    type=int, default=1)
    p.add_argument('--seed',    type=int, default=42)
    p.add_argument('--only',    default=None)
    p.add_argument('--out',     default='experiment_logs')
    args = p.parse_args()

    device = get_device()
    print(f'device={device}  dataset={args.dataset}  budgets={args.budgets}', flush=True)
    data = retrieval_data.load(args.dataset)

    results, mx = [], max(args.budgets)
    for b in sorted(args.budgets, reverse=True):
        methods = ['Non-multiplex', 'Multiplex'] + (['Collisionless'] if b == mx else [])
        if args.only:
            exact = [m for m in methods if args.only.lower() == m.lower()]
            methods = exact or [m for m in methods if args.only.lower() in m.lower()]
        for m in methods:
            for r in range(args.runs):
                results.append(run(m, b, data, args, device, seed=args.seed + r))

    if args.tied_baseline and not args.tie_io:
        tied_args = argparse.Namespace(**{**vars(args), 'tie_io': True})
        for r in range(args.runs):
            res = run('Collisionless', 1.0, data, tied_args, device,
                      seed=args.seed + r)
            res['note'] = 'tied input/output item embeddings; ~0.5x the untied rows'
            results.append(res)

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / f'{ts}_sasrec_{args.dataset}.json').write_text(
        json.dumps({'config': vars(args), 'results': results}, indent=2))
    print(f'\nsaved experiment_logs/{ts}_sasrec_{args.dataset}.json')
    print(f"\n{'method':<22}{'budget':>7}{'rows':>10}{'MB':>8}{'NDCG@10':>9}"
          f"{'HR@10':>8}{'NDCG@20':>9}{'emb_l2':>8}")
    for r in results:
        print(f"{r['method']:<22}{r['budget']:>7}{r['rows']:>10,}{r['size_mb']:>8.2f}"
              f"{r['test_ndcg10']:>9.4f}{r['test_hr10']:>8.4f}{r['test_ndcg20']:>9.4f}"
              f"{r['emb_l2_final']:>8.3f}")


if __name__ == '__main__':
    main()
