import datetime
import json
import pathlib
from argparse import ArgumentParser
from dataclasses import replace

import numpy as np
import torch
import torch.nn.functional as F

from dataset_utils import load as load_retrieval
from dataset_utils import load_vklsvd_seq
from embeddings import MultiplexedEmbeddings
from eval_utils import eval_split
from gsasrec import GSASRec
from utils import get_device, load_config


CKPT_DIR = pathlib.Path('checkpoints')


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


def build_model(cfg, n_items, method, budget, seed):
    torch.manual_seed(seed)
    probes = cfg.probes if method == 'Multiplex' else 1
    sizes = {'item_in': n_items + 2, 'position': cfg.max_len}
    if not cfg.tie_io:
        sizes['item_out'] = n_items + 2
    tables = MultiplexedEmbeddings(sizes, cfg.emb_dim, method, budget,
                                   align_roles=cfg.align_roles,
                                   probes=probes, combine=cfg.combine)
    label = method + (' (aligned)' if method == 'Multiplex' and cfg.align_roles
                      and not cfg.tie_io else '')
    if probes > 1:
        label += f' ({probes}-probe {cfg.combine})'
    if cfg.tie_io:
        label += ' (tied)'
    model = GSASRec(num_items=n_items, sequence_length=cfg.max_len,
                    embedding_dim=cfg.emb_dim, num_heads=cfg.heads,
                    num_blocks=cfg.blocks, dropout_rate=cfg.dropout,
                    reuse_item_embeddings=cfg.tie_io, tables=tables)
    return model, tables, label


def run(method, budget, data, cfg, device, seed=42) -> dict:
    users, seqs, train_mat, val, test, pad, n_items = build_sequences(data, cfg.max_len)
    model, tables, label = build_model(cfg, n_items, method, budget, seed)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    st = tables.stats()
    print(f"[{cfg.dataset} b={budget}] {label}: table {st['rows']:,} rows / "
          f"{st['size_mb']} MB ({st['rows_over_vocab']:.3f}x vocab)", flush=True)

    X = torch.as_tensor(train_mat, device=device)
    best, best_state, bad, history = -1.0, None, 0, []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        perm = torch.randperm(len(X), device=device)
        n_steps = min(cfg.batches_per_epoch, (len(X) + cfg.batch - 1) // cfg.batch)
        tot, nb = 0.0, 0
        for step in range(n_steps):
            s = step * cfg.batch
            b = X[perm[s:s + cfg.batch]]
            inp, labels = b[:, :-1], b[:, 1:]
            mask = (inp != pad)
            h, _ = model(inp)
            W = model.get_output_embeddings().weight
            loss = F.cross_entropy(h[mask] @ W.T, labels[mask])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1

        v = eval_split(model, users, seqs, val, None, cfg.max_len, pad,
                       n_items, device)
        history.append({'epoch': epoch, 'loss': round(tot / nb, 4),
                        **{f'val_{k}': x for k, x in v.items()},
                        'emb_l2': round(tables.emb_l2(), 4)})
        print(f"    epoch {epoch:>2}: loss {tot/nb:.4f}  val_ndcg100 {v['ndcg100']:.4f}"
              f"  val_ndcg10 {v['ndcg10']:.4f}  val_hr10 {v['hr10']:.4f}"
              f"  emb_l2 {tables.emb_l2():.3f}", flush=True)
        if v['ndcg100'] > best:
            best, bad = v['ndcg100'], 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
        if cfg.patience > 0 and bad >= cfg.patience:
            break

    model.load_state_dict(best_state)
    CKPT_DIR.mkdir(exist_ok=True)
    slug = label.replace(' ', '').replace('(', '').replace(')', '')
    torch.save({'state_dict': best_state, 'arm': 'sasrec', 'method': method,
                'label': label, 'budget': budget, 'seed': seed,
                'tie_io': cfg.tie_io, 'align_roles': cfg.align_roles,
                'probes': cfg.probes if method == 'Multiplex' else 1,
                'combine': cfg.combine},
               CKPT_DIR / f'sasrec-{cfg.dataset}-{slug}-b{budget}-seed{seed}.pt')
    t = eval_split(model, users, seqs, test, val, cfg.max_len, pad, n_items, device)
    res = {'dataset': cfg.dataset, 'method': label, 'budget': budget,
           'align_roles': cfg.align_roles, 'tie_io': cfg.tie_io,
           'seed': seed, **st, **{f'test_{k}': v for k, v in t.items()},
           'best_val_ndcg100': best, 'emb_l2_final': round(tables.emb_l2(), 4),
           'n_params': sum(p.numel() for p in model.parameters()),
           'epochs_run': len(history), 'history': history}
    print(f"[{cfg.dataset} b={budget}] {label}: test NDCG@10 {t['ndcg10']:.4f} "
          f"HR@10 {t['hr10']:.4f}  NDCG@20 {t['ndcg20']:.4f}", flush=True)
    return res


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    cfg = load_config(parser.parse_args().config)

    device = get_device()
    print(f'device={device}  dataset={cfg.dataset}  budgets={list(cfg.budgets)}', flush=True)
    if cfg.dataset == 'vklsvd':
        data = load_vklsvd_seq(cfg.subset, cfg.positive)
    else:
        data = load_retrieval(cfg.dataset)

    results, mx = [], max(cfg.budgets)
    for b in sorted(cfg.budgets, reverse=True):
        methods = ['Non-multiplex', 'Multiplex'] + (['Collisionless'] if b == mx else [])
        if cfg.only:
            exact = [m for m in methods if cfg.only.lower() == m.lower()]
            methods = exact or [m for m in methods if cfg.only.lower() in m.lower()]
        for m in methods:
            for r in range(cfg.runs):
                results.append(run(m, b, data, cfg, device, seed=cfg.seed + r))

    if cfg.tied_baseline and not cfg.tie_io:
        tied_cfg = replace(cfg, tie_io=True)
        for r in range(cfg.runs):
            res = run('Collisionless', 1.0, data, tied_cfg, device,
                      seed=cfg.seed + r)
            res['note'] = 'tied input/output item embeddings; ~0.5x the untied rows'
            results.append(res)

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out = pathlib.Path(cfg.out); out.mkdir(parents=True, exist_ok=True)
    (out / f'{ts}_sasrec_{cfg.dataset}.json').write_text(
        json.dumps({'config': vars(cfg), 'results': results}, indent=2))
    print(f'\nsaved {cfg.out}/{ts}_sasrec_{cfg.dataset}.json')
    print(f"\n{'method':<22}{'budget':>7}{'rows':>10}{'MB':>8}{'NDCG@10':>9}"
          f"{'HR@10':>8}{'NDCG@20':>9}{'emb_l2':>8}")
    for r in results:
        print(f"{r['method']:<22}{r['budget']:>7}{r['rows']:>10,}{r['size_mb']:>8.2f}"
              f"{r['test_ndcg10']:>9.4f}{r['test_hr10']:>8.4f}{r['test_ndcg20']:>9.4f}"
              f"{r['emb_l2_final']:>8.3f}")


if __name__ == '__main__':
    main()
