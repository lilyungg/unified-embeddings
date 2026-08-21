import copy
import datetime
import json
import pathlib
import time
from argparse import ArgumentParser

import numpy as np
import polars as pl
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from dataset_utils import EmbDataset, load_avazu, load_criteo, load_movielens
from embeddings import (
    CollisionlessEmbedding,
    NonMultiplexedEmbedding,
    UnifiedEmbedding,
    build_vocabs,
    embedding_l2_mean,
    embedding_table_stats,
    preencode,
    prehash,
    prehash_split,
)
from eval_utils import _to_device, evaluate_auc
from models import DCNV2, SimpleMLP
from utils import get_device, load_config


CKPT_DIR = pathlib.Path('checkpoints')


def scaled_levels(cfg, budget: float) -> int:
    return max(1, round(cfg.emb_levels * budget))


def table_report(cfg, df: pl.DataFrame, budget: float = 1.0) -> dict:
    emb_dim    = cfg.emb_dim
    emb_levels = scaled_levels(cfg, budget)
    cols       = df.columns
    vocabs     = build_vocabs(df, cols)
    vs         = [len(vocabs[c]) for c in cols]
    total      = sum(vs)

    return {
        'budget':      budget,
        'emb_dim':     emb_dim,
        'emb_levels':  emb_levels,
        'total_vocab': total,
        'vocab_sizes': dict(zip(cols, vs)),
        'tables': {
            'Non-multiplex':  embedding_table_stats(
                NonMultiplexedEmbedding(vs, emb_levels, emb_dim), total),
            'Multiplex':      embedding_table_stats(
                UnifiedEmbedding(emb_levels, emb_dim, cfg.probes), total),
            'Collisionless':  embedding_table_stats(
                CollisionlessEmbedding(vs, emb_dim), total),
        },
    }


def train_model(
    model:      nn.Module,
    tr_loader:  DataLoader,
    va_loader:  DataLoader,
    device:     torch.device,
    lr:         float = 1e-3,
    max_epochs: int   = 30,
    patience:   int   = 5,
    weight_decay: float = 1e-5,
    te_loader:  DataLoader = None,
    writer            = None,
    tb_tag:     str   = '',
) -> tuple:
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    has_reg   = hasattr(model, 'reg_loss')

    best_auc, best_state, no_improve = -1.0, None, 0
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        for x, xd, y in tr_loader:
            x, xd, y = _to_device(x, xd, y, device)
            optimizer.zero_grad()
            loss = criterion(model(x, xd), y)
            if has_reg:
                loss = loss + model.reg_loss()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        train_loss = total_loss / max(n_batches, 1)
        val_auc    = evaluate_auc(model, va_loader, device)
        emb_l2     = embedding_l2_mean(model.emb) if hasattr(model, 'emb') else 0.0

        row = {
            'epoch':      epoch,
            'train_loss': round(train_loss, 5),
            'val_auc':    round(val_auc, 5),
            'emb_l2_mean': round(emb_l2, 5),
        }
        test_str = ''
        if te_loader is not None:
            row['test_auc'] = round(evaluate_auc(model, te_loader, device), 5)
            test_str = f"  test_auc {row['test_auc']:.4f}"
        history.append(row)
        if writer is not None:
            sfx = f'/{tb_tag}' if tb_tag else ''
            writer.add_scalar(f'auc_val{sfx}',     val_auc,    epoch)
            if te_loader is not None:
                writer.add_scalar(f'auc_test{sfx}', row['test_auc'], epoch)
        print(f'    epoch {epoch:>2}: loss {train_loss:.4f}  val_auc {val_auc:.4f}'
              f'{test_str}  emb_l2 {emb_l2:.3f}', flush=True)

        if val_auc > best_auc:
            best_auc   = val_auc
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if patience > 0 and no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def _make_loaders(data: np.ndarray, labels: np.ndarray,
                  tr, va, te, batch_size: int, dense: np.ndarray = None,
                  num_workers: int = 4) -> tuple:
    kw = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available(),
              persistent_workers=num_workers > 0)
    def ds(idx):
        return EmbDataset(data[idx], labels[idx],
                          dense[idx] if dense is not None else None)
    return (
        DataLoader(ds(tr), batch_size=batch_size, shuffle=True,  **kw),
        DataLoader(ds(va), batch_size=batch_size, shuffle=False, **kw),
        DataLoader(ds(te), batch_size=batch_size, shuffle=False, **kw),
    )


def encode(cfg, df, vocabs, kind: str, emb_levels: int, nm_levels=None) -> np.ndarray:
    cols = df.columns
    if kind == 'hash':
        return np.concatenate(
            [prehash(df[c].to_numpy(), tuple(range(cfg.probes)),
                     emb_levels * cfg.probes, feature_id=c) for c in cols], axis=1)
    if kind == 'nm':
        return prehash_split(df, cols, nm_levels)
    return preencode(df, cols, vocabs)


def run_dataset(cfg, df, labels, tr, va, te, device, budget: float,
                include_collisionless: bool, tb_dir: str = None,
                dense=None) -> dict:
    name       = cfg.dataset
    emb_dim    = cfg.emb_dim
    emb_levels = scaled_levels(cfg, budget)
    dnn_dims   = tuple(cfg.dnn)
    cols       = df.columns
    dense_dim  = dense.shape[1] if dense is not None else 0
    emb_out    = len(cols) * emb_dim + dense_dim

    vocabs = build_vocabs(df, cols)
    vs     = [len(vocabs[c]) for c in cols]
    total_vocab = sum(vs)

    def dcn(emb):
        return DCNV2(emb, emb_out, num_cross=cfg.cross, dnn_dims=dnn_dims,
                     dropout=cfg.dropout, use_bn=cfg.bn)

    def mlp(emb):
        return SimpleMLP(emb, emb_out)

    def make_emb(kind):
        if kind == 'nm':   return NonMultiplexedEmbedding(vs, emb_levels, emb_dim)
        if kind == 'hash': return UnifiedEmbedding(emb_levels, emb_dim, cfg.probes)
        return CollisionlessEmbedding(vs, emb_dim)

    methods = [('Non-multiplex', 'nm'), ('Multiplex', 'hash')]
    if include_collisionless:
        methods.append(('Collisionless', 'cl'))

    specs = []
    for label, kind in methods:
        specs.append((f'{label} + DCN', kind,
                      lambda k=kind: dcn(make_emb(k))))
        if cfg.with_mlp:
            specs.append((f'{label} + MLP', kind,
                          lambda k=kind: mlp(make_emb(k))))
    if cfg.only:
        exact = [s for s in specs if cfg.only.lower() == s[0].lower()]
        specs = exact or [s for s in specs if cfg.only.lower() in s[0].lower()]
        if not specs:
            raise ValueError(f"only='{cfg.only}' matched no experiments")

    needed  = {kind for _, kind, _ in specs}
    loaders = {}
    if 'hash' in needed:
        hash_data = encode(cfg, df, vocabs, 'hash', emb_levels)
        loaders['hash'] = _make_loaders(hash_data, labels, tr, va, te, cfg.batch, dense, cfg.workers)
    if 'nm' in needed:
        nm_mod = NonMultiplexedEmbedding(vs, emb_levels, emb_dim)
        nm_data = encode(cfg, df, vocabs, 'nm', emb_levels, nm_levels=nm_mod.levels)
        loaders['nm'] = _make_loaders(nm_data, labels, tr, va, te, cfg.batch, dense, cfg.workers)
    if 'cl' in needed:
        cl_data = encode(cfg, df, vocabs, 'cl', emb_levels)
        loaders['cl'] = _make_loaders(cl_data, labels, tr, va, te, cfg.batch, dense, cfg.workers)

    results = {}
    for exp_name, kind, make_model in specs:
        tr_l, va_l, te_l = loaders[kind]
        runs, histories = [], []
        table = None
        t0 = time.time()

        for r in range(cfg.runs):
            torch.manual_seed(cfg.seed + r)
            model = make_model()
            if table is None:
                table = embedding_table_stats(model.emb, total_vocab)
                print(f"[{name} b={budget}] {exp_name}: table {table['rows']:,} rows / "
                      f"{table['size_mb']} MB ({table['rows_over_vocab']:.3f}x vocab)",
                      flush=True)

            writer = None
            if cfg.tb:
                from torch.utils.tensorboard import SummaryWriter
                suffix = f'_r{r}' if cfg.runs > 1 else ''
                writer = SummaryWriter(
                    f"{tb_dir}/{name}/b{budget}/{exp_name.replace(' ', '')}{suffix}")

            model, history = train_model(
                model, tr_l, va_l, device,
                lr=cfg.lr, max_epochs=cfg.epochs, patience=cfg.patience,
                weight_decay=cfg.wd,
                te_loader=te_l if cfg.paper_protocol else None,
                writer=writer, tb_tag=f'{name}/b{budget}')
            test_auc = evaluate_auc(model, te_l, device)
            if writer is not None:
                writer.close()

            CKPT_DIR.mkdir(exist_ok=True)
            slug = exp_name.replace(' + ', '_').replace(' ', '')
            torch.save({'state_dict': model.state_dict(), 'arm': 'ranking',
                        'exp': exp_name, 'kind': kind, 'budget': budget,
                        'probes': cfg.probes if kind == 'hash' else 1,
                        'seed': cfg.seed + r},
                       CKPT_DIR / f'ranking-{name}-{slug}-b{budget}-seed{cfg.seed + r}.pt')

            run = {
                'seed':         cfg.seed + r,
                'auc':          round(test_auc, 4),
                'best_val_auc': max(h['val_auc'] for h in history),
                'emb_l2_final': round(embedding_l2_mean(model.emb), 4),
                'epochs_run':   len(history),
            }
            if cfg.paper_protocol:
                run['auc_best_epoch'] = max(h['test_auc'] for h in history)
            runs.append(run)
            histories.append(history)
            if cfg.runs > 1:
                print(f'[{name} b={budget}] {exp_name} run {r}: '
                      f'test AUC {test_auc:.4f}'
                      + (f", best-epoch {run['auc_best_epoch']:.4f}"
                         if cfg.paper_protocol else ''), flush=True)

        aucs = [r['auc'] for r in runs]
        results[exp_name] = {
            'auc':          round(float(np.mean(aucs)), 4),
            'auc_std':      round(float(np.std(aucs)), 4),
            'auc_runs':     aucs,
            'n_params':     sum(p.numel() for p in model.parameters()),
            'table':        table,
            'time_sec':     int(time.time() - t0),
            'runs':         runs,
            'histories':    histories,
        }
        if cfg.paper_protocol:
            bests = [r['auc_best_epoch'] for r in runs]
            results[exp_name]['auc_paper_protocol'] = {
                'mean': round(float(np.mean(bests)), 4),
                'std':  round(float(np.std(bests)), 4),
                'max':  round(float(np.max(bests)), 4),
            }
        print(f"[{name} b={budget}] {exp_name}: test AUC {results[exp_name]['auc']:.4f}"
              + (f" ±{results[exp_name]['auc_std']:.4f}" if cfg.runs > 1 else '')
              + (f", paper-protocol {results[exp_name]['auc_paper_protocol']['mean']:.4f}"
                 if cfg.paper_protocol else '')
              + f" ({results[exp_name]['time_sec']}s)", flush=True)

    return results


def load_data(cfg):
    n_rows = 1_000_000 if cfg.fast else None
    if cfg.dataset == 'movielens':
        return load_movielens(cfg.ml1m, label_mode=cfg.ml_labels)
    if cfg.dataset == 'avazu':
        return load_avazu(path=cfg.avazu, n_rows=n_rows)
    return load_criteo(n_rows=n_rows, path=cfg.criteo)


def print_table_report(name: str, report: dict) -> None:
    print(f"\n{name}: total_vocab={report['total_vocab']:,}  d={report['emb_dim']}  "
          f"budget={report['budget']}  emb_levels={report['emb_levels']:,}")
    for method, t in report['tables'].items():
        print(f"  {method:<14} {t['rows']:>9,} rows  {t['size_mb']:>9.4f} MB  "
              f"{t['rows_over_vocab']:>6.3f}x vocab")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--dry-run', action='store_true',
                        help='print embedding table sizes per budget and exit')
    cli = parser.parse_args()
    cfg = load_config(cli.config)

    device = get_device()
    run_ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = pathlib.Path(cfg.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, dense, labels, tr, va, te = load_data(cfg)

    if cli.dry_run:
        for b in cfg.budgets:
            print_table_report(cfg.dataset, table_report(cfg, df, budget=b))
        return

    print(f'device={device}  dataset={cfg.dataset}  budgets={list(cfg.budgets)}  '
          f'run={run_ts}', flush=True)

    max_budget = max(cfg.budgets)
    by_budget = {}
    for b in cfg.budgets:
        by_budget[str(b)] = run_dataset(
            cfg, df, labels, tr, va, te, device, budget=b,
            include_collisionless=(b == max_budget),
            tb_dir=f'{cfg.tb}/{run_ts}' if cfg.tb else None,
            dense=dense)

    dump = dict(vars(cfg))
    (out_dir / f'{run_ts}_{cfg.dataset}.json').write_text(
        json.dumps({'config': dump, 'budgets': by_budget}, indent=2))
    print(f'\nsaved {cfg.out}/{run_ts}_{cfg.dataset}.json')


if __name__ == '__main__':
    main()
