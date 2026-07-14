import time
import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

from ue       import (UnifiedEmbedding, NonMultiplexedEmbedding, CollisionlessEmbedding,
                      prehash, prehash_split, build_vocabs, preencode,
                      embedding_table_stats, embedding_l2_mean)
from models   import SimpleMLP, DCNV2
from data     import EmbDataset
from train    import train_model, evaluate


# emb_dim / emb_levels / architecture follow the paper (App. B):
# movielens: 1 cross + DNN 192; avazu: 1 cross + DNN 512x2; criteo: 2 cross + DNN 748x2.
# batch/lr: movielens tuned locally (reproduces Table 1, see DATASETS.md);
# avazu/criteo start from the paper's values.
DATASET_CFG = {
    "movielens": {"emb_dim": 30, "emb_levels": 13_653, "cross": 1, "dnn": (192,),
                  "batch": 128, "lr": 1e-3},
    "avazu":     {"emb_dim": 32, "emb_levels": 26_542, "cross": 1, "dnn": (512, 512),
                  "batch": 512, "lr": 2e-4},
    "criteo":    {"emb_dim": 39, "emb_levels": 83_886, "cross": 2, "dnn": (748, 748),
                  "batch": 512, "lr": 2e-4},
}


def scaled_levels(name: str, budget: float) -> int:
    return max(1, round(DATASET_CFG[name]["emb_levels"] * budget))


def table_report(name: str, df: pl.DataFrame, budget: float = 1.0) -> dict:
    cfg        = DATASET_CFG[name]
    emb_dim    = cfg["emb_dim"]
    emb_levels = scaled_levels(name, budget)
    cols       = df.columns
    vocabs     = build_vocabs(df, cols)
    vs         = [len(vocabs[c]) for c in cols]
    total      = sum(vs)

    return {
        "budget":      budget,
        "emb_dim":     emb_dim,
        "emb_levels":  emb_levels,
        "total_vocab": total,
        "vocab_sizes": dict(zip(cols, vs)),
        "tables": {
            "Non-multiplex":  embedding_table_stats(
                NonMultiplexedEmbedding(vs, emb_levels, emb_dim), total),
            "Multiplex":      embedding_table_stats(
                UnifiedEmbedding(emb_levels, emb_dim), total),
            "Collisionless":  embedding_table_stats(
                CollisionlessEmbedding(vs, emb_dim), total),
        },
    }


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


def run_dataset(
    name:       str,
    df:         pl.DataFrame,
    labels:     np.ndarray,
    tr, va, te,
    device,
    batch_size: int   = None,
    lr:         float = None,
    max_epochs: int   = 30,
    patience:   int   = 5,
    weight_decay: float = 1e-5,
    budget:     float = 1.0,
    include_collisionless: bool = True,
    tb_dir:     str   = None,
    num_cross:  int   = None,
    dnn_dims:   tuple = None,
    dropout:    float = 0.0,
    use_bn:     bool  = False,
    only:       str   = None,
    n_runs:     int   = 1,
    seed:       int   = 42,
    eval_test_epochs: bool = False,
    dense:      np.ndarray = None,
    with_mlp:   bool  = False,
    num_workers: int  = 4,
) -> dict:
    cfg        = DATASET_CFG[name]
    emb_dim    = cfg["emb_dim"]
    emb_levels = scaled_levels(name, budget)
    num_cross  = num_cross if num_cross is not None else cfg["cross"]
    dnn_dims   = tuple(dnn_dims) if dnn_dims else cfg["dnn"]
    batch_size = batch_size if batch_size is not None else cfg["batch"]
    lr         = lr if lr is not None else cfg["lr"]
    cols       = df.columns
    dense_dim  = dense.shape[1] if dense is not None else 0
    emb_out    = len(cols) * emb_dim + dense_dim

    vocabs = build_vocabs(df, cols)
    vs     = [len(vocabs[c]) for c in cols]
    total_vocab = sum(vs)

    def dcn(emb):
        return DCNV2(emb, emb_out, num_cross=num_cross, dnn_dims=dnn_dims,
                     dropout=dropout, use_bn=use_bn)

    def mlp(emb):
        return SimpleMLP(emb, emb_out)

    # DCN only by default (the paper has no MLP arm); --with-mlp adds it back
    def make_emb(kind):
        if kind == "nm":   return NonMultiplexedEmbedding(vs, emb_levels, emb_dim)
        if kind == "hash": return UnifiedEmbedding(emb_levels, emb_dim)
        return CollisionlessEmbedding(vs, emb_dim)

    methods = [("Non-multiplex", "nm"), ("Multiplex", "hash")]
    if include_collisionless:
        methods.append(("Collisionless", "cl"))

    specs = []
    for label, kind in methods:
        specs.append((f"{label} + DCN", kind,
                      lambda k=kind: dcn(make_emb(k))))
        if with_mlp:
            specs.append((f"{label} + MLP", kind,
                          lambda k=kind: mlp(make_emb(k))))
    if only:
        exact = [s for s in specs if only.lower() == s[0].lower()]
        specs = exact or [s for s in specs if only.lower() in s[0].lower()]
        if not specs:
            raise ValueError(f"--only '{only}' matched no experiments")

    # encode inputs lazily: only for the embedding kinds actually selected
    needed  = {kind for _, kind, _ in specs}
    loaders = {}
    if "hash" in needed:
        hash_data = np.concatenate(
            [prehash(df[c].to_numpy(), (0,), emb_levels, feature_id=c) for c in cols], axis=1)
        loaders["hash"] = _make_loaders(hash_data, labels, tr, va, te, batch_size, dense, num_workers)
    if "nm" in needed:
        nm_mod = NonMultiplexedEmbedding(vs, emb_levels, emb_dim)
        nm_data = prehash_split(df, cols, nm_mod.levels)
        loaders["nm"] = _make_loaders(nm_data, labels, tr, va, te, batch_size, dense, num_workers)
    if "cl" in needed:
        cl_data = preencode(df, cols, vocabs)
        loaders["cl"] = _make_loaders(cl_data, labels, tr, va, te, batch_size, dense, num_workers)

    results = {}
    for exp_name, kind, make_model in specs:
        tr_l, va_l, te_l = loaders[kind]
        runs, histories = [], []
        table = None
        t0 = time.time()

        for r in range(n_runs):
            torch.manual_seed(seed + r)
            model = make_model()
            if table is None:
                table = embedding_table_stats(model.emb, total_vocab)
                print(f"[{name} b={budget}] {exp_name}: table {table['rows']:,} rows / "
                      f"{table['size_mb']} MB ({table['rows_over_vocab']:.3f}x vocab)",
                      flush=True)

            writer = None
            if tb_dir:
                from torch.utils.tensorboard import SummaryWriter
                suffix = f"_r{r}" if n_runs > 1 else ""
                writer = SummaryWriter(
                    f"{tb_dir}/{name}/b{budget}/{exp_name.replace(' ', '')}{suffix}")

            model, history = train_model(
                model, tr_l, va_l, device,
                lr=lr, max_epochs=max_epochs, patience=patience,
                weight_decay=weight_decay,
                te_loader=te_l if eval_test_epochs else None,
                writer=writer)
            test_auc = evaluate(model, te_l, device)
            if writer is not None:
                writer.add_scalar("auc/test", test_auc, len(history))
                writer.close()

            run = {
                "seed":         seed + r,
                "auc":          round(test_auc, 4),
                "best_val_auc": max(h["val_auc"] for h in history),
                "emb_l2_final": round(embedding_l2_mean(model.emb), 4),
                "epochs_run":   len(history),
            }
            if eval_test_epochs:
                # paper protocol (Tsang & Ahle): best test AUC over epochs
                run["auc_best_epoch"] = max(h["test_auc"] for h in history)
            runs.append(run)
            histories.append(history)
            if n_runs > 1:
                print(f"[{name} b={budget}] {exp_name} run {r}: "
                      f"test AUC {test_auc:.4f}"
                      + (f", best-epoch {run['auc_best_epoch']:.4f}"
                         if eval_test_epochs else ""), flush=True)

        aucs = [r["auc"] for r in runs]
        results[exp_name] = {
            "auc":          round(float(np.mean(aucs)), 4),
            "auc_std":      round(float(np.std(aucs)), 4),
            "auc_runs":     aucs,
            "n_params":     sum(p.numel() for p in model.parameters()),
            "table":        table,
            "time_sec":     int(time.time() - t0),
            "runs":         runs,
            "histories":    histories,
        }
        if eval_test_epochs:
            bests = [r["auc_best_epoch"] for r in runs]
            results[exp_name]["auc_paper_protocol"] = {
                "mean": round(float(np.mean(bests)), 4),
                "std":  round(float(np.std(bests)), 4),
                "max":  round(float(np.max(bests)), 4),
            }
        print(f"[{name} b={budget}] {exp_name}: test AUC {results[exp_name]['auc']:.4f}"
              + (f" ±{results[exp_name]['auc_std']:.4f}" if n_runs > 1 else "")
              + (f", paper-protocol {results[exp_name]['auc_paper_protocol']['mean']:.4f}"
                 if eval_test_epochs else "")
              + f" ({results[exp_name]['time_sec']}s)", flush=True)

    return results
