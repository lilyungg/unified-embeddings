from argparse import ArgumentParser

import torch

from utils import get_device, load_config


def eval_ranking(cfg, ck, device):
    from embeddings import (CollisionlessEmbedding, NonMultiplexedEmbedding,
                            UnifiedEmbedding, build_vocabs)
    from eval_utils import evaluate_auc
    from models import DCNV2, SimpleMLP
    from train_ranking import _make_loaders, encode, load_data, scaled_levels

    df, dense, labels, tr, va, te = load_data(cfg)
    vocabs = build_vocabs(df, df.columns)
    vs = [len(vocabs[c]) for c in df.columns]
    emb_levels = scaled_levels(cfg, ck['budget'])
    kind = ck['kind']

    if kind == 'nm':
        emb = NonMultiplexedEmbedding(vs, emb_levels, cfg.emb_dim)
        data = encode(cfg, df, vocabs, 'nm', emb_levels, nm_levels=emb.levels)
    elif kind == 'hash':
        emb = UnifiedEmbedding(emb_levels, cfg.emb_dim, ck.get('probes', 1))
        data = encode(cfg, df, vocabs, 'hash', emb_levels)
    else:
        emb = CollisionlessEmbedding(vs, cfg.emb_dim)
        data = encode(cfg, df, vocabs, 'cl', emb_levels)

    emb_out = len(df.columns) * cfg.emb_dim + (dense.shape[1] if dense is not None else 0)
    if ck['exp'].endswith('MLP'):
        model = SimpleMLP(emb, emb_out)
    else:
        model = DCNV2(emb, emb_out, num_cross=cfg.cross, dnn_dims=tuple(cfg.dnn),
                      dropout=cfg.dropout, use_bn=cfg.bn)
    model.load_state_dict(ck['state_dict'])
    model = model.to(device)
    _, _, te_l = _make_loaders(data, labels, tr, va, te, cfg.batch, dense, cfg.workers)
    return {'test_auc': round(evaluate_auc(model, te_l, device), 4)}


def eval_candgen(cfg, ck, device):
    import numpy as np
    from eval_utils import evaluate_candgen
    from train_candgen import build_data, build_model

    data = build_data(cfg)
    cols = data['user_cols'] + ['item_id']
    vals = {**data['user_vals'], 'item_id': data['item_vals']}
    data['total_vocab'] = sum(len(np.unique(vals[c])) for c in cols)
    model, user_codes, item_codes, *_ = build_model(
        cfg, data, ck['method'], ck['budget'], ck['seed'])
    model.load_state_dict(ck['state_dict'])
    model = model.to(device)
    uc = torch.tensor(user_codes, device=device)
    ic = torch.tensor(item_codes, device=device)
    return evaluate_candgen(model, uc, ic, data['test'], data['seen_trval'], device)


def eval_sasrec(cfg, ck, device):
    from dataclasses import replace

    from dataset_utils import load as load_retrieval
    from dataset_utils import load_vklsvd_seq
    from eval_utils import eval_split
    from train_sasrec import build_model, build_sequences

    if ck['tie_io'] and not cfg.tie_io:
        cfg = replace(cfg, tie_io=True)
    if cfg.dataset == 'vklsvd':
        data = load_vklsvd_seq(cfg.subset, cfg.positive)
    else:
        data = load_retrieval(cfg.dataset)
    users, seqs, _, val, test, pad, n_items = build_sequences(data, cfg.max_len)
    model, tables, label = build_model(cfg, n_items, ck['method'], ck['budget'],
                                       ck['seed'])
    model.load_state_dict(ck['state_dict'])
    model = model.to(device)
    return {'label': label,
            **eval_split(model, users, seqs, test, val, cfg.max_len, pad,
                         n_items, device)}


def eval_candgen_gts(cfg, ck, device):
    import numpy as np

    from dataset_utils import load_vklsvd_gts, load_yambda_gts
    from eval_utils import evaluate_gts
    from models import TwoTowerFeat
    from train_candgen_gts import encode_all

    if cfg.dataset == 'vklsvd':
        data = load_vklsvd_gts(cfg.subset, cfg.positive)
    else:
        data = load_yambda_gts(cfg.dataset.split('_')[1], cfg.interaction)
    probes = ck.get('probes', 1)
    QS, EV, EVAL, IC, n_rows, _ = encode_all(
        data, ck['method'], ck['budget'], cfg.base_levels, probes)
    n_q = len(data['q_static']) + len(data['ev_cols'])
    model = TwoTowerFeat(n_rows, cfg.emb_dim, cfg.k, n_q, len(data['item_cols']),
                         probes=probes)
    model.load_state_dict(ck['state_dict'])
    model = model.to(device)
    QSt = torch.tensor(QS, device=device)
    ICt = torch.tensor(IC, device=device)
    Qe = torch.cat([QSt] + [torch.as_tensor(np.tile(v, (len(QSt), 1)),
                                            device=device) for v in EVAL], dim=1)
    return evaluate_gts(model, Qe, ICt, data['test'], data['n_catalog'], device)


ARMS = {'ranking': eval_ranking, 'candgen': eval_candgen,
        'sasrec': eval_sasrec, 'candgen_gts': eval_candgen_gts}


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    device = get_device()
    print(f"arm={ck['arm']}  method={ck.get('label', ck.get('method', ck.get('exp')))}"
          f"  budget={ck['budget']}  seed={ck['seed']}", flush=True)
    result = ARMS[ck['arm']](cfg, ck, device)
    print(result)


if __name__ == '__main__':
    main()
