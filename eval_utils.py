import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader


def _to_device(x, xd, y, device):
    return (x.to(device),
            xd.to(device) if xd.shape[1] else None,
            y.to(device))


def evaluate_auc(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, xd, y in loader:
            x, xd, y = _to_device(x, xd, y, device)
            preds.append(torch.sigmoid(model(x, xd)).cpu().numpy())
            targets.append(y.cpu().numpy())
    return roc_auc_score(np.concatenate(targets), np.concatenate(preds))


def evaluate_candgen(model, uc, ic, pairs, seen, device, ks=(10, 20, 50, 100), chunk=512) -> dict:
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


def eval_split(model, users, seqs, targets, extra, max_len, pad, n_items,
               device, ks=(10, 20, 100), chunk=256) -> dict:
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


def evaluate_gts(model, Q, IC, targets, n_catalog, device,
                 ks=(10, 50, 100), topn=100, chunk=256) -> dict:
    model.eval()
    users = sorted(targets)
    disc = 1.0 / np.log2(np.arange(2, topn + 2))
    idcg = np.concatenate([[0.0], np.cumsum(disc)])
    rec = dict.fromkeys(ks, 0.0)
    ndcg = dict.fromkeys(ks, 0.0)
    hr = dict.fromkeys(ks, 0)
    cover = {k: [] for k in ks}
    with torch.no_grad():
        V = model.items(IC)
        for lo in range(0, len(users), chunk):
            us = users[lo:lo + chunk]
            top = (model.user(Q[us]) @ V.T).topk(topn, dim=1).indices
            for k in ks:
                cover[k].append(top[:, :k].reshape(-1))
            top = top.cpu().numpy()
            for r, u in enumerate(us):
                total, pos = targets[u]
                hit = np.isin(top[r], pos)
                for k in ks:
                    rec[k] += hit[:k].sum() / min(total, k)
                    ndcg[k] += (hit[:k] * disc[:k]).sum() / idcg[min(total, k)]
                    hr[k] += bool(hit[:k].any())
    n = len(users)
    out = {f'recall{k}': round(rec[k] / n, 4) for k in ks}
    out.update({f'ndcg{k}': round(ndcg[k] / n, 4) for k in ks})
    out.update({f'hr{k}': round(hr[k] / n, 4) for k in ks})
    out.update({f'coverage{k}':
                round(torch.cat(cover[k]).unique().numel() / n_catalog, 4)
                for k in ks})
    return out
