import pathlib

import numpy as np


CACHE = pathlib.Path('datasets')

URLS = {
    'beauty':  'https://raw.githubusercontent.com/kang205/SASRec/master/data/Beauty.txt',
    'steam':   'https://raw.githubusercontent.com/kang205/SASRec/master/data/Steam.txt',
    'ml1m':    'https://raw.githubusercontent.com/kang205/SASRec/master/data/ml-1m.txt',
    'gowalla_train': 'https://raw.githubusercontent.com/gusye1234/LightGCN-PyTorch/master/data/gowalla/train.txt',
    'gowalla_test':  'https://raw.githubusercontent.com/gusye1234/LightGCN-PyTorch/master/data/gowalla/test.txt',
    'yambda_50m':  'https://huggingface.co/datasets/yandex/yambda/resolve/main/flat/50m/likes.parquet',
    'yambda_500m': 'https://huggingface.co/datasets/yandex/yambda/resolve/main/flat/500m/likes.parquet',
    'yambda_50m_listens':  'https://huggingface.co/datasets/yandex/yambda/resolve/main/flat/50m/listens.parquet',
    'yambda_50m_multi':    'https://huggingface.co/datasets/yandex/yambda/resolve/main/flat/50m/multi_event.parquet',
    'yambda_500m_listens': 'https://huggingface.co/datasets/yandex/yambda/resolve/main/flat/500m/listens.parquet',
    'yambda_500m_multi':   'https://huggingface.co/datasets/yandex/yambda/resolve/main/flat/500m/multi_event.parquet',
}

VK_BASE = 'https://huggingface.co/datasets/deepvk/VK-LSVD/resolve/main'


def fetch_vklsvd(rel: str) -> pathlib.Path:
    path = CACHE / 'vklsvd' / rel
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        import urllib.request
        print(f'downloading vklsvd/{rel}', flush=True)
        urllib.request.urlretrieve(f'{VK_BASE}/{rel}', path)
    return path

YAMBDA = dict(LAST=26_000_000, DAY=86_400, GAP=1_800)


def _fetch(key: str, ext: str = 'txt') -> pathlib.Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f'{key}.{ext}'
    if not path.exists():
        import urllib.request
        print(f'downloading {key} -> {path}', flush=True)
        urllib.request.urlretrieve(URLS[key], path)
    return path


def _reindex(pairs: np.ndarray):
    users, u_inv = np.unique(pairs[:, 0], return_inverse=True)
    items, i_inv = np.unique(pairs[:, 1], return_inverse=True)
    return np.stack([u_inv, i_inv], axis=1), users, items


def _leave_one_out(pairs: np.ndarray):
    order = np.argsort(pairs[:, 0], kind='stable')
    p = pairs[order]
    u = p[:, 0]
    last = np.r_[u[1:] != u[:-1], True]
    prev = np.r_[last[1:], False]
    prev &= ~last
    train = p[~(last | prev)]
    return train, p[prev], p[last]


def load_sasrec(name: str) -> dict:
    path = _fetch(name)
    pairs = np.loadtxt(path, dtype=np.int64)
    pairs, users, items = _reindex(pairs)
    tr, va, te = _leave_one_out(pairs)
    return {'users': users, 'items': items, 'train': tr, 'val': va, 'test': te}


def _read_lightgcn(path: pathlib.Path) -> np.ndarray:
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            u = int(parts[0])
            for i in parts[1:]:
                rows.append((u, int(i)))
    return np.array(rows, dtype=np.int64)


def load_gowalla(val_frac: float = 0.1, seed: int = 42) -> dict:
    tr_pairs = _read_lightgcn(_fetch('gowalla_train'))
    te_pairs = _read_lightgcn(_fetch('gowalla_test'))
    n_tr = len(tr_pairs)
    allp = np.concatenate([tr_pairs, te_pairs])
    allp, users, items = _reindex(allp)
    tr_all, te = allp[:n_tr], allp[n_tr:]

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(tr_all))
    cut = int(val_frac * len(tr_all))
    va, tr = tr_all[perm[:cut]], tr_all[perm[cut:]]
    return {'users': users, 'items': items, 'train': tr, 'val': va, 'test': te}


def load_yambda(size: str = '50m') -> dict:
    import polars as pl
    df = pl.read_parquet(_fetch(f'yambda_{size}', ext='parquet'))
    last, day, gap = YAMBDA['LAST'], YAMBDA['DAY'], YAMBDA['GAP']
    test_ts  = last - day
    val_lo, val_hi = test_ts - day - gap, test_ts - gap
    train_hi = test_ts - gap - day - gap

    def arr(f):
        return f.select(['uid', 'item_id']).to_numpy().astype(np.int64)

    tr = arr(df.filter(pl.col('timestamp') < train_hi))
    va = arr(df.filter((pl.col('timestamp') >= val_lo) & (pl.col('timestamp') < val_hi)))
    te = arr(df.filter(pl.col('timestamp') >= test_ts))

    users, items = np.unique(tr[:, 0]), np.unique(tr[:, 1])
    u_map = {u: i for i, u in enumerate(users)}
    i_map = {x: i for i, x in enumerate(items)}

    def remap(p):
        keep = np.array([(u in u_map and x in i_map) for u, x in p], dtype=bool) \
               if len(p) else np.zeros(0, dtype=bool)
        p = p[keep]
        return np.stack([[u_map[u] for u in p[:, 0]],
                         [i_map[x] for x in p[:, 1]]], axis=1) if len(p) \
               else np.zeros((0, 2), dtype=np.int64)

    return {'users': users, 'items': items,
            'train': remap(tr), 'val': remap(va), 'test': remap(te)}


def load(name: str) -> dict:
    if name.startswith('yambda'):
        d = load_yambda(name.split('_')[1] if '_' in name else '50m')
    elif name == 'gowalla':
        d = load_gowalla()
    else:
        d = load_sasrec(name)
    print(f"{name}: {len(d['users']):,} users, {len(d['items']):,} items, "
          f"splits {len(d['train']):,}/{len(d['val']):,}/{len(d['test']):,}",
          flush=True)
    return d


if __name__ == '__main__':
    import sys
    for n in (sys.argv[1:] or ['beauty', 'steam', 'gowalla']):
        load(n)
