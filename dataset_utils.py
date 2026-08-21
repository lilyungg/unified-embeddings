import pathlib

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset


CACHE_DIR = pathlib.Path('datasets')


class EmbDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, dense: np.ndarray = None) -> None:
        self.x = torch.from_numpy(x).long()
        self.y = torch.from_numpy(y).float()
        self.dense = (torch.from_numpy(dense).float() if dense is not None
                      else torch.empty((len(self.x), 0)))

    def __len__(self) -> int: return len(self.x)

    def __getitem__(self, i): return self.x[i], self.dense[i], self.y[i]


def random_split(n: int, train: float = 0.8, val: float = 0.1, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    t = int(train * n)
    v = int((train + val) * n)
    return idx[:t], idx[t:v], idx[v:]


def temporal_split(n: int, train: float = 6 / 7, seed: int = 42):
    cut  = int(train * n)
    rng  = np.random.default_rng(seed)
    tail = rng.permutation(np.arange(cut, n))
    half = len(tail) // 2
    return np.arange(cut), tail[:half], tail[half:]


def load_movielens(path: str, label_mode: str = 'wang') -> tuple:
    path = pathlib.Path(path)
    user_info = {}
    with open(path / 'users.dat') as f:
        for line in f:
            uid, gender, age, occ, zip_ = line.strip().split('::')
            user_info[uid] = (gender, age, occ, zip_)
    rows = []
    with open(path / 'ratings.dat') as f:
        for line in f:
            uid, mid, rating, _ = line.strip().split('::')
            rating = int(rating)
            if label_mode == 'wang':
                if rating == 3:
                    continue
                label = int(rating >= 4)
            else:
                label = int(rating >= 3)
            gender, age, occ, zip_ = user_info[uid]
            rows.append({
                'user_id': uid, 'movie_id': mid,
                'gender': gender, 'age': age, 'occupation': occ, 'zip': zip_,
                'label': label,
            })
    df     = pl.DataFrame(rows)
    labels = df['label'].to_numpy().astype(np.float32)
    cols   = ['user_id', 'movie_id', 'gender', 'age', 'occupation', 'zip']
    tr, va, te = random_split(len(df))
    return df.select(cols), None, labels, tr, va, te


def load_avazu(path: str = None, n_rows: int = None) -> tuple:
    prepared = CACHE_DIR / 'avazu_prepared.parquet'
    if path and path.endswith('.parquet'):
        prepared = pathlib.Path(path)
    if prepared.exists():
        df = pl.read_parquet(prepared)
        if n_rows:
            df = df.sample(n=n_rows, seed=42)
        labels = df['label'].to_numpy()
        df     = df.drop('label')
        tr, va, te = random_split(len(df), train=0.81, val=0.09)
        return df, None, labels, tr, va, te

    if path is not None:
        df = pl.read_csv(path, infer_schema_length=10_000,
                         schema_overrides={'id': pl.Utf8})
    else:
        cache = CACHE_DIR / 'avazu.parquet'
        if cache.exists():
            df = pl.read_parquet(cache)
        else:
            from datasets import load_dataset
            df = pl.DataFrame(load_dataset('reczoo/Avazu_x4', split='train').to_dict())
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            df.write_parquet(cache)
    if n_rows:
        df = df.sample(n=n_rows, seed=42)

    labels = df['click'].cast(pl.Float32).to_numpy()
    cols   = [c for c in df.columns if c not in ('id', 'click')]
    df     = df.select(cols).with_columns(pl.all().cast(pl.Utf8))
    tr, va, te = random_split(len(df))
    return df, None, labels, tr, va, te


def load_criteo(n_rows: int = None, path: str = None) -> tuple:
    prepared = pathlib.Path(path) if path else CACHE_DIR / 'criteo_prepared.parquet'
    if prepared.exists():
        df = pl.read_parquet(prepared)
        n  = len(df)
        if n_rows and n_rows < n:
            df = df.head(n_rows)
        labels = df['label'].to_numpy()
        dense  = df.select([f'I{i}' for i in range(1, 14)]).to_numpy()
        cat    = df.select([f'C{i}' for i in range(1, 27)])
        tr, va, te = temporal_split(len(cat))
        return cat, dense, labels, tr, va, te

    cache = CACHE_DIR / 'criteo.parquet'
    if cache.exists():
        df = pl.read_parquet(cache)
    else:
        from datasets import load_dataset
        df = pl.DataFrame(load_dataset('reczoo/Criteo_x4', split='train').to_dict())
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.write_parquet(cache)
    if n_rows:
        df = df.sample(n=n_rows, seed=42)

    labels = df['Label'].cast(pl.Float32).to_numpy()
    cols   = [f'C{i}' for i in range(1, 27)]
    df     = df.select(cols).with_columns(pl.all().cast(pl.Utf8).fill_null('nan'))
    tr, va, te = random_split(len(df))
    return df, None, labels, tr, va, te



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




def load_vklsvd_seq(subset: str = 'ur0.01_ip0.01', positive: str = 'like') -> dict:
    import polars as pl
    base = 'interactions' if subset == 'whole' else f'subsamples/{subset}'
    need = ['user_id', 'item_id', 'like'] + (['timespent'] if positive == 'watch' else [])
    items_meta = None
    if positive == 'watch':
        items_meta = pl.read_parquet(fetch_vklsvd('metadata/items_metadata.parquet'),
                                     columns=['item_id', 'duration'])

    def positives(rel):
        df = pl.read_parquet(fetch_vklsvd(rel), columns=need)
        if positive == 'like':
            df = df.filter(pl.col('like'))
        else:
            df = (df.join(items_meta, on='item_id', how='left')
                    .filter((pl.col('duration') > 0)
                            & (pl.col('timespent') >= 0.5 * pl.col('duration'))))
        return df.select('user_id', 'item_id').to_numpy().astype(np.int64)

    tr = np.concatenate([positives(f'{base}/train/week_{i:02}.parquet') for i in range(25)])
    va = positives(f'{base}/validation/week_25.parquet')
    te = positives(f'{base}/test/week_26.parquet')

    users, items = np.unique(tr[:, 0]), np.unique(tr[:, 1])

    def remap(p):
        if not len(p):
            return np.zeros((0, 2), dtype=np.int64)
        keep = np.isin(p[:, 0], users) & np.isin(p[:, 1], items)
        p = p[keep]
        return np.stack([np.searchsorted(users, p[:, 0]),
                         np.searchsorted(items, p[:, 1])], axis=1)

    d = {'users': users, 'items': items,
         'train': remap(tr), 'val': remap(va), 'test': remap(te)}
    print(f"vklsvd_seq {subset}: {len(users):,} users, {len(items):,} items, "
          f"splits {len(d['train']):,}/{len(d['val']):,}/{len(d['test']):,}", flush=True)
    return d
