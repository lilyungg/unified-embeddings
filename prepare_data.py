import argparse
import gzip
import json
import pathlib
import shutil
import tempfile

import polars as pl


CRITEO_VOCAB = {
    'C1': 676, 'C2': 533, 'C3': 17447, 'C4': 19995, 'C5': 180, 'C6': 13,
    'C7': 9693, 'C8': 337, 'C9': 3, 'C10': 14637, 'C11': 4378, 'C12': 17795,
    'C13': 3067, 'C14': 26, 'C15': 6504, 'C16': 18679, 'C17': 10, 'C18': 3102,
    'C19': 1557, 'C20': 3, 'C21': 18230, 'C22': 10, 'C23': 14, 'C24': 13079,
    'C25': 56, 'C26': 10581,
}

AVAZU_VOCAB = {
    'hour': 24, 'C1': 8, 'banner_pos': 8, 'site_id': 3317, 'site_domain': 3887,
    'site_category': 24, 'app_id': 4438, 'app_domain': 277, 'app_category': 29,
    'device_id': 67767, 'device_ip': 163804, 'device_model': 6217,
    'device_type': 6, 'device_conn_type': 5, 'C14': 2309, 'C15': 9, 'C16': 10,
    'C17': 405, 'C18': 5, 'C19': 66, 'C20': 167, 'C21': 56,
}

CRITEO_DENSE = [f'I{i}' for i in range(1, 14)]
CRITEO_CAT   = [f'C{i}' for i in range(1, 27)]


def _kept_values(lf: pl.LazyFrame, col: str, target: int) -> list:
    counts = (lf.group_by(col).len()
                .sort(['len', col], descending=[True, False])
                .collect(engine='streaming'))
    values = counts[col].to_list()
    if len(values) <= target:
        return values
    return values[:target - 1]


def _encode_exprs(lf: pl.LazyFrame, vocab_targets: dict, out_json: pathlib.Path):
    exprs, sizes = [], {}
    for col, target in vocab_targets.items():
        kept = _kept_values(lf, col, target)
        sizes[col] = len(kept) + 1
        exprs.append(
            pl.col(col).replace_strict(
                old=kept, new=list(range(1, len(kept) + 1)), default=0,
                return_dtype=pl.Int32).alias(col))
        print(f'  {col}: kept {len(kept):,} of target {target:,}', flush=True)
    out_json.write_text(json.dumps(sizes, indent=1))
    return exprs


def _maybe_gunzip(raw: str) -> str:
    if not raw.endswith('.gz'):
        return raw
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    print(f'decompressing {raw} -> {tmp.name}', flush=True)
    with gzip.open(raw, 'rb') as src, open(tmp.name, 'wb') as dst:
        shutil.copyfileobj(src, dst)
    return tmp.name


def prepare_criteo(raw: str, out: pathlib.Path) -> None:
    schema = {'label': pl.Int32}
    schema.update(dict.fromkeys(CRITEO_DENSE, pl.Float64))
    schema.update(dict.fromkeys(CRITEO_CAT, pl.Utf8))
    lf = pl.scan_csv(_maybe_gunzip(raw), separator='\t', has_header=False,
                     new_columns=list(schema), schema_overrides=schema)
    lf = lf.with_columns(pl.col(c).fill_null('') for c in CRITEO_CAT)

    print('pass 1: vocabularies (Table 4 pruning)', flush=True)
    cat_exprs = _encode_exprs(lf, CRITEO_VOCAB, out.with_suffix('.vocab.json'))

    dense_exprs = [
        pl.max_horizontal(pl.col(c).fill_null(0) + (4 if c == 'I2' else 1),
                          pl.lit(1.0)).log().cast(pl.Float32).alias(c)
        for c in CRITEO_DENSE
    ]

    print('pass 2: encode + normalize -> parquet (row order preserved)', flush=True)
    (lf.select([pl.col('label').cast(pl.Float32)] + dense_exprs + cat_exprs)
       .sink_parquet(out, maintain_order=True))
    print(f'done: {out}', flush=True)


def prepare_avazu(raw: str, out: pathlib.Path) -> None:
    lf = pl.scan_csv(_maybe_gunzip(raw),
                     schema_overrides={'id': pl.Utf8, 'hour': pl.Int64})
    lf = (lf.drop('id')
            .with_columns((pl.col('hour') % 100).cast(pl.Utf8))
            .with_columns(pl.col(c).cast(pl.Utf8).fill_null('')
                          for c in AVAZU_VOCAB))

    print('pass 1: vocabularies (Table 5 pruning)', flush=True)
    cat_exprs = _encode_exprs(lf, AVAZU_VOCAB, out.with_suffix('.vocab.json'))

    print('pass 2: encode -> parquet', flush=True)
    (lf.select([pl.col('click').cast(pl.Float32).alias('label')] + cat_exprs)
       .sink_parquet(out))
    print(f'done: {out}', flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('dataset', choices=['criteo', 'avazu'])
    p.add_argument('--raw', required=True, help='raw Kaggle file (train.txt / train.gz)')
    p.add_argument('--out', default=None, help='output parquet path')
    args = p.parse_args()

    out = pathlib.Path(args.out or f'datasets/{args.dataset}_prepared.parquet')
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.dataset == 'criteo':
        prepare_criteo(args.raw, out)
    else:
        prepare_avazu(args.raw, out)


if __name__ == '__main__':
    main()
