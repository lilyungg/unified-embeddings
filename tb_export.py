import argparse
import json
import pathlib

METRIC = {'val_auc': 'auc_val', 'test_auc': 'auc_test',
          'val_ndcg10': 'ndcg10_val', 'val_ndcg100': 'ndcg100_val',
          'val_recall100': 'recall100_val'}


def _metric(key: str):
    return METRIC.get(key)


def _mean_histories(hists):
    if len(hists) == 1:
        return hists[0]
    by_epoch = {}
    for h in hists:
        for row in h:
            by_epoch.setdefault(int(row.get('epoch', 0)), []).append(row)
    out = []
    for e in sorted(by_epoch):
        rows, m = by_epoch[e], {'epoch': e}
        for k in {k for r in rows for k in r} - {'epoch'}:
            vs = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
            if vs:
                m[k] = sum(vs) / len(vs)
        out.append(m)
    return out


def _write(run_dir: pathlib.Path, history, suffix: str, force: bool) -> bool:
    if run_dir.exists() and not force:
        return False
    if len(history) < 2:
        return False
    if not any(_metric(k) for row in history for k in row):
        return False
    from torch.utils.tensorboard import SummaryWriter
    w = SummaryWriter(log_dir=str(run_dir))
    for row in history:
        step = int(row.get('epoch', 0))
        for k, v in row.items():
            m = _metric(k)
            if m and isinstance(v, (int, float)):
                w.add_scalar(f'{m}/{suffix}', v, step)
    w.close()
    return True


def export(path: pathlib.Path, out: pathlib.Path, force: bool) -> int:
    try:
        d = json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return 0
    stem, n = path.stem, 0

    if isinstance(d, dict) and isinstance(d.get('results'), list):
        groups = {}
        for r in d['results']:
            if isinstance(r, dict) and r.get('history'):
                key = (str(r.get('method', 'run')), r.get('budget'))
                groups.setdefault(key, []).append(r['history'])
        for (method, budget), hists in groups.items():
            n += _write(out / stem / f'b{budget}' / method.replace(' ', '_'),
                        _mean_histories(hists), f'{stem}/b{budget}', force)
    elif isinstance(d, dict) and 'budgets' in d:
        for b, exps in d['budgets'].items():
            for exp, e in exps.items():
                hists = e.get('histories') or []
                if hists:
                    n += _write(out / stem / f'b{b}' / exp.replace(' ', '_'),
                                _mean_histories(hists), f'{stem}/b{b}', force)
    return n


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('paths', nargs='*', help='JSON logs (default: experiment_logs/*.json)')
    p.add_argument('--out', default='runs')
    p.add_argument('--force', action='store_true', help='overwrite existing run dirs')
    args = p.parse_args()

    paths = [pathlib.Path(x) for x in args.paths] or \
        sorted(pathlib.Path('experiment_logs').glob('*.json'))
    out = pathlib.Path(args.out)
    total = 0
    for path in paths:
        k = export(path, out, args.force)
        total += k
        if k:
            print(f'{path.name}: {k} curves -> {out}/{path.stem}/')
    print(f'\n{total} curves exported. View: tensorboard --logdir {out}')


if __name__ == '__main__':
    main()
