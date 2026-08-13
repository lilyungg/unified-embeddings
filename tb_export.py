"""Export experiment_logs/*.json training histories to TensorBoard.

Makes past runs browsable next to live ones (train.py and candgen_gts.py write
TB directly; candgen.py and sasrec_run.py log JSON histories — this converts).

Handles both JSON shapes:
  ranking    {budgets: {b: {exp: {runs: [...], histories: [[row]]}}}}
             row: {epoch, train_loss, val_auc, emb_l2_mean}
  retrieval  {results: [{method, budget, seed, history: [row]}]}
             row: {epoch, loss, val_*, proj_overlap?, emb_l2}

Layout is built for small charts: tag = <metric>/<json_stem>/b<budget>, so
TensorBoard groups by metric and each card holds one experiment's methods —
never more curves than the sweep has methods. Multi-seed runs are averaged
into one curve per method. Final test numbers are NOT exported (single-dot
charts; they live in the JSONs and README tables).

Exported curves: AUC for the ranking arm (auc_val, auc_test under the paper
protocol) and validation NDCG / recall@100 for the retrieval arms. Everything
else — losses, norms, overlap, other cutoffs — stays in the JSON histories.
Runs shorter than 2 epochs (smokes) are skipped, so no single-dot charts.

Usage
  python tb_export.py                     # all experiment_logs/*.json
  python tb_export.py experiment_logs/20260808_*.json --force
Then: tensorboard --logdir runs
"""
import argparse
import json
import pathlib

METRIC = {'val_auc': 'auc_val', 'test_auc': 'auc_test',
          'val_ndcg10': 'ndcg10_val', 'val_ndcg100': 'ndcg100_val',
          'val_recall100': 'recall100_val'}


def _metric(key: str):
    return METRIC.get(key)                       # everything else stays in JSON


def _mean_histories(hists):
    """Average rows across seeds per epoch; ragged tails use available seeds."""
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
        return False                             # smoke run: a dot, not a curve
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

    if isinstance(d, dict) and isinstance(d.get('results'), list):  # retrieval
        groups = {}
        for r in d['results']:
            if isinstance(r, dict) and r.get('history'):
                key = (str(r.get('method', 'run')), r.get('budget'))
                groups.setdefault(key, []).append(r['history'])
        for (method, budget), hists in groups.items():
            n += _write(out / stem / f'b{budget}' / method.replace(' ', '_'),
                        _mean_histories(hists), f'{stem}/b{budget}', force)
    elif isinstance(d, dict) and 'budgets' in d:                    # ranking
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
