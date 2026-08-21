# Реструктуризация под logq

Референс — https://github.com/NonameUntitled/logq («Correcting the LogQ
Correction», RecSys 2025): плоский исследовательский репозиторий без
инсталлируемого пакета. Конвенции референса:

- общие модули на корне: `gsasrec.py` / `transformer_decoder.py` (модель),
  `dataset_utils.py` (данные), `eval_utils.py` (метрики), `utils.py`
  (`load_config`, `get_device`, `build_model`), `config.py` (класс конфига);
- один `train_<вариант>.py` на режим обучения — самодостаточный скрипт,
  цикл обучения читается сверху вниз;
- гиперпараметры не в argparse, а в `configs/<dataset>_<вариант>.py`
  (крошечный python-файл, инстанцирующий класс из `config.py`); у train-скрипта
  единственный флаг `--config`, конфиг грузится через importlib;
- вариант модели — это либо отдельный train-скрипт (если отличается код цикла),
  либо поле конфига (если отличаются только параметры: `ml1m_sasrec.py` vs
  `ml1m_other.py`);
- чекпоинты в `models/` (в .gitignore; у нас — `checkpoints/`), отдельный
  `evaluate.py --config --checkpoint`.

План написан от текущего дерева (пакет `unified_embeddings/` + `benchmark/`);
текущая структура откатывается: пакет, `pyproject.toml` и editable-install
удаляются, всё возвращается на корень.

## Целевая структура

```
unified/
├── config.py               классы конфигов: RankingConfig, CandgenConfig,
│                           SasrecConfig, GTSConfig, OrthogonalityConfig
├── configs/                по файлу на (датасет, арка[, вариант])
│   ├── __init__.py
│   ├── ml1m_ranking.py  avazu_ranking.py  criteo_ranking.py
│   ├── ml1m_candgen.py  beauty_candgen.py  steam_candgen.py
│   │   gowalla_candgen.py  yambda50m_candgen.py
│   ├── ml1m_sasrec.py  beauty_sasrec.py  steam_sasrec.py  yambda50m_sasrec.py
│   ├── ml1m_sasrec_2probe.py  beauty_sasrec_2probe.py  steam_sasrec_2probe.py
│   │                           (нынешний run_sasrec_all.sh: only Multiplex,
│   │                            no-align-roles, probes=2 concat, runs=5)
│   ├── yambda50m_gts_multi.py  yambda50m_gts_likes.py  yambda50m_gts_listens.py
│   ├── vklsvd_gts.py
│   └── ml1m_orthogonality.py
├── embeddings.py           ue-модули + prehash + MultiplexedEmbeddings + _TableView
├── models.py               SimpleMLP, DCNV2, TwoTower, TwoTowerFeat, SingleLayer
├── gsasrec.py              вендоренный (сейчас unified_embeddings/sasrec/)
├── transformer_decoder.py  вендоренный
├── dataset_utils.py        все лоадеры: movielens/avazu/criteo (ranking),
│                           leave-one-out/gowalla/yambda (retrieval),
│                           load_yambda_gts / load_vklsvd_gts (сейчас в candgen_gts)
├── eval_utils.py           evaluate (AUC), eval_split (sasrec),
│                           evaluate (candgen hr/ndcg), evaluate_gts
├── utils.py                load_config (importlib, как в logq), get_device
├── train_ranking.py        run.py + ranking.py слиты: DCN/MLP CTR-свип
├── train_candgen.py        двухбашенный кандген
├── train_candgen_gts.py    GTS с фичами (остаётся только цикл + encode_all)
├── train_sasrec.py
├── train_orthogonality.py  §4.2 (тренирует однослойную модель → train_-префикс)
├── evaluate.py             --config --checkpoint: оценка сохранённого чекпоинта
├── prepare_data.py         подготовка Avazu/Criteo — на корень без изменений
├── report.py  plots.py  tb_export.py   анализ логов — на корень без изменений
├── run_server.sh  run_server_gts.sh  run_sasrec_all.sh
├── configs пишутся так, что скрипты запускаются из корня (как в logq)
├── docs/                   DATASETS.md, THEORY.md — остаются
├── experiment_logs/  plots/  runs/  datasets/  ml-1m/   без изменений
└── requirements.txt  README.md  .gitignore
```

Удаляются: `unified_embeddings/` (пакет), `benchmark/`, `pyproject.toml`,
`unified_embeddings.egg-info/`; editable-инсталл снести
(`uv pip uninstall unified-embeddings`).

## Маппинг файлов

| Сейчас | Станет |
|---|---|
| unified_embeddings/embeddings.py | embeddings.py |
| unified_embeddings/sasrec/embeddings.py | → влить в embeddings.py |
| unified_embeddings/sasrec/gsasrec.py | gsasrec.py |
| unified_embeddings/sasrec/transformer_decoder.py | transformer_decoder.py |
| unified_embeddings/sasrec/LICENSE | удаляется; атрибуция — шапки в gsasrec.py / transformer_decoder.py |
| unified_embeddings/models/networks.py | models.py |
| TwoTower (candgen), TwoTowerFeat (candgen_gts), SingleLayer (orthogonality) | → models.py |
| unified_embeddings/dataset/ranking.py + retrieval.py | dataset_utils.py |
| load_yambda_gts / load_vklsvd_gts (candgen_gts.py) | → dataset_utils.py |
| unified_embeddings/utils.py | utils.py (+ load_config) |
| benchmark/run.py + benchmark/ranking.py | train_ranking.py |
| benchmark/candgen.py | train_candgen.py |
| benchmark/candgen_gts.py | train_candgen_gts.py |
| benchmark/sasrec.py | train_sasrec.py |
| benchmark/orthogonality.py | train_orthogonality.py |
| benchmark/prepare_data.py, report.py, plots.py, tb_export.py | на корень, имена те же |
| DATASET_CFG + argparse-дефолты всех скриптов | config.py + configs/*.py |
| evaluate/train_model, eval_split, evaluate_gts из train-скриптов | eval_utils.py |

## Конфиги

`config.py` — dataclass на арку, поля = текущие argparse-флаги + DATASET_CFG.
Файл в `configs/` только инстанцирует:

```python
# configs/ml1m_ranking.py
from config import RankingConfig

config = RankingConfig(
    dataset='movielens', ml1m_path='ml-1m', ml_labels='wang',
    emb_dim=30, emb_levels=13_653, cross=1, dnn=(192,),
    batch=128, lr=1e-3, epochs=30, patience=5, weight_decay=1e-5,
    budgets=[1.0, 0.5, 0.1], runs=1, seed=42,
)
```

Правила переноса:

- параметры эксперимента (budgets, methods, probes/combine, runs, seed, лейблы,
  interaction/subset/positive, batches_per_epoch, tb/out) → поля конфига;
- у train-скриптов остаётся `--config` (обязательный) и `--dry-run` только у
  train_ranking.py (печать размеров таблиц);
- вариант = отдельный файл конфига, не флаг: суженный SASRec-прогон — это
  `*_sasrec_2probe.py`, а не `--only Multiplex --probes 2 ...`;
- один запуск = один датасет (как в logq); нынешний `--skip avazu criteo`
  исчезает, три датасета ranking — три конфига;
- дефолты конфигов обязаны воспроизводить текущие дефолты скриптов один в один
  (сверить с argparse перед удалением): ranking ML 128/1e-3, avazu/criteo
  512/2e-4; sasrec emb128/maxlen200/heads1/blocks2/dropout0.5/batch128/bpe100/
  lr1e-3/epochs300/patience20, отбор по val NDCG@100, tied-baseline включён,
  beauty max_len=50, yambda_50m batch=32; candgen k32/batch1024/lr1e-3/
  epochs20/patience3/loss=full; gts k32/batch4096/lr1e-3/epochs30/patience5.

## Train-скрипты

- Каждый скрипт самодостаточен по циклу обучения (уже так после инлайна
  train_model в ranking): модель/данные/метрики импортируются из общих модулей,
  цикл — в скрипте.
- `train_ranking.py` = слитые run.py (CLI, json/summary) + ranking.py
  (run_dataset, train_model, evaluate → evaluate уходит в eval_utils).
- Из `train_candgen_gts.py` уезжают лоадеры (dataset_utils), TwoTowerFeat
  (models), evaluate_gts (eval_utils); остаются encode_all и цикл.
- Формат JSON-логов в experiment_logs/, TB-теги и имена файлов не меняются
  (logq только печатает — наши логи это осознанное расширение, оставляем).
- Логика обучения, протоколы, сиды, метрики не меняются нигде.

## Shell-скрипты и README

- `run_sasrec_all.sh` → цикл `python train_sasrec.py
  --config=configs/${ds}_sasrec_2probe.py`; аналогично run_server.sh
  (train_ranking + configs/criteo_ranking.py, avazu_ranking.py) и
  run_server_gts.sh; setup() больше не ставит пакет (строчка
  `uv pip install -e .` удаляется).
- README: раздел Running переписывается в стиле logq — по подразделу на
  train-скрипт с одной командой `python train_<арка>.py
  --config=configs/<dataset>_<арка>.py`; Setup теряет editable-install.
- docs/DATASETS.md, CLAUDE.md — обновить пути после переноса.

## Порядок работ

1. Расплющить модули: git mv из пакета на корень, влить sasrec/embeddings.py в
   embeddings.py, товеры в models.py, лоадеры в dataset_utils.py, метрики в
   eval_utils.py; починить импорты. Удалить pyproject.toml, пакетные
   `__init__.py`, снести editable-install.
2. config.py + configs/: перенести DATASET_CFG и argparse-дефолты, train-скрипты
   перевести на `--config` (utils.load_config — паттерн logq через importlib).
3. Слить run.py+ranking.py в train_ranking.py; переименовать остальные в
   train_*; report/plots/tb_export/prepare_data на корень.
4. Чекпоинты: сохранять best_state в
   `checkpoints/<арка>-<dataset>-<method>-b<budget>-seed<k>.pt` + `evaluate.py
   --config --checkpoint`; `checkpoints/` в .gitignore (не `models/`, как в logq —
   имя конфликтовало бы с модулем models.py). Сейчас best-state живёт
   только в памяти — это единственный кусок workflow logq, которого у нас нет.
5. README / shell-скрипты / docs / CLAUDE.md.

## Проверка

```bash
python train_ranking.py --config=configs/ml1m_ranking.py --dry-run
python train_ranking.py --config=configs/ml1m_ranking.py        # 1 эпоха: epochs=1 в конфиге-копии
python train_sasrec.py  --config=configs/ml1m_sasrec.py         # прервать после 1-2 эпох
python train_candgen.py --config=configs/ml1m_candgen.py
python train_orthogonality.py --config=configs/ml1m_orthogonality.py
bash -n run_server.sh run_server_gts.sh run_sasrec_all.sh
grep -rn "unified_embeddings\|benchmark\." --include="*.py" --include="*.sh" --include="*.md" .   # пусто
```

Смок-эпоха ranking должна дать те же цифры, что текущий
`benchmark/run.py --epochs 1` (loss ~0.37-0.39, val_auc ~0.85-0.87 на ML-1M) —
конфиги воспроизводят дефолты, математика не тронута.

## Метрики по аркам (что логируется)

Отбор чекпоинта = метрика early stopping. TB live = пишется SummaryWriter'ом
по ходу обучения; TB export = `tb_export.py` конвертирует JSON-историю
(whitelist тегов: auc_val, auc_test, ndcg10_val, ndcg100_val, recall100_val;
мультисид усредняется в одну кривую).

| Арка | По эпохе (history/консоль) | Финал (JSON) | Отбор | TB |
|---|---|---|---|---|
| ranking | train_loss, val_auc, emb_l2_mean (+test_auc при paper-protocol) | test AUC mean±std по runs, best_val_auc, table{rows,size_mb,rows_over_vocab}, n_params, emb_l2_final, epochs_run, time_sec | val AUC | live: auc_val, auc_test |
| candgen | loss, val_hr@{10,20,50,100}, val_ndcg@{10,20,50,100}, proj_overlap, emb_l2 | test HR/NDCG@k, best_val_hr10, proj_overlap_mean/max, ranks+rank_sum (Cor. 3), emb_l2_final, table stats | val HR@10 | export: ndcg10_val, ndcg100_val |
| sasrec | loss, val_hr/ndcg@{10,20,100}, emb_l2 | test HR/NDCG@k, best_val_ndcg100, table stats, n_params | val NDCG@100 | export: ndcg10_val, ndcg100_val |
| candgen_gts | loss, val recall/ndcg/hr@{10,50,100}, coverage@k, proj_overlap, emb_l2 | test recall/ndcg/hr/coverage@k, best_val_ndcg100, overlap, ranks, emb_l2_final | val NDCG@100 | live: ndcg100_val, recall100_val |

При leave-one-out (|T|=1: beauty/steam/ml1m в sasrec) HR@k ≡ recall@k.

## Зафиксированные решения

1. LICENSE-файл не создаём; sasrec/LICENSE удаляется, атрибуция остаётся
   шапками-комментариями в вендоренных gsasrec.py / transformer_decoder.py.
2. Чекпоинты + отдельный evaluate.py — в скоупе (фаза 4 обязательна).
3. Имя модуля — embeddings.py.
4. TensorBoard остаётся обоими путями: live-логирование в train-скриптах
   (поле `tb` конфига, '' = off) + tb_export.py для любых JSON-логов.
5. Метрики — как в таблице выше, без изменений.
6. Гиперпараметр `probes` (число хэшей на значение в Multiplex) добавлен во все
   арки: SASRec — как было (probes + combine concat/mean), ranking/candgen/GTS —
   probes с concat при равных байтах (пул строк ×N, ширина d/N); дефолт 1
   воспроизводит прежнее поведение бит-в-бит. Закрывает отличие №6 от статьи
   (multi-probe UE).
