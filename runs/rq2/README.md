# RQ2: mechanism and limits of feature sharing

These controls complement RQ1; they are not additional equal-budget baselines.
They reuse the existing experiment implementations, with analysis in
`src/analysis/` and setup in this directory. No queue or server paths are needed.

## 1. Shared versus independent tables with the same address space

`paired_hashing.py` compares global multiplex, tower-local and extended PF.
Each hashed feature can address M rows in every layout. Global uses one M-row
table; tower-local gives each tower an independent copy; extended PF gives each
feature a copy. Thus the shared portion costs M/2M/4M rows on Beauty and
M/2M/5M on Steam. Low-cardinality exact tables stay separate and unchanged.
Unlike RQ1, total lookup budgets are deliberately unequal.

The copies start with the same values, readers and predictions. Training RNG
is reset after construction, keeping shuffle and negative draws paired.

```bash
python runs/rq2/paired_hashing.py --dataset beauty --layout multiplex --budget-fraction 0.1 --num-hashes 4 --seed 42 --patience 20
python runs/rq2/paired_hashing.py --dataset beauty --layout tower_local --budget-fraction 0.1 --num-hashes 4 --seed 42 --patience 20
python runs/rq2/paired_hashing.py --dataset beauty --layout extended_per_feature --budget-fraction 0.1 --num-hashes 4 --seed 42 --patience 20
```

Beauty's main grid is x1/x10/x25, H1/H4, seed42, patience20; its x10/H4 triplet
also has seed43. Steam's grid is x10/x25, H4, seed42, **patience5** (pass that
override). All are static HT+BPR. Similar quality does not establish absence
of interference or make extended PF a mathematical quality upper bound.

## 2. Numerical gradient and readout checks

```bash
python runs/rq2/check_bpr_gradients.py --dataset ml1m --layout multiplex --budget-fraction 0.1
python runs/rq2/check_reader_interference.py --seeds 42 43 44
```

The first freezes model parameters and compares the full expected BPR gradient
to its namespace decomposition and sampled-negative estimates. It supports
ML1M/Steam, H=1, plus `--checkpoint` for a matching trained state. Without
`--max-interactions` it uses all train interactions and all catalog negatives;
this is a substantive diagnostic, not an import smoke check.
The second checks a controlled foreign update through frozen readers without
a dataset. It checks the readout formula and orthogonal/nonorthogonal cases.

## 3. Frozen readers and the rank boundary

Each job jointly trains MP and extended PF using identical sampled interactions
and negatives, fixed reader geometry, d=64, output_dim=32, and SGD on lookup
tables only. Readers have equal Frobenius norm across ranks.

```bash
CUDA_VISIBLE_DEVICES=0 python runs/rq2/frozen_reader_training.py --dataset beauty --rank 8 --geometry packed --budget-fraction 0.1 --seed 42 --steps 100000 --output-dir experiment_logs/rq2/beauty_frozen_r8
```

The full frozen block uses ML1M/Beauty/Steam, fractions 0.1/0.04,
`(rank, geometry)` = `(8, packed)`, `(8, aligned)`, `(16, packed)`,
`(32, packed)`, and seeds42/43/44. Pass `--steps 100000` explicitly; the
standalone script's older default is 30000. Other settings: batch1024, lr10,
fixed probe4096, logs every1000 steps, full-catalog retrieval on 1024 validation
queries. These are **probe metrics, not full-test results**.

Boundary follow-ups use packed geometry, seed42 and both budgets: Beauty
ranks15/17, Steam and ML1M ranks12/13. With four/five sharing namespaces,
respectively, these bracket sum(rank) <= 64. Packed row spaces wrap around
when the rank sum exceeds 64; aligned spaces overlap even below the boundary.

Saved outputs include geometry, gradient/readout diagnostics, trajectories,
probe IDs, readers and checkpoints. The rank condition is necessary for
orthogonality; it is not a bound on Recall or proof of spontaneous low rank.
Changing rank can also change model expressivity, so cross-rank quality
differences do not isolate interference alone.

## 4. Learned-reader dynamics under the ordinary optimizer recipe

```bash
CUDA_VISIBLE_DEVICES=0 python runs/rq2/reader_rank_training.py --dataset beauty --layout multiplex --seed 42 --output-dir experiment_logs/rq2/beauty_learned_mp
CUDA_VISIBLE_DEVICES=0 python runs/rq2/reader_rank_training.py --dataset beauty --layout extended_per_feature --seed 42 --output-dir experiment_logs/rq2/beauty_learned_extended
```

Repeat for Steam and seed43. Settings: static HT+BPR, H1, x10, d64 -> 32,
random trainable readers, SparseAdam/AdamW at3e-4, batch1024, patience5,
cap100000. No low-rank constraint or orthogonality penalty is applied.
It records numerical/stable ranks, singular spectra, projector overlaps and
fixed-probe readout sensitivity, alongside ordinary full validation/test.
Probe construction preserves training RNG. Gradient sensitivity is not a
decomposition of the coordinate-wise Adam update; the fixed-subspace SGD
trajectory theorem does not apply to this learned-reader run.

## 5. Post-hoc checkpoint diagnostics

`check_reader_diagnostics.py` measures namespace gradient readout effects;
`check_reader_alignment.py` compares trained reader geometry with initialization.
Both accept ML1M/Steam/Yambda H1 static checkpoints. Supply the original layout
and budget, and use the matching preprocessed hashes. These tools are for
32 -> 32 HT/exact setups; use the trajectory outputs above for the d64 learned
experiment. Run `--help` for checkpoint and fixed-probe options. They do not
retrain a checkpoint or generalize the theorem to sequential models.
