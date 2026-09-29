# RQ1: static BPR at a fixed lookup budget

Does feature multiplexing improve retrieval within each embedding family, and
how does the effect depend on compression? Compare PF and MP within the same
dataset, method, H, seed, and stopping protocol.

`train.py` fixes static+BPR and uses the shared CLI and runtime. All seven
approaches and four datasets are available. It defaults to collisionless ML1M;
explicitly select the method, budget, and H for a compressed run.

```bash
python runs/rq1/train.py --dataset beauty --approach collisionless --seed 42 --patience 5
python runs/rq1/train.py --dataset beauty --approach hashing_trick --budget-fraction 0.1 --num-hashes 4 --seed 42 --patience 5
python runs/rq1/train.py --dataset beauty --approach hashing_trick --budget-fraction 0.1 --num-hashes 4 --seed 42 --patience 5 --multiplex
python runs/rq1/popularity.py --dataset beauty
```

The same recipe accepts `hash_embedding`, `pq`, `qr`, `hashed_net`, and `robe_z`.
Use budget fractions `1.0`, `0.1`, `0.04` for x1/x10/x25. The compact family
comparison uses H=4; the broader H study uses H=1/4 (the earlier HT study also
includes H=2). These are configurable recipes, not a command that automatically
launches the Cartesian product or a claim that every cell was measured.

## Comparison conventions

- Equal total **main lookup** parameters and row width within a budget-matched
  PF/MP pair. Readers and HE importance weights are additional parameters;
  report their counts separately. Low-cardinality exact tables are included
  in the main budget. x1 hashing is not a collisionless mapping.
- PQ with one component duplicates the HT H=1 reference. QR H=1 cannot compress
  an injective code. Other infeasible QR allocations are N/A, not enlarged
  budgets. Do not report aliases as independent methods.
- All default dataset features, d=32, output_dim=32, FP32,
  SparseAdam/AdamW at 3e-4, explicit feature regularisation off; train batch
  1024, eval 128, workers 4, at most 100000 steps.
- Best validation Recall@1000 selects the checkpoint; report full-catalog test
  Recall/NDCG@1000 and warm/cold slices. Popularity uses the same target protocol.
- Historical HT runs on Beauty/Steam/Yambda used patience 20; later family
  comparisons commonly used 5, and the last Yambda x10/H4 family batch used 2.
  Supply `--patience` explicitly and label these strata; they are not all one
  matched stopping-budget comparison. Training seeds do not change hash seeds.

## QR initialization control

Keep this separate from the default QR results. It matches the initial product
variance to one uniform lookup without changing the main parameter budget:

```bash
python runs/rq1/qr_initialization.py --dataset steam --budget-fraction 0.1 --num-hashes 4 --seed 42 --patience 5
```

Add `--multiplex` for MP. The control changes initialization only, not QR's
indexing or combination rule. Commands can use `--device cuda:0` or select a
physical GPU with `CUDA_VISIBLE_DEVICES`.
