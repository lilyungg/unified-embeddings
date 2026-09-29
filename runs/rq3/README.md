# RQ3: transfer across retrieval objectives

Compare the MP-minus-PF effect under BPR, sampled CE, and full-catalog CE.
This is retrieval softmax CE, not pointwise ranking BCE.

`train.py` fixes the static model. Defaults select Beauty, HT, x10, H=4,
full CE, seed 42, patience 5. The matched block covers Beauty and Steam,
PF/MP, all three losses, and seeds 42/43/44.

```bash
python runs/rq3/train.py --dataset beauty --loss bpr --seed 42
python runs/rq3/train.py --dataset beauty --loss bpr --seed 42 --multiplex
python runs/rq3/train.py --dataset beauty --loss sampled_ce --seed 42
python runs/rq3/train.py --dataset beauty --loss sampled_ce --seed 42 --multiplex
python runs/rq3/train.py --dataset beauty --loss full_ce --seed 42
python runs/rq3/train.py --dataset beauty --loss full_ce --seed 42 --multiplex
```

Repeat with `--dataset steam` and seeds 43/44. The compact design uses HT rather
than every embedding family. All ordinary training options remain available;
overriding them defines a different experiment. For a collisionless reference,
use `--approach collisionless` without `--multiplex`. Beauty has a matched
seed42 collisionless reference per loss; older Steam collisionless pilots are
not automatically matched references for this block.

## Held fixed and changed

All static features, d=32, output_dim=32, main budget x10, H=4,
SparseAdam/AdamW at 3e-4, no feature regularisation, train batch1024, eval128,
workers4, cap100000 steps, patience5, full-catalog evaluation and checkpoint
selection by validation Recall@1000 stay fixed.

- BPR: one uniform negative per target, no exclusions.
- Sampled CE: one pool of 256 uniform negatives shared by the batch;
  positive score first, no positive/history filtering.
- Full CE: all catalog items, with the positive item index as the class.

The negative estimator and candidate count change with the objective. This is
an objective-recipe comparison, not an isolated causal effect of a loss formula.
Compare PF/MP within each loss and seed first. Report paired effects and seed
variation; do not select H or a checkpoint by test results.
