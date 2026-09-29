# RQ4: sequential retrieval with item features

Does the effect persist when the user representation comes from a Transformer
over interaction history? The main recipe is Beauty/Steam, BPR, collisionless
or HT PF/MP, budgets x1/x10/x25, H=1/4, seed42. The H4 x10/x25 slice and
collisionless references also have seed43 replications.

## Prepare sequential indices once

Set `PREPARE_SEQUENTIAL = True` in the corresponding
`runs/<dataset>/retrieval/prepare_retrieval.py`, then run it. Existing static
artifacts are reused; raw files are not reprocessed unless explicitly requested.
Do not rebuild prepared data used by running experiments.

## Training

`train.py` fixes sequential+BPR. Defaults select Steam, HT, x10, H=4,
seed42 and patience5. It uses train batch256 on Steam and 128 on Beauty,
because full-catalog train metrics cover all valid history positions.

```bash
python runs/rq4/train.py --dataset steam --approach collisionless --seed 42
python runs/rq4/train.py --dataset steam --budget-fraction 0.1 --num-hashes 4 --seed 42
python runs/rq4/train.py --dataset steam --budget-fraction 0.1 --num-hashes 4 --seed 42 --multiplex
python runs/rq4/train.py --dataset beauty --budget-fraction 0.04 --num-hashes 4 --seed 43 --multiplex
```

The default feature sets are `item_idx, brand, categories` on Beauty and
`item_idx, developer, publisher, genres` on Steam. History and candidate items
share the item tower and its feature set. There is **no user-feature prefix**
in this main experiment. Supplying user features via `--features` enables a
different optional ablation, not the published default recipe.

Architecture: embedding/output width32, two pre-norm Transformer encoder layers,
two attention heads, FFN128, dropout0.1, learned positional embeddings,
max history200. Optimizers and lr match static training, feature regularisation
is off, eval batch128, cap100000 steps, patience5, logging every20 steps.

Train samples are per-user windows with all valid next-item positions; users
without a train next-item pair have no train sample. Evaluation has one target
interaction per sample and reads the last real history position. Earlier
validation/test events can enter later histories under the causal protocol.
Static and sequential evaluation exclude first-ever user events consistently.
Full-catalog Recall/NDCG@1000 and warm/cold slices use the same target protocol;
the different training samples and available history are part of the model
comparison, not a controlled change of architecture alone.
