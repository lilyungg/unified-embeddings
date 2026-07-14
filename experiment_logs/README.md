Experiment logs — one JSON per run per dataset.

Files produced by run.py or test.py follow the naming convention:
  YYYYMMDD_HHMMSS_<dataset>.json
  YYYYMMDD_HHMMSS_summary.json

Files prefixed with `historic_` are results transcribed from EXPERIMENTS.md.
They may have null n_params/time_sec fields and use the old full-rank DCN-V2
architecture (not the low-rank factorization now in models.py).

Historic files:
  historic_movielens.json              — MovieLens-1M, full-rank DCN, 10 epochs, CPU
  historic_avazu_fullrank_dcn.json     — Avazu 1M, full-rank DCN (overfit), 10 epochs, CPU
  historic_criteo_fullrank_dcn.json    — Criteo 1M, full-rank DCN (overfit), 10 epochs, MPS
  historic_avazu_lr_dcn.json           — Avazu 1M, low-rank DCN r=64 (partial), patience=3
