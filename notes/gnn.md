# GNN on the transaction graph (Colab) - work log

(in progress; final write-up replaces this log)

- 2026-10-24: 9-family cached features: train 40,131 x 353, test 40,101 x 353; 151 columns have partial
  NaNs, 26 binary, no constant columns, none monotone in absolute time (max |spearman vs row order| 0.40,
  cycles.recip_next - relative-in-week, not an identity), train/test KS max 0.15 (motifs.s120_amt_cv).
- torch is not in the project venv; the smoke test uses a scratch install outside the repo.
- Graph sparsity (train, 8 directed relation types = shared-wallet role pair x before/after, k=8 nearest):
  hand-off <=1h / same-wallet <=10m: 118k edges for 40k nodes (median in-degree 0 per type);
  6h / 1h: 619k edges; 24h / 6h: 1.45M. Edges into positives that come from positives
  (label homophily, analysis only): 1h/10m -> forward hand-off 0.55-0.59, reverse-time hand-off
  0.07-0.09, same-wallet 0.89-0.91; 6h/1h -> 0.24 / 0.04 / 0.64-0.70; 24h/6h -> 0.13 / 0.04 / 0.29.
  Decision: export 6h/1h with raw signed time gaps; the model filters to any narrower window at load
  (config handoff_window / same_window), so spec (1h/10m) vs wide is a Colab config switch, not a re-export.
- Export (`python -m src.gnn_export`, 24 s, ~1 GB RAM): train 40,131 nodes x 411 features (26 binary
  0/1, 327 quantile-normal, 50 NaN-pattern indicators, 8 log candidate counts), 618,531 edges (139
  isolated nodes); test 40,101 x 411, 606,349 edges. colab_bundle.zip 39.6 MB (x as float16).
  Checks: hash/label order = load_split; folds identical to src.evaluate seed 0 (250 positives per
  fold, every cv group in one fold); edge wallet relations and time directions verified; narrow-window
  filter reproduces 117,989 edges; best single node feature AP 0.41 (funding.novel_gap2_min), i.e. no
  label leak.
- CPU smoke test (torch 2.14.1+cpu in a scratch dir, NOT the project venv): `--smoke` (3,000 nodes per
  split, 2 epochs) runs CV + test end to end in 8.7 s, 0.74 GB RSS; CSVs well formed.
- CPU learning sanity check (first 10,000 train nodes = ~1/4 of the week, 198k edges, 286 positives,
  hidden 32, 1 seed, max 150 epochs, outer folds = bundle folds restricted to the subset):
  OOF AP 49.2, top-1% precision 0.76 / recall 0.27, AP_easy 68.2, medium 3.7, hard 0.8;
  best epochs 15-35 (fast convergence: full batch, lr 2e-3); ~0.45 s/epoch on 3 CPU threads; 1.24 GB RSS.
- Same 10k subset, same folds, LightGBM (project defaults, 9 families, 1 seed): AP 63.6 vs GNN 49.2;
  rank corr 0.455; rank blends w_gnn 0.2/0.3/0.5 -> 63.8/63.1/61.1. A pure-feature GNN is far behind the
  trees; next: optional nested-OOF stage-1 LightGBM score as an extra node input (GNN as learned stage 2).
- Export with stage 1 (XRPL_NJOBS=3, 36 LightGBM fits on a loaded machine): 760 s, 1.42 GB RSS;
  colab_bundle.zip 40.7 MB. stage1.npz `oof` AP = 59.8 = the team's 9-family 1-seed stage-1 CV, so the
  nested scores reproduce src.evaluate / src.stack exactly.
- 10k subset, stage-1 score as 2 extra INPUT columns only (hidden 32, 1 seed): OOF AP 44.6 (< pure 49.2),
  while stage 1 alone on the same nodes = 64.6; rank corr GNN vs stage 1 0.45 -> the net ignores the
  input (early stop at 10-30 full-batch steps). Blends with stage 1: input-variant never helps; pure GNN
  w=0.1 -> 65.3 (+0.7), w=0.2 64.9. Fix: residual stage 2 - logit = a*logit(s1) + b + GNN correction,
  correction head zero-initialised (starts exactly at stage 1), s1 still an input so neighbours share it.
