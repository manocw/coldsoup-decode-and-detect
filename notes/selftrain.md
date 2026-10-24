# selftrain: transductive self-training / pseudo-labelling (stage 1)

Module: src/selftrain.py. Idea: score the target split with the plain stage-1 model, add its most
confident payments as pseudo-labels (top fraction positive, bottom fraction negative), retrain, re-score.
Optionally also pseudo-label the unlabelled validation split.

CV simulation (`python -m src.selftrain`): outer StratifiedGroupKFold(5) over src.evaluate.cv_groups;
each held-out fold is the transductive target (its labels never used except to score at the end);
baseline = plain stage 1 with the same folds and seeds. Caveat: the target fold shares wallets and the
week with the training folds, unlike the real test.

## Log
- Setup: 9 families (353 feats), LightGBM defaults, 1 model seed = outer-fold seed 0. One fit ~23 s at
  XRPL_NJOBS=3; a 4-setting run = 30 fits ~12 min (run in two chunks with --folds/--save, merged by --report).
- SEED 0 (folds of seed 0; baseline identical to ablate's 59.8):
  | setting | AP | easy | medium | hard | relay | dAP |
  |---|---|---|---|---|---|---|
  | baseline (plain stage 1) | 59.8 | 76.9 | 6.4 | 1.133 | 2.0 | - |
  | hard_r1: target top 0.5% pos / bottom 50% neg, 1 round | 60.3 | 77.3 | 6.8 | 1.139 | 2.0 | +0.53 |
  | wide_r2: target top 1.5% / bottom 50%, 2 rounds | 61.3 | 78.8 | 7.1 | 1.149 | 2.0 | +1.52 |
  | hard_r1+val: hard_r1 + validation pseudo-labelled same way | 61.0 | 78.6 | 6.9 | 1.099 | 1.9 | +1.21 |
  | val_only: only validation pseudo-labelled (target untouched) | 61.0 | 78.7 | 6.5 | 1.132 | 1.9 | +1.26 |
  Pseudo-pos precision in target fold: 0.815 (top 0.5%, 40/fold), 0.80 (top 1.5%, 120/fold); ~12 of
  250 fold positives fall in the bottom-50% pseudo-negatives. Gain is mostly easy. Needs a 2nd seed
  (1-seed noise ~+-0.5 AP).
- SEED 1 (other folds + model seed):
  | setting | AP | easy | medium | hard | relay | dAP |
  |---|---|---|---|---|---|---|
  | baseline | 60.1 | 77.6 | 6.2 | 1.143 | 2.0 | - |
  | hard_r1 | 59.8 | 77.3 | 5.9 | 1.160 | 2.0 | -0.35 |
  | wide_r2 | 60.6 | 78.5 | 6.0 | 1.154 | 2.0 | +0.48 |
  | hard_r1+val | 59.9 | 77.8 | 5.8 | 1.125 | 1.9 | -0.24 |
  | val_only | 60.5 | 78.3 | 6.3 | 1.117 | 1.9 | +0.37 |
  Two-seed mean dAP: hard_r1 +0.09, wide_r2 +1.00, hard_r1+val +0.49, val_only +0.82. Only wide_r2 and
  val_only are positive on both seeds; gain is in easy (+1.0..1.9), medium/hard/relay ~unchanged.
