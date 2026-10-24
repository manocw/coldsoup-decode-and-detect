# cycles feature family - notes

Scratch: /tmp/claude-1000/-home-peroxide-Downloads-OpenCode-xrpl-hack/42315158-3b36-488f-933b-da88881ad637/scratchpad/cycles/ (cyc_v1..v4.py, ev3.py harness, abl*.txt)

## Recovered from the crashed run (scratch ev3.py harness: starter + subset, 1 seed, grouped CV)
- starter alone: AP 33.7 (AP_medium 1.35)
- starter + cycle counts only (n_15m,n_2h,n_6h): 36.8; + n_12h,n_3d: 37.2; + cyc_ attrs: 38.3
- starter + pair_count only: 40.3 (pair recurrence is a big chunk)
- starter + pair_count + n_* + cyc_* + vc_*: 45.1 (AP_easy 59.4, AP_medium 3.0)
- + vcx_score (best value-consistent cycle: retention minus slowness): 45.7
- longer-window plain counts (12h K3..6, 24h, 3d, 7d K2/3) add nothing beyond n_12h/n_3d.
- dropping cyc_hop*/vc_hop* slightly better (45.1 vs 44.6) - noise level.

## Module v1 saved: src/features/cycles.py (34 cols; build ~4 s/split)
- `python -m src.ablate starter cycles --families` (1 seed): starter+cycles AP 44.3 (easy 58.2, medium 2.8,
  hard 0.84, relay 1.16); cycles alone 13.7. cycle-easy median rank 0.002, cycle-medium 0.070.
- `--only` all 7 + cycles (1 seed): 59.3 vs all 7 alone 59.0 in a separate run (other agents' caches drift, so
  only same-run comparisons count). cycle-easy rank 0.008 both ways -> other families already catch fast cycles
  (motifs s5_*, hops trail_*).
- Same-run, 2 seeds (scratch ev4.py, base = all 7): none 59.51 / +module 59.88 / +compact 17 cols 59.68.
  AP_medium 5.93 -> 6.11; random-medium median rank 0.248 -> 0.211; cycle-medium 0.058 -> 0.054.
  => marginal gain ~+0.3 AP, at noise level.

## Why the ceiling is low
- Only cycle (29 scenarios: 10 easy, 2 medium, 17 legit) and random (35) families contain loops inside a
  scenario; all other motifs are acyclic (checked with networkx per scenario).
- Oracle bound on oof(all7+cycles): legit cycles -> 0: +0.64 AP; cycle medium -> 1: +1.1; all cycle pos -> 1 and
  legit -> 0: +3.9. Random family perfect: +5.2.
- Generator: legit cycles decay monotonically (each hop x0.65-0.85, up-share 0), suspicious cycles random-walk
  near flat (x0.85-1.15, some rising hops). Medium suspicious = same flat amounts but 20-40 min gaps (span ~2.3 h)
  vs legit 1.3 h. Within cycle family AUC: cyc_ret 0.90, cyc_geo 0.89, cyc_up 0.85; medium vs legit: cyc_dur
  0.92, cyc_ret 0.80. One easy positive (C482..) decays like legit (geo -0.17, conversions), so not separable.
- Long-window novel-pair cycles (24h/72h): S rows 0.9 vs L 0.24 in 72h, but this is just pair novelty
  (S rows are 95% first-time pairs vs 31% legit) which fields/motifs already have.
- Cycle features have no signal for medium positives of other families (best AUC 0.55).
- Ambiguous (hard, 25 single txs): recurring pairs (pair_count ~4.7), no cycles - nothing to find.

## Test/validation shift (novel-pair cycles, 24h, K<=14)
- fast (<=15 min) cycles of length 4-8: train 10, validation 11, test 9; 1-3 h length 4-8: 9 / 5 / 3.
  No sign of new cycle shapes (no fast cycles longer than 8) in test.

## Final module (35 cols: + cyc_minlen), official commands, 2 seeds
- `ablate starter cycles --families --seeds 2`: starter+cycles AP 45.5 (easy 60.2, medium 2.9, hard 0.83,
  relay 1.14, top1% prec 0.718); cycles alone AP 14.1. cycle-easy median rank 0.001, cycle-medium 0.057.
- `ablate <all 7> --only --seeds 2`: 59.5 (easy 76.8, medium 5.9, hard 1.13, relay 2.1)
  `ablate <all 7> cycles --only --seeds 2`: 60.0 (easy 77.4, medium 6.1, hard 1.15, relay 2.1)  => +0.5 AP
  (same time window, hops cache hops-930394fb6d9d).
- builds on validation (~5 s) and test (~8 s); train/test feature distributions match (in-6h-cycle share 18.5%
  train vs 16.5% test, validation 10.8% because it is a smaller week).
