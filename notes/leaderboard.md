# Leaderboard log (test split, site score = 100 x AP)

| Date | Run | CV AP (train, grouped) | Site score | Top-1% caught | Notes |
|---|---|---|---|---|---|
| 2026-10-24 | v1_7fam | 59.4 | 71.0 | easy 40% of 800, medium 10% of 225, hard 0%, ambiguous 0% | starter funding relays fields motifs hops amounts, LightGBM 3 seeds |
| 2026-10-24 | v2a_8fam | 60.1 | 71.3 | P@1% 96%, R@1% 31%; easy 50%, medium 10%, hard 0%, amb 0% | + history |
| 2026-10-24 | v2b_8fam_stack | 61.6 (nested) | 72.2 | P@1% 94%, R@1% 30%; easy 50%, medium 10%, hard 0%, amb 0% | v2a + stage-2 neighbour score sharing |
| 2026-10-24 | v3_9fam_stack | 62.0 (nested) | 72.3 (ROC AUC 0.95) | P@1% 94%, R@1% 30%; easy 50%, medium 10%, hard 0%, amb 0% | v2b + cycles family |
| 2026-10-24 | v4_9fam_stack_selftrain | - | 72.2 (ROC AUC 0.95) | P@1% 94%, R@1% 30%; easy 40%, medium 10% | v3 + stage-2 self-training: no gain |
| 2026-10-24 | v6_9fam_stack_aux | - | **73.0** (ROC AUC 0.95) | P@1% 94%, R@1% 30%; easy 50%, medium 0%, hard 0% | v3 + auxiliary relay/motif targets |

Reading the site's breakdown: top 1% = ~401 tx, so at most ~32% of the 1,250 positives fit; 40% easy +
10% medium ~= 342 hits -> top-1% precision ~0.85. Site score > CV: test seems at least as easy as train for
our features; CV remains the ranking signal for choosing changes.

ROC AUC reference (offline, grouped OOF, stage 1): 7 families 0.921 (easy 0.995, medium 0.914, hard 0.662);
8 families 0.927 (0.996 / 0.929 / 0.678); Marat's features 0.905 (0.995 / 0.867 / 0.626). Medium
positives rank above ~93% of legit payments yet AP_medium is ~6: they sit just below a thick band of
legit look-alikes and relays near the top. Site ROC AUC for v3: 0.95 (vs offline ~0.93).
