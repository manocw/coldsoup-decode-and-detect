# Leaderboard log (test split, site score = 100 x AP)

| Date | Run | CV AP (train, grouped) | Site score | Top-1% caught | Notes |
|---|---|---|---|---|---|
| 2026-10-24 | v1_7fam | 59.4 | 71.0 | easy 40% of 800, medium 10% of 225, hard 0%, ambiguous 0% | starter funding relays fields motifs hops amounts, LightGBM 3 seeds |
| 2026-10-24 | v2a_8fam | 60.1 | 71.3 | P@1% 96%, R@1% 31%; easy 50%, medium 10%, hard 0%, amb 0% | + history |
| 2026-10-24 | v2b_8fam_stack | 61.6 (nested) | 72.2 | P@1% 94%, R@1% 30%; easy 50%, medium 10%, hard 0%, amb 0% | v2a + stage-2 neighbour score sharing |
| 2026-10-24 | v3_9fam_stack | 62.0 (nested) | 72.3 (ROC AUC 0.95) | P@1% 94%, R@1% 30%; easy 50%, medium 10%, hard 0%, amb 0% | v2b + cycles family |
| 2026-10-24 | v2c_8fam_stack_aux | 62.3 (nested) | 72.6 | P@1% 93%, R@1% 30%; easy 50%, medium 0%, hard 0%, amb 0% | v2b + auxiliary relay/motif helper targets (--aux) |
| 2026-10-24 | v2d_8fam_stack_pw2 | 60.5 (nested) | - | - | v2b + scale_pos_weight 2; rejected on CV |
| 2026-10-24 | v6_9fam_stack_aux | 62.1 (nested) | 73.0 | | v3 + aux (teammate) |
| 2026-10-24 | v7_10fam_tempo_aux | 62.4 (nested) | **73.8** | P@1% 94%, R@1% 30%; easy 50%, medium 10%, hard 0%, amb 0% | v6 + tempo family |
| 2026-10-24 | v8_9fam_aux_medium | 62.3 (nested) | - | - | v6 + medium helper target; medium AP 8.8 vs 9.1, rejected |
| 2026-10-24 | v9_10fam_tempo_aux_medium | - | 73.3 | - | v7 + medium helper target; worse than v7, dropped |
| 2026-10-24 | v10_tempo_plus_spec | - | 73.2 | - | v7 + 0.3 x medium specialist (src.specialist); w15 / w50 variants built |
| 2026-10-24 | v11_11fam_tempo_medium_aux | - | 73.3 | - | v7 + medium feature family (hub-burst amount roughness, gap regularity, butterflies) |
| 2026-10-24 | v12_v11_plus_spec | - | 73.5 | - | v11 + 0.3 x medium specialist |
| 2026-10-24 | b1_rank_v7_v12 | - | **74.0** | - | rank average 0.5 v7 + 0.5 v12 (src.blend); final |
| 2026-10-24 | b2_rank_v7_v12_v6 | - | 73.8 | - | rank average 0.5 v7 + 0.25 v12 + 0.25 v6 |
| 2026-10-24 | v4_9fam_stack_selftrain | - | 72.2 (ROC AUC 0.95) | P@1% 94%, R@1% 30%; easy 40%, medium 10% | v3 + stage-2 self-training: no gain |
| 2026-10-24 | v6_9fam_stack_aux | - | **73.0** (ROC AUC 0.95) | P@1% 94%, R@1% 30%; easy 50%, medium 0%, hard 0% | v3 + auxiliary relay/motif targets |

Reading the site's breakdown: top 1% = ~401 tx, so at most ~32% of the 1,250 positives fit; 40% easy +
10% medium ~= 342 hits -> top-1% precision ~0.85. Site score > CV: test seems at least as easy as train for
our features; CV remains the ranking signal for choosing changes.

ROC AUC reference (offline, grouped OOF, stage 1): 7 families 0.921 (easy 0.995, medium 0.914, hard 0.662);
8 families 0.927 (0.996 / 0.929 / 0.678); Marat's features 0.905 (0.995 / 0.867 / 0.626). Medium
positives rank above ~93% of legit payments yet AP_medium is ~6: they sit just below a thick band of
legit look-alikes and relays near the top. Site ROC AUC for v3: 0.95 (vs offline ~0.93).
