# Leaderboard log (test split, site score = 100 x AP)

| Date | Run | CV AP (train, grouped) | Site score | Top-1% caught | Notes |
|---|---|---|---|---|---|
| 2026-10-24 | v1_7fam | 59.4 | 71.0 | easy 40% of 800, medium 10% of 225, hard 0%, ambiguous 0% | starter funding relays fields motifs hops amounts, LightGBM 3 seeds |
| 2026-10-24 | v2a_8fam | 60.1 | 71.3 | P@1% 96%, R@1% 31%; easy 50%, medium 10%, hard 0%, amb 0% | + history |
| 2026-10-24 | v2b_8fam_stack | 61.6 (nested) | **72.2** | P@1% 94%, R@1% 30%; easy 50%, medium 10%, hard 0%, amb 0% | v2a + stage-2 neighbour score sharing |

Reading the site's breakdown: top 1% = ~401 tx, so at most ~32% of the 1,250 positives fit; 40% easy +
10% medium ~= 342 hits -> top-1% precision ~0.85. Site score > CV: test seems at least as easy as train for
our features; CV remains the ranking signal for choosing changes.
