# review: independent code review of feature families (no model runs)

Started 2026-10-24 ~12:30. Modules: starter funding relays fields motifs hops amounts.
Cache hashes match current source (starter 1c8a, funding d3b1, relays dc70, fields fd5c,
motifs 9d6e, hops 1fc0, amounts 5300). Validation cache only has amounts/fields/relays.

## Findings (appended as I go)

### Raw split stats (load_split, one split at a time)
- All splits: 700 wallets, 676 senders, 24 receive-only wallets, 5 assets, 8 LPs, fee uniform 10/12/15,
  last_ledger_offset always 20, SEQUENCE contiguous from 1 per sender, max 2 same-sender txs per ledger.
- Conversion mix shifts: n_legs>1 train 41.0% / val 52.6% / test 50.5%; 3-leg 9.9/12.5/12.2%;
  dst XRP 32.2/35.9/36.4%; dst GBP 19.8/14.9/15.4%. Train is the outlier, val ~ test.
- Amounts: dst_usd median train 59.4 / val 68.4 / test 50.0; q10 10.4/10.7/8.1; max 4576/9763/9593
  (test+val have payments ~2x above train max).
- Volume: train 40,131 / val 33,862 / test 40,101; tx/hour cv 0.70/0.65/0.68. Start weekday Tue/Thu/Sun.
- Pair reuse: novel-pair frac 35.4/32.8/35.5%; pair>=5 24.1/18.8/22.8%.
- Intermediaries in every labelled family pay out in exactly the asset they received (train, 100%),
  so asset-keyed hub matching in relays.py is structurally sound. No module uses is_conversion /
  n_legs / asset identity directly (good: conversion mix is the clearest batch shift).
- In train, suspicious stack conv rate 0.06 vs legit stack 0.26; chain 0.33 vs 0.47 -> do NOT add
  conversion flags as features (they would memorise a train-only mix).
- Cached feature column order identical across splits; predict.py also reindexes Xte[Xtr.columns].
- No inf in any cached column. Marginal train-vs-test KS is small everywhere (max 0.145, motifs.*_amt_cv).
- Validation (not test) has a volume shift (33.9k tx, fewer per wallet): motifs.*_ncomp KS 0.34-0.46,
  funding.net2_3h 0.20, starter.sender/receiver_payments 0.15-0.20. Test volume == train, so these
  are fine on test but are volume-sensitive absolute counts (fragile to any volume variation).
- AMOUNTS are the clearest test batch shift: std(log dst_usd) train 1.36 / val 1.38 / test 1.47;
  share < $20: 22.0% / 20.4% / 27.3%. The extra dispersion sits in novel pairs (std 1.31->1.47) and
  2-4x pairs (1.31->1.52), i.e. the planted/relay populations, NOT in 5+x commerce pairs (1.09->1.04).
  Train suspicious are rarely < $10 (2.5% vs 9.4% legit). => absolute-amount features (starter.log_value,
  motifs.*_log_amt_med, *_amt_cv) are the main drift risk; ratio/retention features are safer.
- Row alignment verified on all 3 splits (log_value, pair_total, edge_n, novel_pair recomputed from p match cache).
- No within-scenario hop in train has dt==0 (same ledger), so strict (t_i, ...] windows in hops/funding/amounts lose nothing.
- Relay hub in->out lag histogram (bursts, same wallet-asset) nearly identical train/val/test; 2-4x edge
  repeat gap median 57.7h / 51.4h / 62.4h. relays LAGS constants transfer.
- Fast novel-pair motifs: components (novel-only, link<=2m, size>=5) train 83 comps/550 tx (prec 0.82),
  val 68/431, test 61/390 (-29%); at link<=5m train 133/965, test 129/929 (same). Inner gaps similar.
  => maybe slightly slower easy motifs in test; features with hard 2-5 min cut-offs are most exposed.
- BUG (hops): r_out_after_{1h,6h} = log1p(r_out) - log(vin) and s_in_before = log1p(s_in) - log(vout) mix
  log1p/log. When there is NO onward flow the value is -log(vin), which is > -0.7 (looks like >=50%
  pass-on) whenever vin < ~$2. Such fake pass-on rows: train 146 (3.6/1k), test 303 (7.6/1k) for 1h,
  because test has 4x more sub-$1 payments. Fix: log1p on both sides, or NaN/indicator when r_out == 0.
- Load-percentile artifact (fields.load_*) persists in test: fast novel motif comps mean load_1h pct
  train 0.361 / val 0.360 / test 0.366 (train suspicious 0.367 vs legit 0.504).
- Redundant: relays.edge_n == fields.pair_total exactly; hops.s_in_before_6h ~= -amounts.b_vs_in
  (spearman 0.999); hops.r_out_after_6h ~ amounts.f_vs_out (0.998).
- Relay mix shift: on 2-4x recurring edges conversion rate train 0.374 / val 0.564 / test 0.537 and
  relays.*_relay_is_chain mean 0.406 / 0.321 / 0.313; onward-hop lag mass in the distributed band
  (1.5-3h) 0.165 -> 0.147, aggregate band (3.5-5.5h) 0.103 -> 0.117. Bands themselves unchanged, so
  the LAGS constants hold; test simply has relatively more aggregate-like (single hub) relay structure.
  In train aggregate relays convert 61% vs distributed 13% -> any conversion-aware feature would break.
- Wallet reuse: 323 of 535 wallets touching suspicious txs host >=2 suspicious scenarios (max 7);
  per-receiver suspicious rate up to 0.52. Week-level per-wallet aggregates (starter.*_payments,
  *_pass_through, motifs.*_ncomp, relays.*_sess_k_*, hops.trail_*_deg) can fingerprint wallets and
  learn per-wallet rates across scenario-grouped folds: CV inflation that cannot transfer (wallets disjoint).
- motifs s120 scale is saturated by CAP=30: 49% of defined (novel) rows sit in comps of size >=28;
  purity (share of comp from the row's own scenario) for suspicious medium 0.31, easy 0.43 (s30: 0.88).
  s120_* columns describe cap-truncated blobs whose make-up depends on novel-payment density and amount
  mix -> they top the train->test KS list (amt_cv 0.145, max_out_share 0.10, inter_frac 0.087).
- No module references labels, ids, hashes, addresses, absolute ts, tx_index, fee, LPs or conversion
  fields (grep + read). ledger used only as relative windows (fields.load), sequence only as tiebreak.
- Unbounded linear ratios explode more on test (more tiny payments): funding.s_split_ratio >100:
  train 370 / test 664 (max 934 -> 3252); motifs.s5_retention >10: 651 -> 933; s30_retention >10:
  366 -> 488; hops.par_lr_1h > 3: 29 -> 61. Use log ratios with a scale-aware floor and clip.
- Dense fan structures are rarer in test (mostly legit look-alikes): novel fan-in >=6 senders in +-30m
  19.4/1k -> 9.6/1k (train prec 0.15), fan-out 20.6 -> 12.9; bipartite copayers>=2 55.9 -> 37.8/1k.
  The near-pure fast signal (>=6 in +-5m, train prec 0.95-0.99) holds: 1.9 -> 1.3 and 1.8 -> 1.6 /1k.
  => the NEGATIVE look-alike population shifts too; expect a different false-positive mix on test.

## Verdicts
starter keep_with_changes (log_value absolute; *_payments volume/fingerprint) | funding keep_with_changes
(s_split_ratio log+clip; counts volume-sensitive) | relays keep_with_changes (+$1 smoothing, TOL fit
on 12 positives; drift small) | fields keep (drop pair_total dup) | motifs keep_with_changes (drop or
re-cap s120, drop *_log_amt_med/*_amt_cv or rank them, log retention, normalise *_ncomp) | hops
keep_with_changes (log1p/log bug; drop near-dup flows) | amounts keep.
