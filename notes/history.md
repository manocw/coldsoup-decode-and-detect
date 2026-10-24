# history feature family (whole-week wallet history and deviation)

Module: src/features/history.py. Overlap check: fields already has pair_seq/pair_total/rev_pair_total,
amount vs sender/receiver median, prev/next gaps; relays has edge_n, edge_mingap_h, edge_span_h;
funding has one-off (novel) pair counts in windows. history focuses on wallet roles from whole-week
profiles and deviation of a tx from the wallet's own history (hour-of-day, asset, rate, percentiles).

## Log
- Exploration (train, labels in scratch only): wallets in suspicious and look-alike motif scenarios
  come from the same pool (wallet-level corr of scenario counts ~0.05); no per-wallet hour-of-day
  profile (entropy = shuffled); per-sender asset concentration is real (top asset share 0.49 vs 0.37
  shuffled) because of scheduled pairs. Background = 5,044 one-off pairs + scheduled pairs with
  3-7 payments at a fixed period (gap CV ~0.01, amount ~constant) + a few irregular busy pairs.
  ambiguous/hard (25 single-tx scenarios) mostly sit INSIDE scheduled background series with the
  same amount and period as their legit siblings -> essentially indistinguishable.
- Medium suspicious motifs span ~1.8 h (median) vs look-alikes ~1.0 h; univariate AUC of all
  history v1 features for medium-vs-look-alike <= 0.57 (no single strong signal).
- v1 (48 feats: pair regularity, wallet roles, amount/asset percentiles, bursts, common cps), 1 seed:
  starter+history AP=49.3 (easy 63.0, medium 6.4, hard 0.75, relay 1.05); history alone AP=35.1
  (medium 8.3). Runtime: build 0.4 s, ablation 25 s.
- All 7 families, 2 seeds: AP=59.4 (easy 76.7, med 6.0, hard 1.11, relay 2.0).
  All 7 + history v1, 2 seeds: AP=59.8 (easy 76.9, med 6.5, hard 1.15, relay 2.0) -> +0.4 AP, +0.5 AP_medium.
  (1 seed: 59.0 -> 59.8.)
- Hard-negative analysis (all-7 OOF): the median medium positive ranks ~1,490th among negatives; negatives
  above the 20th-pct medium score = 4,938: relays 3,184 (519 of the high ones are one-off pairs),
  background 832, motif look-alikes ~920. Scenario-level, medium vs look-alike differs only in speed
  (mean gap AUC 0.69, slower) and slightly larger amounts (0.58); wallet-history stats give <= 0.6.
- Relay-round density (txs on 2-4x recurring, >10h-gap edges within +-1h): AUC 0.69 medium vs hard
  negatives, but fields.load_1m already gives 0.69 (same information: cohort rounds = load spikes).
- Leave-one-group-out (all 7 + history, 2 seeds; full v1 = 59.8): -pair 60.3, -role 59.4, -dev 60.4,
  -burst 60.9, -familiarity 60.4. role-only 60.1, role+dev 60.5, role+pair 60.3, role+fam 59.9.
  => wallet roles carry the gain; local burst ratios hurt (redundant with funding/starter windows).
- On top of role+pair+dev+fam (60.9), none of these helped (2 seeds): wallet temporal profile
  (Goh-Barabasi burstiness, sessions, relay-like share, conversion share, assets, pass-through) 60.2;
  counterparty roles 60.7; slow-scale one-off excess 6h/12h + relay-round density 60.4;
  wallet-rate-normalised gaps 60.5. Noise between 2-seed runs is ~+-0.4 AP.
- Fan-out/fan-in "episode recurrence" (how many separate times in the week the wallet shows the same
  or larger fan within +-0.5/1.5 h; relay hubs repeat per round, motif wallets should not): univariate
  AUC <= 0.58 vs relay hard negatives, wrong sign for the hypothesis. Dropped.
- KEY DIAGNOSTIC (all-7 OOF, rank among negatives): medium scenarios median rank 1,878 vs legit
  look-alike scenarios 1,115 and easy 114. The model learned "fast = suspicious" from easy motifs, and
  medium suspicious motifs are SLOWER than the look-alikes (span 1.84 h vs 1.03 h), so they rank below
  their own look-alikes. Non-monotone speed effect with only 29 medium scenarios: a modelling issue
  (weighting / separate medium head), not something wallet-history features fix.
- Sample weight x4 on medium positives (all 7 + history minus bursts, 2 seeds): AP 60.9 -> 61.0, AP_medium 7.1 -> 7.5 (small; modelling-side lever).
- v2 = v1 minus the 6 burst features, plus a bug fix (src/dst asset codes now share one code space,
  affects s_asset_share_in / r_asset_share_out). 42 features.
  3 seeds (harness, before the asset fix): all-7 59.5 (med 6.0) -> +history(no bursts) 60.8 (med 6.9);
  without common_cp also 60.8 (med 6.7).
  v2 standalone, 1 seed: starter+history AP=39.7 (easy 51.5, med 2.8); history alone 10.4 -> the
  bursts were the standalone signal (v1: 49.3 / 35.1) but are redundant once funding/starter are in.
- v2 official (ablate --only --seeds 2): all 7 + history AP=61.2 (easy 78.4, med 6.9, hard 1.21, relay 2.1) vs all 7 59.4 (76.7 / 6.0 / 1.11 / 2.0) => +1.8 AP.
