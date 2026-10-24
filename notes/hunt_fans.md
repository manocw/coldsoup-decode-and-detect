# hunt_fans - fan_out / fan_in / bipartite: suspicious (medium) vs legit look-alikes

Scratch: /tmp/claude-1000/-home-peroxide-Downloads-OpenCode-xrpl-hack/42315158-3b36-488f-933b-da88881ad637/scratchpad/hunt_fans/
Population (train): medium 10 scenarios (fan_out 5, fan_in 3, bipartite 2; 65 payments), easy 39,
look-alike (normal) 43 scenarios (303 payments). Very small n: medium AUC SE ~0.09.

## Log
- Read all 10 medium + 43 look-alike scenarios side by side (show.py).
- Null (payment-level, fans medium vs look-alike): ledger size, tx_index position, last-in-ledger,
  ts mod 60, SendMax/Paths flags (identical given conversion): AUC 0.49-0.53.
- Null / inconsistent (scenario AUC medium vs look-alike, fans): wallet-relative asset affinity
  (sender pays in its usual asset, receiver usually spends what it gets), conversion affinity,
  per-wallet hour-of-day affinity (all 0.42-0.57, no family-consistent direction); conv_cost
  (fan_out 0.64 / fan_in 0.31: opposite); amount dispersion sd(log amt) (fan_in 0.26, others ~0.5);
  max share / last-is-max / first-is-min (0.40-0.63); gap-amount coupling (0.40); amount skew (0.41);
  sequence consecutiveness of the hub (confounded with span).
- Amount relative to the wallets' weekly median (rel_snd 0.76 / payment 0.69): medium fans are
  bigger, but wallets have NO amount scale (background between-sender var of means 0.17 of 1.19;
  repeated PAIRS have a fixed amount, within-pair sd 0.047; novel-pair amounts uncorrelated with the
  wallets' medians, rho 0.03). So this is plain absolute size (raw log amt payment AUC 0.685), only
  in fans (all-family 0.58). Not new.

## Hit 1 - amount ROUGHNESS (order randomness) inside the hub burst  [PASSES]
Look-alike (and easy fan_out) amounts form a SMOOTH sequence in time (fan_out: nearly sorted
ascending, the largest payment last; fan_in/bipartite: positively autocorrelated), medium amounts
come in RANDOM order. Statistic: von Neumann ratio VN = sum (x[i+1]-x[i])^2 / sum (x[i]-mean)^2,
x = log dst_usd in time order (iid ~2, smooth << 2). Not the same as "rising hops share" (fans
up_share 0.45) or trend (Spearman rho fans 0.66; fan_in has no trend but VN still separates).
- Scenario level (true grouping, per hub >= 3 payments): fans AUC 0.78 (global order 0.80);
  fan_out 0.86, fan_in 0.74, bipartite 0.75 (n=2). Medians VN medium/look-alike: fan_out 2.78/0.90,
  fan_in 2.46/1.96, bipartite 1.96/1.64. Other families: multi_stage 0.83, cycle 0.68, chain 0.58,
  but fan_out_in 0.43, stack 0.33, random 0.12-0.36 (tiny n).
- Label-free payment level (recipe below, G=60m, pair_n<=2): fans 0.820 (lag-1 autocorr 0.848
  inverted), fan_out 1.00, fan_in 0.79, bipartite 0.62; also multi_stage 0.86, random 0.77,
  fan_out_in 0.44, stack 0.54. Stable over G 30/45/60/90 min and pair_n<=1/2 (fans 0.795-0.829).
  Scenario mean of the label-free value: AUC 0.90, Mann-Whitney p=1e-4 (10 vs 42).
- Caveat: fan_out's perfect 1.00 is partly luck - BE59 is smooth under true grouping (VN 0.71) but a
  1.77 USD background payment lands in its label-free burst and makes it rough.
- vs all negatives that sit in a burst >= 3 (mostly relays): 0.63-0.66 only. Low VN marks the
  structured legit (and easy fan_out) fans; it is a look-alike discriminator, needs a tree to combine
  with speed/fan shape. Easy vs look-alike: 0.54.
- Incremental: among look-alikes VN is uncorrelated with speed (rho 0.14 with log median gap) and with
  gap regularity (0.01). Grouped-CV logistic (StratifiedGroupKFold by scenario, 20 seeds), medium vs
  look-alike fan payments: model OOF score alone 0.62, score+VN 0.845, speed alone 0.59,
  speed+VN 0.74.
- Split shift: bursts >=4 per 1k tx 30.6/28.6/33.4 (train/val/test), VN q10/50/90 identical
  (0.96/1.93/2.86 vs 0.94/1.95/2.95), share VN<1.2 0.185/0.178/0.190.

## Hit 2 - REGULAR gaps (scale-free), independent of speed  [passes at scenario level, payment level
needs the leave-one-out robust version]
Medium fans are emitted with near-constant spacing (each scenario its own base gap, jitter ~+-20%,
e.g. 54/44/53 min, 15/12/16, 25/25/20), look-alike gaps look exponential (Poisson). Pooled within-scenario
gaps: medium fan_out 3.7-54 min, CV 0.65 vs look-alike 0.1-87 min, CV 1.09.
- Scenario level: mean |diff log gap| fans 0.80 (fan_out 0.84, fan_in 0.97, bipartite 0.47 - bipartite
  medium is regular only GLOBALLY across hubs, e.g. 98E5 global gaps 11,7,9,7,8,5,7,8); CV fans 0.82,
  random 0.81, cycle 0.71, chain 0.62; not stack / multi_stage / fan_out_in.
- Label-free payment level (same bursts): plain mean|dlog gap| 0.645, CV 0.642; robust (bursts>=5:
  min over leave-one-payment-out, removes one contaminant) 0.716 (G=60m; 0.64-0.76 over G/pair_n
  settings, so less stable than VN). Mostly fan_in (0.97) and fan_out (0.63); bipartite 0.52.
- Uncorrelated with median gap among look-alikes (rho -0.02). Grouped CV: speed 0.59 -> speed+reg
  0.67; score 0.62 -> score+reg 0.775; speed+reg+VN 0.77.
- Split shift: q10/50/90 0.39/1.05/2.02 train vs 0.39/1.05/2.04 test.

## Near misses
- Amount trend (Spearman rho amount vs time in hub burst): fan_out scenario 0.84 / payment 0.91, but
  fans 0.66 / 0.70 (fan_in no trend in any class). Subsumed by VN.
- Absolute / wallet-relative size (see above): fans 0.73-0.76 scenario, 0.69-0.72 payment; not new.
- Amount dispersion sd(log) fan_in only (0.74 inverted).

## Label-free recipe (both hits)
1. pair_n = payments of (sender, receiver) in the split; keep pair_n <= 2 (novel-ish edges).
2. Two streams: (sender, src_asset) and (receiver, dst_asset) - the hub's own asset is constant
   inside every fan/bipartite scenario (100%). Sort by ts; new burst when gap > 60 min.
3. For bursts with >= 3 payments: VN of log(dst_usd) in time order; reg = mean |diff log(gap_min+0.25)|
   (bursts >= 5: min over leave-one-payment-out).
4. Each payment takes the values of its larger burst (sender side on ties); NaN if no burst >= 3
   (coverage: fans 89-98%, background 2.2%, relays 38%).
Code: scratch burstfeat.py (~15 s per split). Estimated ~20-30 min to port as src/features/<family>.py.
