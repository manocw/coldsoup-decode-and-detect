# hunt_mixed - medium suspicious vs look-alikes in fan_out_in / multi_stage / random

Scratch: /tmp/claude-1000/-home-peroxide-Downloads-OpenCode-xrpl-hack/42315158-3b36-488f-933b-da88881ad637/scratchpad/hunt_mixed/
Population (train, scenario_id used only for evaluation):
- my families: 9 medium scenarios (fan_out_in 2, multi_stage 4, random 3; 86 payments) vs 54 look-alikes
  (+38 easy). Every check is also repeated on all 9 typology families (27 medium / 140 look / 105 easy)
  to see whether an effect is consistent or a 9-scenario fluke.
- Bar: scenario AUC >= 0.75 AND label-free payment AUC >= 0.65 vs look-alikes.

## Eyeballing (print.py)
- fan_out_in / multi_stage medium are exact templates (k fan-out, k fan-in, (+1 exit)), FIFO forwarding,
  intermediaries pay out in the received asset, same as look-alikes. No extra/cover payments inside.
- Only visible difference = slower steps (already known).

## Log (AUC: sc = scenario mean, p = payment; m = medium vs look, e = easy vs look)

### Batch 1 - static per-payment context (h1.py)
asset home-ness of sender/receiver (share of the wallet's other flows in that asset), amount z-score vs the
wallet's own other sends/receipts, rate-normalised gap to sender's prev/next send and receiver's prev/next
receipt, ledger crowding, tx_index, conversion cost, pair reuse, reverse edge, day of week, ts mod 60.
=> all 0.40-0.62 (all-family medium 0.44-0.57). Null.

### SendMax / Paths (sm.py)
SendMax is ALWAYS first-leg cost x 1.01 (slack 0.0100000 +- 5e-8, every class, background too); Paths
count/length identical by class. FLAGS always 0, no memos/tags. Null (no generator fingerprint there).

### Batch 2 - scenario timing/shape (h2.py, oracle grouping)
gap CV, max/median, min/median, skew, amount CV / spread, n assets, conversion share, start hour, hold
time CV, hold/gap ratio, sequence contiguity of senders: 0.38-0.60. Only the known speed stats move
(gap_med 0.70, hold_med 0.68). gap_cv is LOWER for medium (0.38 all-family) - medium steps are more
regular, but weak.

### Unit retention (h4.py)
Retention in units of the received asset (intermediaries always pay out the received asset):
look median 0.838 (range 0.50-1.155), medium 0.868 (0.59-1.14), easy 0.957 (0.71-1.16). Same as USD
retention (already tested). Null.

### Batch 3 - wallet windows (h5.py)
hold time, activity during hold vs baseline, commingling (same-asset receipts in 2h before a send),
activity/novel share +-30m/2h of each side: all-family medium 0.41-0.60. Mine-only some 0.29-0.33
(=0.67-0.71 inverted) but these include the scenario's own siblings => speed confound.

### Balance trajectory (h6.py, h7.py) - best new signal, near miss
Per wallet-asset running net flow (units) over the week; position of the balance after/before the payment
in the wallet-asset's weekly [min, max] range (label-free).
- sender bp_before: medium 0.670 sc / 0.576 p (all fam), 0.710 / 0.609 (mine). Mean bp_after: medium 0.543
  vs look 0.466, easy 0.462, background 0.483; higher for medium in 8/9 families (stack equal).
- receiver bp_after: 0.604 / 0.539 all, 0.683 / 0.594 mine.
- Variants (headroom / amount, USD-total balance, rank in trajectory) are weaker (0.49-0.58).
- Look-alikes and easy sit at the background level, so this is a medium-only placement artefact if real.
  Below the bar.

### LP / conversion cost / stale rate (h8.py, h9.py, h13.py)
LP choice distribution same for all classes. Relative conversion cost looked lower for planted payments
than background, but within amount bins it is identical (pure fixed-cost-vs-amount effect) - null.
Realised conversion rate = current-hour market rate x (1 + U(0.0001, 0.006)) for every class; no stale-rate
pricing. Null.

### Balance follow-up (h10.py, h11.py, h16.py)
- Excluding the scenario's own payments (oracle) keeps it: sender bp_before 0.697 all / 0.733 mine.
  Time rank of the payment in the wallet-asset history is flat (0.49), so it is not "later in the week".
- Using only recurring-pair flows (pair_n>1) kills it (0.50 all): the effect comes from OTHER novel-pair
  flows of the wallet (other planted motifs / one-offs), not from the wallet's regular commerce.
- Pre/post net-flow windows (6h/24h/week, asset or all-asset): 0.43-0.58 all-family. Diffuse.
- Per family (sender bp, medium vs look): bipartite 0.84, fan_in 0.92, multi_stage 0.83, fan_out_in 0.70,
  random 0.64, cycle 0.59, fan_out 0.50, stack 0.48, chain 0.42 -> inconsistent; with ~60 features
  tested, a max null AUC ~0.64 is expected at n=27/140, so 0.67 is only borderline. NEAR MISS, not a find.

### Batch 4 - wallet traits (h17.py)
hold time / inter-send gap vs the wallet's own typical latency: |dev| medium 0.42 all, 0.26 mine (i.e.
medium is CLOSER to the wallet's normal latency), easy 0.69-0.93 - this is just the speed effect
(slow = close to background latency). Hour-of-day alignment with the wallet's usual hours: 0.45-0.57. Null.

### Batch 5 - ordering (h18.py)
Fan-out amount vs send order: rho ~ +0.35-0.43 in every class (generator sorts amounts loosely upward),
no class difference (0.46). FIFO forwarding (Kendall tau receipt vs forward order): exactly 1.0 for
chain / cycle / fan_out_in / multi_stage in every class. Hold time vs amount: 0.48.
Aside for whoever hunts stack: stack medium tau -0.33 (LIFO-ish) vs look +0.42, AUC 0.04 - but n=3 vs 4.

### Batch 6 - graph embedding (h19.py)
common neighbours of payment endpoints, normalised, directed 2-paths a->c->b, back 2-paths b->c->a,
source-sink direct link outside the scenario, source-sink common neighbours: 0.42-0.58. Null.

### Batch 7 - timing grid & metadata (h12.py, h20.py, h21.py)
Sequence strictly follows time for every sender, consecutive (+1) always, first seq 1. ts on a 4 s grid;
gaps / ts mod 60/300/600/3600 uniform in every class. DATE_CREATED/UPDATED == DATETIME, all tesSUCCESS,
FLAGS 0. Scenario start times uniform over the week (KS p=0.86 medium vs look). Null.

## Conclusion
No hypothesis clears the bar for fan_out_in / multi_stage / random. Inside these families a medium
scenario is the same template as its look-alike (same k, FIFO, same-asset forwarding, same retention
distribution, same amounts/assets/conversion mix, same wallet pool, same placement in the week); the only
generator knob that differs is step timing (already known). Best new near-miss: balance position of the
wallet-asset at payment time (0.67-0.73 scenario, 0.58-0.64 payment), driven by other novel-pair flows
and inconsistent across families. Recommend NOT building features from it; if anything, a cheap
`bp_before` (sender + receiver, per wallet-asset running net flow position) could be ablated once.
