# hunt_handoff - chain / cycle / stack: suspicious vs legit look-alikes

Scratch: /tmp/claude-1000/-home-peroxide-Downloads-OpenCode-xrpl-hack/42315158-3b36-488f-933b-da88881ad637/scratchpad/hunt_handoff/
(base.py -> train.parquet; show.py side-by-side scenario dumps chain/cycle/stack.txt; ev.py/ev2.py AUC harness;
feat1-3.py payment features; orc1-3.py oracle scenario stats; bf.py label-free butterfly + split check)

Population (train): chain 12 easy / 2 medium / 12 look-alike scenarios; cycle 10 / 2 / 17; stack 6 / 4 / 14.
Only 8 medium scenarios in these families -> AUC SE ~0.1-0.15; I checked consistency on all typology families.
Runtime: every script < 1 min on 40k rows.

## PASS (stack only): one-off butterflies over long windows = stack middle-layer width
- Generator fact (oracle): every stack is 2 sources -> m middles -> 2 sinks, complete bipartite in both layers.
  Suspicious m = {1:1, 2:2, 3:4, 4:3} scenarios; look-alikes m = {1:8, 2:2, 3:2, 4:2}.
  AUC(m) 0.75, Mann-Whitney p = 0.035. All 4 medium stacks have m >= 2; 8/14 look-alikes are 2->1->2.
- Label-free recipe (bf.py): keep one-off payments (sender->receiver pair occurs once in the week).
  For one-off u->v at t: V = u's other one-off receivers within +-W, U = v's other one-off senders within +-W;
  bf_W = sum over u2 in U of |one-off receivers of u2 within +-W  ∩  V|. NaN/0 for recurring pairs.
- Results (stack rows; suspicious vs look-alike / medium vs look-alike):
  | W | payment AUC | payment AUC medium | scenario AUC | scenario AUC medium |
  |---|---|---|---|---|
  | 1h | 0.740 | 0.655 | 0.804 | 0.786 |
  | 6h | 0.717 | 0.793 | 0.789 | 0.866 |
  | 12h | 0.709 | 0.799 | 0.818 | 0.911 |
  Pooled over chain+cycle+stack (medium vs look-alike rows): payment 0.767, scenario 0.795 (chain/cycle are all 0).
- Selectivity: bf_6h > 0 on 50/52 stack-medium rows but only 1,096 train rows overall (2.7%): 322 agg relay,
  130 dist relay, 157 background, 145 bipartite look-alike, 58 stack look-alike. The existing
  funding.bipartite_copayers_2h (any pairs, 2h) fires on 45/52 stack-medium rows but also on ~5,500 relay/background
  rows -> the one-off restriction + 6h window is the new part (medium stacks span 2.5-8 h).
- Transfer: bf_6h > 0 on 2.73% train, 1.92% validation, 2.28% test rows; bf >= 4: 209 / 189 / 188. No shift.
- Caveats: one family, 10 positive scenarios; bipartite family has butterflies in both classes (no signal there).

## Near-misses
- Background activity of the motif wallets around each payment, excluding the own scenario (act60m_s_ex):
  chain+cycle medium vs look-alike scenario 0.92 / payment 0.78 - but only 4 scenarios, stack 0.41 and all typology
  medium 0.50-0.58 -> not consistent, treat as noise. (Context notes: medium source idle time 2.1 h vs 5.3 h.)
- Stack amount dispersion (CV of payment USD in the scenario): stack 0.71 / medium 0.91, but among stacks with
  >= 2 middles only 0.56 -> mostly the width effect again.
- One-off fan-out of the sender within 6h (fo_6h): chain+cycle medium scenario 0.86, payment 0.65; all typology
  medium 0.59; and the same quantity already exists as funding.s_novel_out_3h.
- Hop log-ratio median / mean |log ratio| (chain+cycle): 0.83 / 0.82 all, 0.80 / 0.78 medium - but this is the
  already-tested value-retention signal.

## Null (tested, nothing)
- Community coherence of sender/receiver (Jaccard, common neighbours, Adamic-Adar over the week graph): 0.43-0.61.
- Amount against the wallet's own history (z-score / rank vs sender sends, receiver receipts): inconsistent
  (cycle 0.72, chain 0.40); also already in fields.
- Asset typicality for the wallet (share of its payments in that asset): confounded by the motif's own payments.
- Wallet hour-of-day typicality; weekly degree; weekly net flow: 0.45-0.62.
- Interleaving (other payments of the intermediate during the hold, vs its rate): chain 0.56, cycle 0.73, stack 0.77
  medium, direction flips -> speed-confounded.
- Within-sender amount order (amounts rise with time inside a sender's sends): a generator artifact shared by
  suspicious and look-alikes (Spearman ~0.7 both); only fan_out medium breaks it (not my family).
- Modular time artifacts (step seconds mod 60, start mod 300/3600): flat.
- SEQUENCE: just the sender's running payment count (no gaps, no scenario-level counters).
- Cohort ids: none for chain/cycle/stack.
