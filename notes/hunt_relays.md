# hunt_relays - new differences between the 12 hard suspicious relays and legit relay siblings

Scratch: /tmp/claude-1000/-home-peroxide-Downloads-OpenCode-xrpl-hack/42315158-3b36-488f-933b-da88881ad637/scratchpad/hunt_relays/
Population (train): 12 suspicious relay scenarios (6 aggregate, 6 distributed; 200 payments) vs 1188 legit
relay siblings (588 forward, 600 refill). AUCs below: scenario-level sus vs legit FORWARD unless noted
(12 positives: SE ~0.08), payment-level within relay rows (20,000 rows, 200 positive).

## Reference points
- Current model OOF score, within relay rows: payment AUC 0.656 (vs forward only 0.600); scenario mean 0.757.
- Existing relays.hub_aok (label-free per-hop conservation): payment AUC 0.769 relay / 0.719 vs fwd.
- Oracle per-hop conservation (true scenarios): share of hub-rounds with |log out/in| < 0.55: sus 0.95,
  legit fwd 0.50, legit refill 0.93 (refills conserve too!); scenario AUC vs fwd 0.964. std of hop
  log-ratios: sus 0.29 vs fwd 0.94. Contamination by foreign payments at the same wallet-asset in
  the round window: ~0.6-0.8 extra payments per hub-round (cuts sus "ok" share 0.93 -> 0.77).
- No sequence / tx_index / ts-granularity / last_ledger_offset artefacts (sequence == send order for
  every wallet; all relay AUCs 0.47-0.54).

## Log
Scripts: scen.py/scen2.py (scenario table), auc.py, cons.py, contam.py/contam2.py, ctx.py/ctx2.py,
deg.py/deg2.py, hubact.py, wprof.py, net.py, lags.py, comove.py/comove_oracle.py, overdraft.py, od_lf.py,
prefund.py, seq.py. Each run takes 5-40 s.

### Structure (what the generator does)
The 12 suspicious relays use the legit FORWARD template unchanged: same rounds (3-4), same round
times, same hub lags (aggregate last-in -> first-out 4.36 h vs 4.36 h; distributed per hop 2.15 vs
2.24 h; hop-round AUCs 0.43-0.53), same in/out spread, fan sizes and counterparty recurrence. The only
large difference is value conservation (already known). Refills conserve as well (93% of hub-rounds)
but in reverse order; legit forwards conserve in 50% of hub-rounds by chance.

### Hypotheses tested (scenario AUC sus vs legit forward, n = 12 vs 588; payment AUC in relay rows)
1. Round escalation (log last/first round in-sum, Spearman over rounds): 0.57-0.60 (agg 0.70, dis 0.45). null
2. Round-total volatility (cv, log max/min): 0.60-0.64. null
3. Hub timing (lag last-in->first-out, lag std, in/out burst spread): 0.38-0.61; hop-round 0.43-0.53. null
4. Fan structure (n sources, n sinks, outs per round, in/out count ratio): 0.57-0.61. null
5. Counterparty recurrence inside the scenario (source / sink pair repeat share): 0.47-0.51. null
6. Wallet activity: hub weekly degree 0.67 inv (113 vs 124), hub's relay-scenario count 0.73 inv,
   receive degree of sources / hubs 0.70-0.71 inv. Label-free payment level: receiver+sender receive
   degree 0.64 inv (relay and fwd). NEAR MISS, weak; degrees already in starter.
7. Source pre-funding (receipts into the source 1-12 h before its payment): distributed-only scenario
   0.83-0.88 inv (6 positives), aggregate 0.52-0.57 -> inconsistent; payment level (rate-adjusted)
   0.47-0.55. It is the degree effect of (6), not placement. null
8. Sink onward payment / cash-out within 1-6 h: 0.49-0.56. null
9. Hub activity during the holding window (other payments per hour): 0.60 inv. null
10. Hub wallet-asset WHOLE-WEEK net flow |log(out/in)|: scenario 0.76 inv (fwd), 0.72 (all);
    payment min(receiver-side, sender-side) 0.64 inv. NEAR MISS (a retention variant).
11. Running balance / no overdraft: cumulative out never exceeds cumulative in at the hub, peaking at
    81-98% of it. Oracle: all 12 sus have max log(cum_out/cum_in) in [-0.21, -0.02]; 121/588 legit
    forward and 0/600 refills do. Band score -|od+0.08|: scenario AUC 0.932 fwd / 0.966 all relays
    (= per-hop share 0.964). Label-free (wallet-asset sessions, gap 2/4/6.5 h): payment 0.60-0.67
    relay, 0.55-0.59 fwd. FAILS the payment bar (contamination); conceptually still retention.
12. Edge co-movement across rounds (recurring in-edge and out-edge amounts co-scale): oracle scenario
    0.81; label-free covers 41% of sus payments, AUC ~0.50 fwd. null
13. Generation artefacts: sequence (== send order for every wallet), tx_index, ledger fill, ts mod 60 /
    10, last_ledger_offset, has_paths, has_sendmax, fee: 0.47-0.54. null
14. Wallet week profile (one-off / recurring / frequent receipts and sends, distinct counterparties,
    median amounts) of sender and receiver: best 0.63 (fwd). null

### Why the label-free conservation tops out (~0.77 relay / 0.72 fwd for relays.hub_aok)
Foreign payments at the hub's wallet-asset inside a round window: 0.6-0.8 per hub-round, ~90% of
them from OTHER relay scenarios (recurring 2-4x edges as often as one-offs), each ~37% of the round's
in-value (median log -1.0). Background payments in the window are small (~7%). So the noise comes from
overlapping relays on the same hub (cohort rounds share time windows), and pair recurrence, asset and
timing cannot separate them.

## Conclusion
No NEW difference clears the bar. Hard relays = legit forward template + value conservation (known).
Only lever left: disentangle overlapping relays at a hub (choose the in/out subsets that conserve in
every round, forward order), oracle ceiling 0.93-0.96 scenario AUC. Risk: subset choice also makes
chance matches in legit forwards more likely; needs the multi-round constraint. Estimate: 1.5-3 h to
prototype and measure label-free.
