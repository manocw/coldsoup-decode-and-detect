# context - what happens just before and just after a motif

Question: within-scenario statistics cannot separate medium suspicious from look-alike scenarios
(oracle AUC 0.48-0.57). Does the CONTEXT (funding of the sources before, onward payments of the
sinks after, activity of the wallets around the motif) separate them?

Scratch: /tmp/claude-1000/-home-peroxide-Downloads-OpenCode-xrpl-hack/42315158-3b36-488f-933b-da88881ad637/scratchpad/context/

Population (train, true scenario_id, analysis only): non-relay, non-ambiguous typology scenarios:
105 easy suspicious, 27 medium suspicious, 140 legit look-alikes.

## Log

### Exp 1 - scenario context table (ctx1.py -> ctx1.parquet, auc.py; 1.6 s build)
Per scenario: sources = send but never receive inside it, sinks = receive but never send (median 1 and
1; 12-16% of scenarios have none, e.g. cycles). Pre = payments INTO a source in the W before its first
motif send (not part of the scenario); post = payments OUT of a sink in the W after its last motif
receipt. W = 10m / 1h / 6h.

Class means (easy / medium / look-alike):
| stat | easy | medium | look-alike |
|---|---|---|---|
| span_h (within-scenario, reference) | 0.13 | 2.37 | 1.61 |
| sources funded within 10m | 0.06 | 0.08 | 0.05 |
| sources funded within 1h | 0.18 | 0.21 | 0.24 |
| sources funded within 6h | 0.66 | 0.78 | 0.70 |
| common funder share (6h) | 0.46 | 0.59 | 0.46 |
| pre-funding one-off share (6h) | 0.43 | 0.49 | 0.35 |
| log pre-funding value / motif outflow (1h) | -4.7 | -4.5 | -4.2 |
| sinks pay onward within 10m | 0.03 | 0.06 | 0.03 |
| sinks pay onward within 1h | 0.23 | 0.12 | 0.22 |
| sinks pay onward within 6h | 0.71 | 0.63 | 0.71 |
| common destination share (6h) | 0.47 | 0.38 | 0.45 |
| post one-off share (6h) | 0.29 | 0.42 | 0.30 |
| log onward value / sink inflow (1h) | -4.4 | -4.6 | -3.9 |
| wallet activity before (1h, per wallet) | 0.57 | 0.71 | 0.73 |
| wallet activity after (1h, per wallet) | 0.59 | 0.63 | 0.66 |
| pre payments inside another scenario (6h) | 0.60 | 0.50 | 0.58 |
| source idle time before first send (h, mean) | 6.6 | 2.1 | 5.3 |
| whole-week degree of motif wallets | 110 | 110 | 109 |

Reading: there is NO placement or integration step around planted motifs. Sources are funded in the
hour before at the background rate (a wallet has ~110 events/week, ~0.65/h, so P(any receipt in 1h)
~0.25), and sinks pay onward at the same background rate. (The two value-ratio rows above count an
unfunded source as 0 USD, so they mostly measure "funded or not"; the conditional version is in
Exp 2.) Easy, medium and look-alike look the same.

Univariate AUC, medium (27) vs look-alike (140): best context stat is ctx_oneoff_1h 0.62 (other
one-off payments of the motif wallets in [t0-1h, t1+1h], confounded with span), post_oneoff_6h 0.61,
pre_inscen_1h 0.38 (72% NaN). Everything else 0.42-0.58 (27 positives: SE ~0.06).
Univariate AUC, easy (105) vs look-alike: span_h 0.09 (reference); context best act_during_per_h
0.31 and ctx_oneoff_1h 0.36 (both confounded with span), the rest 0.41-0.59.

Scenario-level CV (5-fold x10, AUC):
| set | medium logreg | medium RF | easy logreg | easy RF |
|---|---|---|---|---|
| context table (incl. n, span) | 0.545 | 0.534 | 0.807 | 0.869 |
| oracle within-scenario stats | 0.553 | 0.583 | 0.885 | 0.926 |
| oracle + context | 0.574 | 0.558 | 0.834 | 0.919 |
=> context adds nothing for medium (chance either way) and nothing for easy (RF 0.919 vs 0.926).

### Exp 2 - conditional value ratios, rate-normalised context, wallet profiles, residual over the model (ctx2.py, auc2.py, nested.py)
Class means (easy / medium / look-alike):
- log(pre-funding / motif outflow), only for sources funded in the 1h before: -1.25 / -2.03 / -0.73
  (72% NaN); 6h: -0.58 / -0.93 / -0.10. Medium AUC 0.42. When a source IS funded, the funding is
  about the motif's size for look-alikes and somewhat smaller for suspicious, but too few are funded.
- log(onward / sink inflow), forwarding sinks, 6h: -0.46 / +0.03 / +0.27. Medium AUC ~0.45.
- one-off payments of the motif wallets just outside the motif vs the wallets' own weekly one-off
  rate (excess per wallet, 10m/1h): ~0 in every class; share of them that are "fast" one-offs
  (another one-off sharing a wallet within 10 min): 0.28 / 0.21 / 0.23 (10m).
- whole-week profile of the motif wallets (outside the motif): one-off share of sends 0.33 / 0.34 /
  0.34, median pair count 2.2 / 2.2 / 2.2, distinct counterparties 46 / 46 / 46, in-out balance ~0.
  Same for the sources and sinks separately. Wallets are drawn from one shared pool.
- (labels, analysis only) share of motif wallets that also appear in ANOTHER suspicious scenario:
  0.80 / 0.84 / 0.77 - wallets are reused everywhere, no "criminal wallet" pool.

Scenario CV (5-fold x6, AUC), 75 label-free context stats (no size/span):
| set | easy logreg | easy RF | medium logreg | medium RF |
|---|---|---|---|---|
| context only | 0.560 | 0.524 | 0.553 | 0.445 |
| scenario mean/max OOF score of current model (8 fam.) | 0.886 | 0.863 | 0.623* | 0.635* |
| OOF + all context | 0.838 | 0.851 | 0.564 | 0.495 |
| OOF + top-6 context (selected on all data: biased) | 0.881 | 0.875 | 0.697 | 0.642 |
(*) the model ranks medium BELOW look-alikes (univariate OOF AUC 0.378); the classifier flips it.
Honest check of the 0.697 (nested.py: top-k selected inside each training fold, logreg, 5x10):
easy base 0.884 -> +top3 0.874, +top6 0.869; medium base 0.621 -> +top3 0.499, +top6 0.492.
Label-permuted runs of the same procedure give 0.40-0.61. => the 0.697 was selection bias; adding
context makes the scenario classifier WORSE for both easy and medium.

### Exp 3 - campaign context: are suspicious motifs near other suspicious motifs? (neigh.py)
Other planted motifs sharing a wallet within [t0-1h, t1+1h]: share suspicious 0.57 easy / 0.27
medium / 0.53 look-alike; within 6h 0.50 / 0.40 / 0.45. Label AUC 0.45-0.49 => no campaign
structure; suspicious motifs are placed independently of each other. (Max OOF score of neighbouring
payments within 1h: medium AUC 0.65, but it is span-confounded - medium neighbours are LESS
suspicious by label, 0.015 vs 0.028.) Stage 2's gain therefore comes from within-scenario siblings,
not from cross-scenario campaigns.

## Conclusion (null result)
The context of a motif - placement before, integration / cash-out after, common funder, common
destination, one-off pre/post edges, value in vs out, fast motifs around it, activity before vs
after, wallet week profiles, neighbouring suspicious motifs - carries no information about the label
beyond chance, for medium (context-only CV AUC 0.45-0.55) and for easy (0.52-0.56), and adding it to
within-scenario statistics or to the current model's scenario score lowers the scenario AUC
(nested check). The generator plants motifs on top of background traffic without any surrounding
placement or cash-out structure. Per the task gate, src/features/context.py was NOT implemented
(a per-transaction, label-free version is expected to be noisier than this true-grouping check).
