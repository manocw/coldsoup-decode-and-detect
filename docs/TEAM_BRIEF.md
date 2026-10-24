# Team brief: Decode & Detect (team coldsoup)

Living fact sheet for whoever writes the slides/report (< 10 slides or <= 4 pages). Every number
here is measured; the source is named so it can be re-checked. Last updated: 2026-10-24.

The organisers want the slides/report to cover: methodology, features, model, validation approach,
results, interesting findings, limitations. Sections 2-8 map onto those.

---

## 0. Submission checklist (final ZIP)

| Required item | Where it is / status |
|---|---|
| Test predictions CSV | `submissions/<run>.csv` (best so far: `v3_9fam_stack.csv`, site score 72.3) |
| Validation results | Section 6 below + `notes/leaderboard.md`; CV printed by `src.predict --cv` |
| Code | `src/` (built into `submissions/code.zip` by `python -m src.package`) |
| Model details / settings | Section 4 below; exact settings in `src/evaluate.py:make_model` |
| Run instructions | `README.md` (exact command, ~1-2 min on a laptop CPU, no external services) |
| Slides or report | **team to write** from this brief |
| Team details | **TODO: team name coldsoup, captain = ?, members = ?, final commit = ?** |

No prompts/LLMs are used in the model itself, so there is no prompt set. AI assistance (Claude Code)
was used to write the code; this is declared in the README.

---

## 1. The task in one paragraph

A synthetic, fictional XRPL-style payment ledger: one week per split, ~40k payments between 700
wallets, in XRP and issued USD/GBP/EUR tokens (two issuers), with some payments converting currency
through liquidity providers. Some payments were planted as money-laundering motifs. We must give
every test payment a 0-1 suspicion score; the site scores 100 x average precision (AP). Train has
labels (1,250 suspicious = 3.1%); validation is unlabelled; test (40,101 payments) is scored by the
site. Wallets, issuer addresses and dates are all different in every split.

## 2. Key data insights (these drive everything; good slide material)

1. **It's about bad motif instances, not bad wallets.** Labels are pure per planted scenario (a motif
   instance of ~5-28 payments): 169 suspicious scenarios vs 1,328 legitimate look-alike scenarios of
   the *same shapes* (fan-out, fan-in, cycle, chain, ...). Every suspicious sender also sends mostly
   legitimate traffic (~5% of its payments are suspicious), and 535 of 700 wallets touch something
   suspicious. Wallet blacklists cannot work.
2. **Speed separates the easy cases.** Easy suspicious motifs complete in 2-14 minutes (e.g. a 6-hop
   cycle in 2.4 min); legit look-alikes take ~1 hour. Medium suspicious motifs are slowed down to
   1-3 h and overlap the look-alikes, which is why medium is hard.
3. **One-off relationships mark the planted edges.** A sender->receiver pair used exactly once in the
   week is typical of planted motifs; legit commerce repeats on a schedule (3-7 payments at a fixed
   period, near-constant amount).
4. **Legit relays are the biggest false-positive source.** 12 "relay cohorts" of 100 look-alike
   scenarios each run hub-and-spoke rounds over the whole week. In the baseline's top 5%, 998 false
   positives were legit relays. Relay edges repeat 2-4 times tens of hours apart; half the legit
   relays run "backwards" (hub pays out first, refilled later), while all 12 suspicious relays run
   forwards and pass on ~95% of hops at near-equal value (only ~36% for legit forward relays).
5. **Currency swaps are a red herring here.** The share of 1/2/3-leg payments is identical for
   suspicious and legit (59/31/10%), and the organisers state conversions ignore the label. The
   conversion mix also shifts between splits (41% of train payments convert vs 50% of test), so
   conversion features would actively hurt. We deliberately excluded them.
6. **Raw XRPL fields carry no signal** (fee uniform 10/12/15 drops, LastLedgerSequence always +20,
   SendMax always 1.01x cost on conversions, transaction order within a ledger random). Only the
   *context* matters: timing, counterparties, value flow.
7. **Planted motifs are placed uniformly in time**, while real-looking traffic is diurnal or in
   rounds, so payments in quiet periods are enriched for planted motifs (suspicious and look-alike).
8. **25 "ambiguous" hard cases are unlearnable by design**: single payments sitting inside regular
   scheduled payment series with the same amount and period as their legit siblings.

Null results worth stating (we tested and rejected): round amounts, just-under-threshold amounts
(no excess mass under 100/500/1k/3k/5k/10k; amounts top out ~4.6k USD in train), repeated identical
amounts, overpaying conversion spreads, cross-issuer swaps, hour-of-day profiles, and "mule wallet
reuse" (wallets in medium/hard scenarios are no more linked to other suspicious scenarios than
look-alike wallets: 6.9% vs 6.6%).

## 3. Methodology (pipeline)

```
parquet (transactions + transfers + market rates)
  -> one row per payment: sender's real cost (first transfer leg), delivered value, USD at the hourly rate
  -> feature families computed over the whole week of the SAME split (offline investigation is allowed)
  -> stage 1: LightGBM classifier trained on train labels
  -> stage 2 (v2b): LightGBM re-scores each payment using its linked neighbours' stage-1 scores
  -> test scores, min-max squashed to [0, 1]
```

Design rules: features describe behaviour relative to each wallet and to the split itself (no
addresses, no absolute dates, no labels), because wallets/issuers/dates all change between splits.

## 4. Features (slide ideas -> what we built)

| Family (module) | Slide idea it implements | What it measures |
|---|---|---|
| starter | organiser baseline | value, sender/receiver activity, forwarding within 1 h, 10-min bursts, pass-through |
| funding | "Who funded it?", mule network finder | time since the sender was funded, by whom, funder's fan-out to wallets that pay on, fan-in/out widths in 5 min-3 h windows, one-off pairs around each payment |
| relays | (false-positive killer) | edge recurrence across rounds, hub in-burst -> out-burst matching, forward vs refill direction, value conservation per hop |
| hops | "Fast hops", money trail tracker | longest matched chain through a payment (10 m / 1 h / 6 h), hop delays, value kept per hop, cumulative skim, cash-out hub at the trail end, stack layering |
| motifs | group-level scoring | unsupervised reconstruction of motif instances from wallet timelines; size, span, shape, value retention of the payment's group |
| fields | raw-field forensics | account cadence, pair novelty, amount vs the wallet's own median, network load around the payment |
| amounts | "Just under limits" | value relationships to adjacent payments (pass-on ratios); static amount tests were null |
| cycles (experimental, not in v2) | round trips | time-respecting paths back to the sender, cycle length/duration/value drift |
| history (v2) | wallet history | pair regularity, wallet roles, deviation from the wallet's own history |

## 5. Model

- LightGBM gradient-boosted trees (binary log-loss): 400 trees, learning rate 0.03, 31 leaves,
  min 20 samples per leaf, 80% row/column subsampling, L2 = 1.0. Averaged over 3 random seeds.
- About 270 features in v1. Training plus scoring takes ~1-2 minutes on a laptop CPU; no GPU,
  no external services, no pre-trained models.
- Stage 2 (v2b, "score sharing"): for every payment, the max / mean / top-2 mean of the stage-1
  scores of (a) the same sender's and (b) the same receiver's payments within 10 min, 1 h and 6 h,
  (c) the receiver's onward payments and (d) the sender's funding payments in those windows, and
  (e) the mean/max score of its one-off-pair motif group (links within 2, 10, 30 min). A second
  LightGBM with the same settings is trained on these plus the original features. To avoid leakage
  it learns from out-of-fold stage-1 scores, and its CV is nested (inner folds inside each outer fold).
  Runtime ~4 min on a laptop CPU (21 LightGBM fits for 3 seeds).
- Organiser starter for comparison: logistic regression on 12 rules.

## 6. Validation approach and results

**Validation**: 5-fold cross-validation on train where whole relay cohorts and whole scenarios stay
in one fold (`StratifiedGroupKFold`). Random row splits leak sibling payments of the same planted
motif between train and test folds. Metric = the site's AP, plus AP restricted to each difficulty.

| Run | CV AP | AP easy | AP medium | AP hard | Site score |
|---|---|---|---|---|---|
| Starter notebook (logistic regression, its own random split) | 24.6 | | | | |
| Starter features, logistic regression, grouped CV | 29.9 | 38.3 | 1.3 | 1.1 | |
| Starter features + LightGBM | 33.7 | 44.5 | 1.4 | 0.9 | |
| v1: 7 families + LightGBM | 59.4 | 76.7 | 6.0 | 1.1 | **71.0** |
| + stage-2 neighbour score sharing (nested CV, 7 families) | 61.3 | 79.5 | 6.1 | 1.2 | |
| v2a: 8 families (+ history) | 60.1 | 77.3 | 6.4 | 1.1 | 71.3 (P@1% 96%, R@1% 31%) |
| v2b: v2a + stage-2 score sharing (nested CV) | 61.6 | 79.6 | 6.3 | 1.2 | 72.2 (P@1% 94%, R@1% 30%) |
| v3: v2b + cycles family (nested CV) | 62.0 | 79.9 | 6.5 | 1.2 | **72.3** (P@1% 94%, R@1% 30%) |

Hyperparameters barely matter (1-seed CV AP, 8 families incl. history): default 60.1, 15 leaves /
min 50 per leaf 60.3, 7 leaves / min 100 per leaf 59.2. Differences are within seed noise (~0.5),
so effort went into features and the two-stage design rather than tuning.

Single-family gains over the starter (1-seed CV AP): funding 58.7, relays 53.8, fields 52.4,
motifs 51.8, hops 50.8, amounts 48.0, history 49.3, cycles ~45.7.

Site breakdown for v1 (share of each kind caught in the top 1% = ~401 payments): easy 40% of 800,
medium 10% of 225, observable-hard 0% of 200, ambiguous 0% of 25. Because only ~401 payments fit in
the top 1%, at most 32% of all 1,250 positives can appear there, so ~342 hits means the top 1% is
~85% precise.

## 7. Interesting findings for the story

- The relay agent cut legit-relay false positives in the top 5% from 998 to 462 and background from
  206 to 94, using only "does this pair repeat on a schedule?" and "does the hub run forwards?".
- The site score (71.0) came in above our conservative grouped CV (59.4): the behavioural features
  transferred to a week with different wallets, issuers and more conversions.
- The organiser's random-split check over- and under-states performance by several points depending
  on the split, because only ~250 positives land in a 20% holdout.
- **Where the remaining errors are (train, out-of-fold, v1):** easy cases are essentially solved
  (754 of 800 easy positives rank in the top 1,250). Negatives in that top 1,250 are mostly
  *legitimate look-alike motifs* (329), then relays (109) and background (22). Medium positives sit
  around rank 2,500 (behind ~730 relay and ~660 look-alike payments); hard relay positives are
  near random.
- **The look-alikes are adversarially designed.** Example: a legitimate bipartite look-alike steps
  every 6-12 seconds, faster than the easy suspicious bipartite (~40 s); a legitimate multi-stage
  look-alike (2-3 min steps) outranks an easy suspicious multi-stage that moves in 12-second steps.
  Speed alone cannot separate them; only combinations of context do.
- **Oracle check on medium (true scenario grouping, analysis only):** medium suspicious motifs are
  *slower* than their legitimate look-alikes (median step ~15-18 min vs ~9 min; best single
  statistic AUC 0.70) and keep more value per hop (0.91-0.96 vs 0.83-0.88). Even with perfect
  grouping the separation is weak, which explains the low medium AP.
- **Medium is at the information limit for whole-scenario statistics.** A classifier given the TRUE
  scenario groupings and 22 scenario statistics (size, span, step gaps, value spread and retention,
  pair novelty, wallet degrees, conversions, network load, account sequence) separates medium
  suspicious from legitimate look-alike scenarios at AUC 0.48 (logistic) / 0.57 (random forest),
  i.e. chance; the same check gives AUC 0.87 / 0.91 for easy vs look-alike. Remaining gains are in
  ranking easy cases above look-alikes, not in medium.
- **Teammate model blend:** Marat's flow-episode features inside our CV score 56.9; combined with
  ours in one model 60.3, rank-blended 0.9/0.1 60.4, vs 60.1 for ours alone: the two feature
  designs capture the same signal, so blending gives no real gain.
- **Grouping by shared wallets bleeds into background traffic** (one-off-pair grouping at a 30-min
  link is only 58% pure, 37% at 1 h), so medium motifs cannot simply be reconstructed and scored.
- **Hard relay cases are near the information limit.** With the true relay scenarios (analysis
  only), the single property separating the 12 suspicious relays from 1,188 legitimate ones is
  direction: all 12 run forwards (hub receives, then pays on) while half the legitimate relays run
  as refills (AUC 0.75). Size, rounds, counterparties, value retention, amounts and conversions are
  all at AUC ~0.5. That caps what any model can do on these 200 payments.
- Score sharing between linked payments (stage 2, nested CV) adds +2.3 AP (59.0 -> 61.3), almost
  all of it on easy cases.

## 8. Limitations (be honest on the slide)

- Medium cases remain hard (AP ~6): slowed-down motifs look like legitimate ones; hard cases ~1.
- 25 ambiguous payments are unlearnable by construction.
- Test has more tiny payments (27% under $20 vs 22% in train) and more conversions; features based
  on absolute amounts are the main drift risk, so we prefer ratio/retention features.
- Whole-week per-wallet aggregates could memorise wallets inside train; wallets are disjoint across
  splits, so this can only inflate CV, not leak into test.
- Synthetic data: the patterns found are those of the simulator; real-ledger transfer is untested.

## 9. Team repo and the two models

GitHub: https://github.com/manocw/xrpl_hackathon. `main` holds Marat's flow-episode LightGBM
(`xrpl_hackathon/marat_model/`, site 66.8, CV 59.0) and `blend.py` (rank averaging). Branch
`two-stage-model` adds this model as `xrpl_hackathon/two_stage_model/` (site 72.2) with its test
predictions in `xrpl_hackathon/predictions/two_stage_v2b.csv`. Both are LightGBM on behavioural
features; they differ in feature design (Marat: flow episodes + first-time-pair episodes at many
time scales; two-stage: eight behavioural families + neighbour score sharing).

## 10. Reproduce

See `README.md`. One command, CPU only, ~1-2 minutes after placing the data under `data/`.
