# Decode & Detect - suspicious payment scorer (team coldsoup)

Scores every transaction of the synthetic XRPL-style ledger from the UKCBT "Decode & Detect"
hackathon with a suspicion score in [0, 1]. Each transaction gets behavioural features computed
over the whole week of its own split (funding/fan structure, relay structure, hop trails,
reconstructed motif groups, amounts, raw-field cadence, wallet history). A LightGBM model is
trained on the labelled `train/` split and scores the test split. Optionally a second LightGBM
stage re-scores each payment using the first-stage scores of its linked neighbours (same wallet
close in time, hand-offs, one-off-pair motif groups), trained on out-of-fold first-stage scores.

## Reproduce the predictions

Python 3.14 (tested 3.14.4), Linux, CPU only.

```sh
python -m venv venv && ./venv/bin/pip install -r requirements.txt
# put the organiser data under data/: data/train/ (with labels.csv) and data/test/
# (unzip the test download into data/test/)
F="starter funding relays fields motifs hops amounts history"
# v3: two-stage model with the cycles family added
./venv/bin/python -m src.predict --features $F cycles --seeds 3 --stack --name v3_9fam_stack
# two-stage model (v2b, site 72.2)
./venv/bin/python -m src.predict --features $F --seeds 3 --stack --name v2b_8fam_stack
# single-stage model (v2a)
./venv/bin/python -m src.predict --features $F --seeds 3 --name v2a_8fam
# v1 (site score 71.0): --features starter funding relays fields motifs hops amounts --name v1_7fam
# -> submissions/<name>.csv  (TRANSACTION_HASH,score; 40,101 rows)
```

Runtime on a 14-core laptop CPU: single-stage ~45 s, two-stage ~4 min (21 LightGBM fits);
peak memory 1.4 / 1.9 GB. Feature building takes < 20 s per split per family and is cached under
`cache/`. Add `--cv` to also print the grouped cross-validation report on train (nested CV for
`--stack`, ~5 min). `src/features/groups.py` and `marat.py` (teammate features) are experimental and unused by
these runs; `cycles.py` is used by v3 only.

## Layout

| Path | Purpose |
|---|---|
| `src/load.py` | One row per transaction: sender cost from the first transfer leg, delivered value, reference USD |
| `src/features/*.py` | Feature families; each `build(p)` sees one split's unlabelled table only |
| `src/evaluate.py` | Grouped CV (relay cohort > scenario > transaction) with the site's AP metric |
| `src/ablate.py` | Measures each feature family's gain |
| `src/predict.py` | Trains on train, scores test, validates and writes the CSV |
| `src/stack.py` | Stage 2: neighbour score sharing, nested CV |
| `src/package.py` | Builds this code ZIP |

Labels (`is_suspicious`, `family`, `difficulty`, `scenario_id`, `cohort_id`) are used only as the
training target and for grouping CV folds, never as features. No external data is used.

## Validation (grouped 5-fold CV on train)

| Model | AP x 100 |
|---|---|
| Organiser starter notebook (logistic regression) | 29.9 |
| Starter features + LightGBM | 33.7 |
| v1: 7 feature families + LightGBM | 59.4 (site 71.0) |
| v2a: + wallet history (8 families) | 60.1 (site 71.3) |
| v2b: v2a + stage-2 neighbour score sharing (nested CV) | 61.6 (site 72.2) |
| v3: v2b + cycles family (nested CV) | 62.0 |

## Not written by us

- Libraries: pandas, numpy, pyarrow, scikit-learn, LightGBM, networkx, scipy (see `requirements.txt`).
- `src/features/starter.py` re-implements the 12 rules of the organiser's starter notebook, and
  `src/load.py` follows the valuation logic of the organiser's `payments.py`.
- AI assistance: the code was written with Claude Code (Anthropic), under our direction and review.
- No pre-trained models or outside services are used at runtime.

## Licence

MIT, see `LICENSE`.
