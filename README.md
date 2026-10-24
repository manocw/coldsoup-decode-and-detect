# Decode & Detect - suspicious payment scorer (team coldsoup)

Scores every transaction of the synthetic XRPL-style ledger from the UKCBT "Decode & Detect"
hackathon with a suspicion score in [0, 1]. Each transaction gets behavioural features computed
over the whole week of its own split (funding/fan structure, relay structure, hop trails,
reconstructed motif groups, amounts, raw-field cadence, wallet history). A LightGBM model is
trained on the labelled `train/` split and scores the test split. A second LightGBM stage
re-scores each payment using the first-stage scores of its linked neighbours (same wallet close in
time, hand-offs, one-off-pair motif groups), trained on out-of-fold first-stage scores. The final
model (v7) uses 10 feature families, including the motif pace ("tempo") of each payment's one-off-pair
group, and two auxiliary stage-1 helper models (relay-cohort and planted-motif membership; annotation
targets only, never features) whose neighbour-pooled scores also feed stage 2.

## Team

- Team: coldsoup
- Captain: TODO_CAPTAIN
- Members: TODO_MEMBERS
- Final submission: `b1_rank_v7_v12` (site score 74.0), a rank average of v7 (site 73.8) and v12 (site 73.5). Report: `docs/report.pdf` (source `docs/report.html`).

## Reproduce the predictions

Python 3.14 (tested 3.14.4), Linux, CPU only.

```sh
python -m venv venv && ./venv/bin/pip install -r requirements.txt
# put the organiser data under data/: data/train/ (with labels.csv) and data/test/
# (unzip the test download into data/test/)
F="starter funding relays fields motifs hops amounts history"
# final - b1 (site 74.0): rank average of v7 and v12, half each (~20 min on CPU)
./venv/bin/python -m src.predict --features $F cycles tempo --seeds 3 --stack --aux --name v7_10fam_tempo_aux
./venv/bin/python -m src.predict --features $F cycles tempo medium --seeds 3 --stack --aux --name v11_11fam_tempo_medium_aux
./venv/bin/python -m src.specialist --base v11_11fam_tempo_medium_aux --weight 0.3 --name v12_v11_plus_spec
./venv/bin/python -m src.blend submissions/v7_10fam_tempo_aux.csv submissions/v12_v11_plus_spec.csv --weights 0.5 0.5 --name b1_rank_v7_v12
# v7: v6 + the tempo family (site 73.8)
./venv/bin/python -m src.predict --features $F cycles tempo --seeds 3 --stack --aux --name v7_10fam_tempo_aux
# v6: v3 + auxiliary stage-1 targets (is-relay, is-any-planted-motif) pooled in stage 2 (site 73.0)
# v7: v6 + 0.3 x medium specialist (stage-1 LightGBM trained on train + slowed copies of easy motifs)
./venv/bin/python -m src.specialist --base v6_9fam_stack_aux --weight 0.3 --name v7_v6_plus_spec
# v6: v3 + auxiliary stage-1 targets (is-relay, is-any-planted-motif) pooled in stage 2
./venv/bin/python -m src.predict --features $F cycles --seeds 3 --stack --aux --name v6_9fam_stack_aux
# v2c: 8 families + auxiliary targets (site 72.6)
./venv/bin/python -m src.predict --features $F --seeds 3 --stack --aux --name v2c_8fam_stack_aux
# v4: v3 + transductive self-training of stage 2 on the test week (2 rounds; site 72.2, not better)
./venv/bin/python -m src.predict --features $F cycles --seeds 3 --stack --selftrain 2 --name v4_9fam_stack_selftrain
# v3: two-stage model with the cycles family added
./venv/bin/python -m src.predict --features $F cycles --seeds 3 --stack --name v3_9fam_stack
# two-stage model (v2b, site 72.2)
./venv/bin/python -m src.predict --features $F --seeds 3 --stack --name v2b_8fam_stack
# single-stage model (v2a)
./venv/bin/python -m src.predict --features $F --seeds 3 --name v2a_8fam
# v1 (site score 71.0): --features starter funding relays fields motifs hops amounts --name v1_7fam
# -> submissions/<name>.csv  (TRANSACTION_HASH,score; 40,101 rows)
```

Runtime on a 14-core laptop CPU: single-stage ~45 s, two-stage ~4-7 min (21 LightGBM fits; v4
adds 6 self-training fits, ~9 min), two-stage with `--aux` ~4.5 min (57 fits; a re-run on a second
machine gave identical scores); peak memory 1.4 / 1.9 GB. Feature building takes < 20 s per split per family and is cached under
`cache/`. Add `--cv` to also print the grouped cross-validation report on train (nested CV for
`--stack`, ~5 min). `src/features/groups.py` and `marat.py` (teammate features) are experimental and unused by
these runs; `cycles.py` is used from v3. `src/selftrain.py` is the CV simulation of self-training;
`src/gnn.py`, `src/gnn_export.py` and `colab/gnn_colab.ipynb` are an experimental graph neural
network for Colab (not used in the submitted predictions).

On Windows, if Application Control blocks pyarrow's parquet DLL ("DLL load failed while importing
_parquet"), `pip install fastparquet` and run with that engine:
`python -c "import sys,runpy,pandas as pd; pd.set_option('io.parquet.engine','fastparquet'); sys.argv=['src.predict',*sys.argv[1:]]; runpy.run_module('src.predict',run_name='__main__')" --features $F cycles tempo --seeds 3 --stack --aux --name v7_10fam_tempo_aux`

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

Labels (`is_suspicious`, `family`, `difficulty`, `scenario_id`, `cohort_id`) are used only as
training targets and for grouping CV folds, never as features. No external data is used.
Annotation columns (`family`, `cohort_id`, `scenario_id`) are used as auxiliary training targets
only, never as features: two stage-1 helper models learn "is this payment part of a relay cohort"
and "is it part of a planted non-relay scenario (suspicious or legitimate look-alike)" from the
behavioural features; their out-of-fold scores feed stage 2. At test time nothing from
`labels.csv` is needed.

## Validation (grouped 5-fold CV on train)

| Model | AP x 100 |
|---|---|
| Organiser starter notebook (logistic regression) | 29.9 |
| Starter features + LightGBM | 33.7 |
| v1: 7 feature families + LightGBM | 59.4 (site 71.0) |
| v2a: + wallet history (8 families) | 60.1 (site 71.3) |
| v2b: v2a + stage-2 neighbour score sharing (nested CV) | 61.6 (site 72.2) |
| v2c: v2b + auxiliary relay/motif helper targets (nested CV) | 62.3 (site 72.6) |
| v3: v2b + cycles family (nested CV) | 62.0 (site 72.3) |
| v4: v3 + stage-2 self-training on the test week | stage-1 CV simulation +1.0 (2 seeds); site 72.2 |
| v6: v3 + auxiliary relay/motif targets (targets only, never features) | 62.1 (site 73.0) |
| v7: v6 + tempo family | 62.4 (site 73.8) |
| v8: v6 + a third helper target "medium suspicious" (`--aux-targets relay motif medium`) | 62.3, medium AP 8.8 vs 9.1: rejected |
| v9: v7 + medium helper target | site 73.3 |
| v10: v7 + 0.3 x medium specialist (`src.specialist`) | site 73.2 |
| v11: v7 + medium feature family | site 73.3 |
| v12: v11 + 0.3 x medium specialist | site 73.5 |
| **b1: rank average 0.5 v7 + 0.5 v12 (final)** | **site 74.0** |
| b2: rank average 0.5 v7 + 0.25 v12 + 0.25 v6 | site 73.8 |
| v6: v3 + auxiliary relay/motif targets (targets only, never features) | site 73.0 |
| v7: v6 + 0.3 x medium specialist (slowed-easy augmentation) | stage-1 CV 59.8 -> 61.9, medium AP 6.4 -> 10.4 |

## Not written by us

- Libraries: pandas, numpy, pyarrow, scikit-learn, LightGBM, networkx, scipy (see `requirements.txt`).
- `src/features/starter.py` re-implements the 12 rules of the organiser's starter notebook, and
  `src/load.py` follows the valuation logic of the organiser's `payments.py`.
- AI assistance: the code was written with Claude Code (Anthropic), under our direction and review.
- No pre-trained models or outside services are used at runtime.

## Licence

MIT, see `LICENSE`.
