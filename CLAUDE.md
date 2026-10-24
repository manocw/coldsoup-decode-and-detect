# CLAUDE.md - Decode & Detect (UKCBT XRPL AI Hack)

Goal: give every transaction in the **test split** a suspicion score in [0, 1]. Scored by
100 x average precision (AP) at https://hackathon.ukcbt.org (team `coldsoup`, 50 uploads/day,
resets 00:00 UTC). Full organiser brief: `reference/COMPETITION_README.md`.

## Data (gitignored, under `data/`)

| Split | Txs | Labels | Window |
|---|---|---|---|
| `data/train/` | 40,131 | `labels.csv` (1,250 suspicious, 3.1%) | 2026-09-01..07 |
| `data/validation/` | 33,862 | none (unlabelled practice) | 2026-09-10..16 |
| `data/test/` | 40,101 | none - this is what we submit | 2026-09-20..26 |

Each split: `transactions.parquet`, `transfers.parquet` (1-3 legs per tx), `blocks.parquet`,
`market_rates.csv` (hourly micro-USD per asset, `XRP` or `issuer:currency`).

- **Synthetic fictional ledger.** Real XRPL data (Bithomp, XRPSCAN) does not match its
  distribution and has no labels - do not use it for training.
- Wallets and hashes are disjoint between splits: no account-level label leakage, so features
  must describe *behaviour*, not identity.
- Value money from `DELIVERED_AMOUNT` x hourly rate (see `reference/payments.py`). `SendMax` is a
  ceiling, not the debit; the first executed transfer leg is the real cost.
- Group assets by (TOKEN_ADDRESS, SYMBOL); two issuers both issue USD/GBP/EUR.
- Suspicious = 64% easy / 18% medium / 18% hard. Hard cases play out over long histories with
  changing counterparties: use aggregate/whole-week account context. Test has "batch variations
  absent from training" - prefer robust behavioural features over memorised routes.
- `labels.csv` columns `family`, `difficulty`, `scenario_id`, `cohort_id` are annotation
  metadata: OK for analysis, grouped CV and as auxiliary training targets (the `--aux` stage-1
  helpers, scored out-of-fold), never as model inputs. Note "normal" difficulty rows
  in a typology family are legitimate look-alikes (hard negatives).
- Rules: no external labels, organiser files, or test answers; split identity is not a feature.
  Using the unlabelled validation/test transactions for unsupervised context is allowed.

## Validation discipline

Evaluate with AP on held-out training data using **GroupKFold by scenario/connected component**,
not random rows - random splits leak a scenario's sibling transactions and inflate AP.

## Submission

- CSV: `TRANSACTION_HASH,score`, exactly the 40,101 test hashes, once each.
- Code ZIP (<= 8 MiB): source, README with exact command + runtime, requirements.txt, Python
  version, LICENSE (MIT), list of third-party libs and AI assistance. No data, no venv.

## Code layout

- `src/load.py` - `load_split(split)`: one row per tx (sender cost from first leg, delivered value, USD).
- `src/features/<family>.py` - `build(p)` on the unlabelled table of one split; cached in `cache/`.
- `src/evaluate.py` - grouped CV (cohort > scenario > hash), AP + per-difficulty AP + top-1%.
- `src/ablate.py` - `python -m src.ablate starter <family>`: measure a family's gain.
- `src/predict.py` - train on all of train, score test, write `submissions/<name>.csv` (validated).
- `src/package.py` - build `submissions/code.zip` for upload.

Baseline (starter features, LightGBM, grouped CV): AP 33.7.

## Environment

`./venv/bin/python` (3.14; pandas, pyarrow, scikit-learn, lightgbm, networkx). No GPU; 14 cores.
Set `XRPL_NJOBS=2` when several CV runs share the machine.
