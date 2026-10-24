"""Self-training on the test week: learn from our own most confident test predictions.

The test week differs from train (slower easy motifs, more tiny payments, more conversions, a
different look-alike mix). A model trained on train alone cannot adapt to that. Here a first
model's most confident test payments are added to the training data as pseudo-labels:
the top-K scores as suspicious, the bottom share as legitimate. No test answers are used.

To stop the model simply memorising its own pseudo-labels, the test week is cross-fitted: the
week is cut into time blocks, and each block is scored by a model whose pseudo-labels come only
from the other blocks.

    python -m src.pseudo --base submissions/v2b_8fam_stack.csv --features <families> --name pl_v2b
"""
import argparse
import json

import numpy as np
import pandas as pd

from src.evaluate import make_model
from src.features import build_features
from src.load import ROOT, load_split
from src.predict import to_unit_range, validate_submission

SUBMISSIONS = ROOT / "submissions"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="submission CSV whose scores define the pseudo-labels")
    ap.add_argument("--features", nargs="+", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--top", type=int, default=300, help="test payments taken as pseudo-suspicious")
    ap.add_argument("--neg-share", type=float, default=0.6, help="lowest share taken as pseudo-legitimate")
    ap.add_argument("--weight", type=float, default=0.5, help="sample weight of pseudo-labelled rows")
    ap.add_argument("--blocks", type=int, default=4, help="time blocks for cross-fitting the test week")
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()

    train, test = load_split("train"), load_split("test")
    Xtr = build_features("train", a.features, train)
    Xte = build_features("test", a.features, test)[Xtr.columns]
    Xtr, Xte = Xtr.to_numpy(float), Xte.to_numpy(float)
    ytr = train.is_suspicious.to_numpy()

    base = pd.read_csv(a.base).set_index("TRANSACTION_HASH").score.reindex(test.hash).to_numpy()
    assert np.isfinite(base).all(), "base file must cover every test transaction"
    order = np.argsort(-base)
    pseudo = np.full(len(test), -1)
    pseudo[order[: a.top]] = 1
    pseudo[order[int(len(order) * (1 - a.neg_share)):]] = 0

    # time blocks over the test week (planted motifs are local in time)
    t = test.ts.to_numpy()
    block = np.minimum((a.blocks * (t - t.min()) / (t.max() - t.min() + 1)).astype(int), a.blocks - 1)

    score = np.zeros(len(test))
    for b in range(a.blocks):
        use = (pseudo >= 0) & (block != b)
        X = np.vstack([Xtr, Xte[use]])
        y = np.concatenate([ytr, pseudo[use]])
        w = np.concatenate([np.ones(len(ytr)), np.full(use.sum(), a.weight)])
        rows = block == b
        for seed in range(a.seeds):
            m = make_model("lgbm", seed).fit(X, y, sample_weight=w)
            score[rows] += m.predict_proba(Xte[rows])[:, 1] / a.seeds
        print(f"  block {b + 1}/{a.blocks}: {use.sum():,} pseudo rows ({pseudo[use].sum()} positive)", flush=True)

    sub = pd.DataFrame({"TRANSACTION_HASH": test.hash, "score": to_unit_range(score)})
    validate_submission(sub, test)
    SUBMISSIONS.mkdir(exist_ok=True)
    out = SUBMISSIONS / f"{a.name}.csv"
    sub.to_csv(out, index=False, float_format="%.8f")
    (SUBMISSIONS / f"{a.name}.json").write_text(json.dumps(vars(a), indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
