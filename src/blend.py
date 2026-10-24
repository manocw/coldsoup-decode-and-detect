"""Rank-average several submission CSVs into one (AP only depends on the order of the scores).

    python -m src.blend submissions/v2b_8fam_stack.csv submissions/marat.csv --weights 0.8 0.2 --name blend_v2b_marat
    python -m src.blend a.csv b.csv --compare     # rank correlation and top-1% overlap only

Each input must cover exactly the same transactions. The output is written to
submissions/<name>.csv with scores in [0, 1] and validated like src.predict.
"""
import argparse

import numpy as np
import pandas as pd

from src.load import ROOT

SUBMISSIONS = ROOT / "submissions"


def _load(path):
    s = pd.read_csv(path, usecols=["TRANSACTION_HASH", "score"])
    assert s.TRANSACTION_HASH.is_unique and np.isfinite(s.score).all(), path
    return s.set_index("TRANSACTION_HASH").score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--weights", nargs="+", type=float)
    ap.add_argument("--name", default="blend")
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()

    scores = [_load(f) for f in a.files]
    ids = scores[0].index
    for f, s in zip(a.files[1:], scores[1:]):
        assert set(s.index) == set(ids), f"{f} covers different transactions"
    ranks = pd.concat([s.reindex(ids).rank(pct=True) for s in scores], axis=1, keys=a.files)

    if a.compare:
        print(ranks.corr(method="spearman").round(3).to_string())
        k = max(1, int(round(0.01 * len(ids))))
        tops = {f: set(r.nlargest(k).index) for f, r in ranks.items()}
        for i, f in enumerate(a.files):
            for g in a.files[i + 1:]:
                print(f"top-1% overlap {f} vs {g}: {len(tops[f] & tops[g])}/{k}")
        return

    w = np.asarray(a.weights or [1.0] * len(scores), float)
    assert len(w) == len(scores), "one weight per file"
    blended = ranks.to_numpy() @ (w / w.sum())
    lo, hi = blended.min(), blended.max()
    out = pd.DataFrame({"TRANSACTION_HASH": ids, "score": (blended - lo) / (hi - lo)})
    assert out.score.between(0, 1).all() and len(out) == len(ids)
    SUBMISSIONS.mkdir(exist_ok=True)
    path = SUBMISSIONS / f"{a.name}.csv"
    out.to_csv(path, index=False, float_format="%.8f")
    print(f"wrote {path} ({len(out):,} rows, weights {np.round(w / w.sum(), 3).tolist()})")


if __name__ == "__main__":
    main()
