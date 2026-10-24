"""Train on all of train/, score the test split, write a validated submission CSV.

    ./venv/bin/python -m src.predict --features starter --model lgbm --name v0_starter_lgbm
    ./venv/bin/python -m src.predict ... --cv      # also print the grouped-CV report first
    ./venv/bin/python -m src.predict ... --stack   # two-stage neighbour score sharing (src/stack.py)
    ./venv/bin/python -m src.predict ... --params '{"num_leaves": 15, "min_child_samples": 50}'
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from src import stack
from src.evaluate import cv_report, lgbm_params, make_model, metrics
from src.features import build_features
from src.load import ROOT, load_split

SUBMISSIONS = ROOT / "submissions"


def fit_predict(Xtr, ytr, Xte, kind, seeds):
    s = np.zeros(len(Xte))
    for seed in seeds:
        s += make_model(kind, seed).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    return s / len(seeds)


def to_unit_range(scores):
    """Rank-preserving squash into [0, 1]; AP only depends on order."""
    lo, hi = float(np.min(scores)), float(np.max(scores))
    return (scores - lo) / (hi - lo) if hi > lo else np.full_like(scores, 0.5)


def validate_submission(df, test):
    assert list(df.columns[:2]) == ["TRANSACTION_HASH", "score"]
    assert len(df) == len(test) == df.TRANSACTION_HASH.nunique()
    assert set(df.TRANSACTION_HASH) == set(test.hash)
    assert df.score.between(0, 1).all() and df.score.notna().all()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", nargs="+", default=["starter"])
    ap.add_argument("--model", default="lgbm")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--name", required=True)
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--stack", action="store_true", help="add stage 2: neighbour score sharing (src.stack)")
    ap.add_argument("--aux", action="store_true", help="with --stack: auxiliary relay/motif targets")
    ap.add_argument("--params", default="{}", help='LightGBM overrides as JSON, e.g. \'{"num_leaves": 15}\'')
    a = ap.parse_args()
    os.environ["XRPL_LGBM"] = a.params

    train, test = load_split("train"), load_split("test")
    Xtr = build_features("train", a.features, train)
    Xte = build_features("test", a.features, test)[Xtr.columns]
    seeds = list(range(a.seeds))
    meta = {"features": a.features, "model": a.model, "seeds": len(seeds), "n_features": Xtr.shape[1],
            "stack": a.stack, "aux": a.aux, "lgbm_params": lgbm_params()}
    if a.cv:
        if a.stack:
            s1, s2 = stack.nested_cv(Xtr, train, a.model, aux=a.aux)
            meta["cv_stage1"], meta["cv"] = metrics(s1, train), metrics(s2, train)
            print(f"[{a.name}] stage1 AP={meta['cv_stage1']['AP']:.1f}  stage2 AP={meta['cv']['AP']:.1f}")
        else:
            meta["cv"], _ = cv_report(Xtr, train, a.model, label=a.name)

    if a.stack:
        s = stack.fit_predict(Xtr, train, Xte, test, a.model, seeds, aux=a.aux)
    else:
        s = fit_predict(Xtr.to_numpy(float), train.is_suspicious.to_numpy(), Xte.to_numpy(float), a.model, seeds)
    sub = pd.DataFrame({"TRANSACTION_HASH": test.hash, "score": to_unit_range(s)})
    validate_submission(sub, test)
    SUBMISSIONS.mkdir(exist_ok=True)
    out = SUBMISSIONS / f"{a.name}.csv"
    sub.to_csv(out, index=False, float_format="%.8f")
    (SUBMISSIONS / f"{a.name}.json").write_text(json.dumps(meta, indent=2, default=float))
    print(f"wrote {out} ({len(sub):,} rows, {out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
