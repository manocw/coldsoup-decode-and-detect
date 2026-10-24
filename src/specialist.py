"""Medium specialist: a stage-1 LightGBM trained on train plus slowed copies of the easy suspicious
motifs (src/augment.py), added to the main model's scores with a small weight.

Medium suspicious motifs are slowed-down easy motifs; training on slowed copies doubles medium AP
(6.4 -> 12.8 in grouped CV) but also lifts slow legitimate look-alikes, so on its own the
specialist is worse (AP 54.4). Added to the main model's probability with weight 0.3 it keeps the
main model's confident top and lifts medium through the deep ranks: stage-1 CV AP 59.8 -> 61.9,
medium AP 6.4 -> 10.4, medium AUC 0.926 -> 0.960.

    ./venv/bin/python -m src.specialist --base v6_9fam_stack_aux --weight 0.3 --name v7_v6_plus_spec
"""
import argparse
import json

import numpy as np
import pandas as pd

from src.augment import FAMILIES, augmented_rows
from src.evaluate import make_model
from src.features import build_features
from src.load import load_split
from src.predict import SUBMISSIONS, to_unit_range, validate_submission


def specialist_scores(features=FAMILIES, seeds=(0, 1, 2), aug_weight=1.0):
    train, test = load_split("train"), load_split("test")
    Xtr = build_features("train", features, train)
    Xte = build_features("test", features, test)[Xtr.columns]
    Xa, _ = augmented_rows(train)
    Xa = Xa.reindex(columns=Xtr.columns) if set(Xtr.columns) <= set(Xa.columns) else Xa[Xtr.columns]
    X = np.vstack([Xtr.to_numpy(float), Xa.to_numpy(float)])
    y = np.r_[train.is_suspicious.to_numpy(), np.ones(len(Xa))]
    w = np.r_[np.ones(len(Xtr)), np.full(len(Xa), aug_weight)]
    s = np.zeros(len(Xte))
    for seed in seeds:
        s += make_model("lgbm", seed).fit(X, y, sample_weight=w).predict_proba(Xte.to_numpy(float))[:, 1]
    return pd.Series(s / len(seeds), index=test.hash.to_numpy()), test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="submission name whose scores get the specialist added")
    ap.add_argument("--weight", type=float, default=0.3)
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    spec, test = specialist_scores()
    SUBMISSIONS.mkdir(exist_ok=True)
    spec.rename("score").rename_axis("TRANSACTION_HASH").reset_index().to_csv(
        SUBMISSIONS / "specialist_raw.csv", index=False, float_format="%.8f")
    base = pd.read_csv(SUBMISSIONS / f"{a.base}.csv").set_index("TRANSACTION_HASH").score
    s = base.reindex(spec.index).to_numpy() + a.weight * spec.to_numpy()
    sub = pd.DataFrame({"TRANSACTION_HASH": spec.index, "score": to_unit_range(s)})
    validate_submission(sub, test)
    sub.to_csv(SUBMISSIONS / f"{a.name}.csv", index=False, float_format="%.8f")
    (SUBMISSIONS / f"{a.name}.json").write_text(json.dumps(
        {"base": a.base, "specialist_weight": a.weight, "specialist": "stage-1 LightGBM, 9 families, "
         "train + slowed easy copies (weight 1.0), 3 seeds", "blend": "base score + weight * specialist probability"}, indent=2))
    print(f"wrote {SUBMISSIONS / (a.name + '.csv')}")


if __name__ == "__main__":
    main()
