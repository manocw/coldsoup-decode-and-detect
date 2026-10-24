"""Measure feature families with grouped CV on train.

    ./venv/bin/python -m src.ablate starter funding          # starter, starter+funding, funding alone
    ./venv/bin/python -m src.ablate starter funding --only   # just the full combination
    XRPL_NJOBS=2 ./venv/bin/python -m src.ablate ...         # fewer threads when sharing the CPU

The first name is the base set; every later name is reported as base+name and name alone.
The base set alone is only re-run with --base (starter alone, 1 seed: AP 33.7). One seed by
default keeps a run to 5 model fits; use --seeds 2 for a final, steadier number.
"""
import argparse

import pandas as pd

from src.evaluate import cv_report, family_table
from src.features import build_features
from src.load import load_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--model", default="lgbm")
    ap.add_argument("--only", action="store_true", help="only evaluate all names together")
    ap.add_argument("--families", action="store_true", help="print per-family rank table")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--base", action="store_true", help="also re-run the base set alone")
    ap.add_argument("--seeds", type=int, default=1)
    a = ap.parse_args()

    p = load_split("train")
    feats = {n: build_features("train", [n], p, use_cache=not a.no_cache) for n in a.names}
    runs = [("+".join(a.names), a.names)] if a.only else (
        ([(a.names[0], [a.names[0]])] if a.base or len(a.names) == 1 else [])
        + [(f"{a.names[0]}+{n}", [a.names[0], n]) for n in a.names[1:]]
        + [(n, [n]) for n in a.names[1:]])
    for label, names in runs:
        X = pd.concat([feats[n] for n in names], axis=1)
        _, oof = cv_report(X, p, a.model, seeds=tuple(range(a.seeds)), label=label)
        if a.families:
            pos, fp = family_table(oof, p)
            print(pos.to_string())
            print("top-5% false positives:", fp.to_dict())


if __name__ == "__main__":
    main()
