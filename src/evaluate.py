"""Honest offline scoring on train: grouped cross-validation with the site's metric.

Folds keep whole relay cohorts and whole scenarios together, so sibling transactions of one
planted motif never sit on both sides of a split (random row splits leak them and inflate AP).
Reports overall AP (= site score / 100), top-1% precision/recall, and AP restricted to each
difficulty (that difficulty's positives against all negatives).

Usage from code:
    from src.evaluate import cv_report
    rep = cv_report(X, p)            # X row-aligned with p = load_split("train")
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

LABEL_COLS = ["is_suspicious", "family", "difficulty", "scenario_id", "cohort_id"]


def unlabelled(p):
    """The table with every annotation column removed: the only thing feature code may see."""
    return p.drop(columns=[c for c in LABEL_COLS if c in p.columns])


def cv_groups(p):
    g = p.cohort_id.fillna(p.scenario_id).fillna(p.hash)
    return pd.factorize(g)[0]


LGBM_DEFAULTS = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=20,
                     subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0)


def lgbm_params():
    """Defaults, overridden by the JSON in $XRPL_LGBM (set by src.predict --params)."""
    return {**LGBM_DEFAULTS, **json.loads(os.environ.get("XRPL_LGBM", "{}"))}


def make_model(kind="lgbm", seed=0):
    if kind == "lgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(**lgbm_params(), random_state=seed,
                                  n_jobs=int(os.environ.get("XRPL_NJOBS", 8)), verbose=-1)
    if kind == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=seed)
    if kind == "logreg":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))
    raise ValueError(kind)


def oof_predict(X, p, kind="lgbm", seeds=(0, 1), n_splits=5):
    """Out-of-fold scores averaged over repeated grouped splits."""
    y = p.is_suspicious.to_numpy()
    groups = cv_groups(p)
    X = X.to_numpy(dtype=float) if hasattr(X, "to_numpy") else np.asarray(X, float)
    oof = np.zeros(len(y))
    for seed in seeds:
        skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for tr, te in skf.split(X, y, groups):
            m = make_model(kind, seed).fit(X[tr], y[tr])
            oof[te] += m.predict_proba(X[te])[:, 1]
    return oof / len(seeds)


def metrics(scores, p):
    y = p.is_suspicious.to_numpy()
    out = {"AP": 100 * average_precision_score(y, scores)}
    k = max(1, int(round(0.01 * len(y))))
    order = np.lexsort((p.hash.to_numpy(), -scores))[:k]
    out["top1%_precision"] = y[order].mean()
    out["top1%_recall"] = y[order].sum() / y.sum()
    neg = y == 0
    for d in ("easy", "medium", "hard"):
        pos = (p.difficulty.to_numpy() == d) & (y == 1)
        m = pos | neg
        out[f"AP_{d}"] = 100 * average_precision_score(y[m], scores[m])
    relay = p.cohort_id.notna().to_numpy()
    out["AP_relay_cohorts"] = 100 * average_precision_score(y[relay], scores[relay])
    return out


def family_table(scores, p):
    """Median rank percentile of positives (lower = better) and top-5% false positives per family."""
    r = pd.DataFrame({"y": p.is_suspicious.to_numpy(), "s": scores,
                      "fam": p.family.to_numpy(), "diff": p.difficulty.to_numpy()})
    r["rank_pct"] = r.s.rank(ascending=False, pct=True)
    pos = r[r.y == 1].groupby(["fam", "diff"]).rank_pct.median().unstack().round(3)
    fp = r.nlargest(int(0.05 * len(r)), "s").query("y == 0").fam.value_counts().rename("top5%_FP")
    return pos, fp


def cv_report(X, p, kind="lgbm", seeds=(0, 1), verbose=True, label=""):
    oof = oof_predict(X, p, kind, seeds)
    m = metrics(oof, p)
    if verbose:
        print(f"[{label or kind}] " + "  ".join(f"{k}={v:.3f}" if v < 1.5 else f"{k}={v:.1f}" for k, v in m.items()))
    return m, oof
