"""Augmentation: slow easy suspicious motifs down to medium pace to teach "slow but suspicious".

Medium suspicious motifs are the easy motifs slowed down (flat amounts, steps 15-40 min apart instead
of ~1 min), and only 27 medium scenarios exist in train, so the model learns "fast = suspicious" from
the 800 easy payments and ranks medium below the legitimate look-alikes. Here every easy suspicious
(non-relay) scenario is copied into an altered week where its steps are re-timed to medium pace
(same order, same wallets and amounts; the original is removed so its pairs stay one-off). All feature
families are recomputed on that altered week and the slowed copies are added as extra positives.

    XRPL_NJOBS=8 ./venv/bin/python -m src.augment            # grouped-CV check vs the plain stage 1
"""
import numpy as np
import pandas as pd

from src.evaluate import LABEL_COLS, cv_groups, make_model, metrics, unlabelled
from src.features import _module
from src.load import load_split

FAMILIES = "starter funding relays fields motifs hops amounts history cycles".split()
GAP_MIN, GAP_MAX = 10 * 60, 35 * 60


def slowed_week(p, seed=0, gap=(GAP_MIN, GAP_MAX)):
    """Train week with every easy suspicious non-relay scenario re-timed to medium pace."""
    rng = np.random.default_rng(seed)
    p = p.copy()
    p["aug"] = False
    easy = p[(p.is_suspicious == 1) & (p.difficulty == "easy") & p.cohort_id.isna()]
    new_ts = p.ts.copy()
    for sid, g in easy.groupby("scenario_id"):
        g = g.sort_values(["ts", "tx_index"])
        steps = rng.uniform(*gap, size=len(g) - 1)
        new_ts.loc[g.index] = g.ts.iloc[0] + np.r_[0, np.cumsum(steps)].astype(np.int64)
    p.loc[easy.index, "aug"] = True
    p["ts"] = new_ts
    p["hash"] = np.where(p.aug, p.hash + "_slow", p.hash)
    return p.sort_values(["ts", "ledger", "tx_index", "hash"], kind="mergesort").reset_index(drop=True)


def features_for(p, names=FAMILIES):
    """All families for an arbitrary (unlabelled) table; no cache, column names as in build_features."""
    u = unlabelled(p).drop(columns=["aug"], errors="ignore")
    parts = []
    for n in names:
        f = _module(n).build(u).reset_index(drop=True).astype(float)
        f.columns = [f"{n}.{c}" for c in f.columns]
        parts.append(f)
    return pd.concat(parts, axis=1)


def augmented_rows(p, seed=0):
    """Features and labels of the slowed copies, with the scenario id of their original."""
    w = slowed_week(p, seed)
    F = features_for(w)
    keep = w.aug.to_numpy()
    return F[keep].reset_index(drop=True), w.loc[keep, "scenario_id"].to_numpy()


if __name__ == "__main__":
    import time
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold

    from src.features import build_features

    t0 = time.time()
    p = load_split("train")
    X = build_features("train", FAMILIES, p)
    Xa, sid_a = augmented_rows(p)
    Xa = Xa[X.columns]
    print(f"slowed copies: {len(Xa)} payments from {len(set(sid_a))} easy scenarios ({time.time() - t0:.0f}s)")
    y = p.is_suspicious.to_numpy()
    groups = cv_groups(p)
    Xn, Xan = X.to_numpy(float), Xa.to_numpy(float)
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(Xn, y, groups))
    d = p.difficulty.to_numpy()
    for weight in (0.0, 0.5, 1.0):
        oof = np.zeros(len(y))
        for tr, te in folds:
            train_sids = set(p.scenario_id.iloc[tr].dropna())
            use = np.array([s in train_sids for s in sid_a]) if weight > 0 else np.zeros(len(Xan), bool)
            Xt = np.vstack([Xn[tr], Xan[use]])
            yt = np.r_[y[tr], np.ones(use.sum())]
            wt = np.r_[np.ones(len(tr)), np.full(use.sum(), weight)]
            m = make_model("lgbm", 0).fit(Xt, yt, sample_weight=wt)
            oof[te] = m.predict_proba(Xn[te])[:, 1]
        r = metrics(oof, p)
        neg = y == 0
        for dd in ("medium", "hard"):
            mk = neg | ((d == dd) & (y == 1))
            r[f"AUC_{dd}"] = roc_auc_score(y[mk], oof[mk])
        print(f"[aug weight {weight}] " + "  ".join(f"{k}={v:.3f}" for k, v in r.items()), flush=True)
