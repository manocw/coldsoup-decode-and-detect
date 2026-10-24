"""Transductive self-training (pseudo-labelling) for stage 1.

The rules allow using the unlabelled target week. Score the target split with the plain stage-1 model,
take its most confident payments as pseudo-labels (top `pos_k` as suspicious, bottom `neg_q` as legit),
add them to the labelled training rows, retrain and re-score. The unlabelled validation week can be
pseudo-labelled the same way and added as extra training data (`Xextra`).

CV simulation (honest, labels of the target fold are only used to score at the end):

    XRPL_NJOBS=3 ./venv/bin/python -m src.selftrain [--seed 0] [--no-val]

Outer StratifiedGroupKFold(5) with src.evaluate.cv_groups; each held-out fold plays the test week:
train on the other folds, pseudo-label inside the held-out fold only, retrain, re-score. The plain
stage-1 baseline uses the same folds and model seeds (it is the round-0 model of every variant).
"""
import time

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from src.evaluate import cv_groups, make_model, metrics

FEATURES = "starter funding relays fields motifs hops amounts history cycles".split()


def _count(k, n):
    """k >= 1 is a row count, 0 <= k < 1 a fraction of the n target rows."""
    return min(n, int(k) if k >= 1 else int(round(k * n)))


def pseudo_labels(s, pos_k, neg_q):
    """Positions and hard labels of the confident rows of one unlabelled split scored s."""
    n = len(s)
    order = np.argsort(-s, kind="stable")
    k, q = _count(pos_k, n), _count(neg_q, n)
    pos, neg = order[:k], order[n - q:] if q else order[:0]
    return np.concatenate([pos, neg]), np.r_[np.ones(len(pos)), np.zeros(len(neg))]


def _fit(X, y, w, kind, seed):
    return make_model(kind, seed).fit(X, y, sample_weight=w)


def self_train_one(Xtr, ytr, Xte, seed, rounds=1, pos_k=0.005, neg_q=0.5, weight=1.0, Xex=None,
                   use_target=True, kind="lgbm", s_te=None, s_ex=None, info=None):
    """One model seed. s_te / s_ex: round-0 scores if already computed (skips the plain fit).
    Returns the target scores after each round (index 0 = plain stage-1 model)."""
    if s_te is None or (Xex is not None and s_ex is None):
        m = _fit(Xtr, ytr, None, kind, seed)
        s_te = m.predict_proba(Xte)[:, 1]
        s_ex = None if Xex is None else m.predict_proba(Xex)[:, 1]
    hist = [s_te]
    for r in range(1, rounds + 1):
        Xs, ys = [Xtr], [ytr]
        if use_target:
            ix, yy = pseudo_labels(s_te, pos_k, neg_q)
            Xs.append(Xte[ix]), ys.append(yy)
            if info is not None:
                info.append((seed, r, ix, yy))
        if Xex is not None:
            ix, yy = pseudo_labels(s_ex, pos_k, neg_q)
            Xs.append(Xex[ix]), ys.append(yy)
        w = np.r_[np.ones(len(ytr)), np.full(sum(len(v) for v in ys[1:]), float(weight))]
        m = _fit(np.vstack(Xs), np.concatenate(ys), w, kind, seed)
        del Xs
        s_te = m.predict_proba(Xte)[:, 1]
        s_ex = None if Xex is None else m.predict_proba(Xex)[:, 1]
        hist.append(s_te)
    return hist


def fit_predict_selftrain(Xtr, ytr, Xte, rounds=1, pos_k=0.005, neg_q=0.5, weight=1.0, Xextra=None,
                          use_target=True, kind="lgbm", seeds=(0,), return_rounds=False, info=None):
    """Self-trained stage-1 scores for the target rows Xte (averaged over model seeds).

    rounds    : retrain/re-score cycles (pseudo-labels re-selected from the latest scores each round)
    pos_k     : top rows taken as pseudo-positive (fraction of the split if < 1, else a count)
    neg_q     : bottom rows taken as pseudo-negative (same convention)
    weight    : sample weight of every pseudo-labelled row (labelled rows weigh 1)
    Xextra    : optional extra unlabelled split (e.g. validation), pseudo-labelled the same way
    use_target: pseudo-label the target itself (False = only Xextra is pseudo-labelled)
    return_rounds: also return the target scores after each round (round 0 = plain stage-1 model)
    info      : optional list; per seed and round, (seed, round, target positions, pseudo-labels)
    """
    Xtr, Xte = np.asarray(Xtr, float), np.asarray(Xte, float)
    Xex = None if Xextra is None else np.asarray(Xextra, float)
    ytr = np.asarray(ytr, float)
    hist = np.zeros((rounds + 1, len(Xte)))
    for seed in seeds:
        hist += np.array(self_train_one(Xtr, ytr, Xte, seed, rounds, pos_k, neg_q, weight, Xex,
                                        use_target, kind, info=info))
    hist /= len(seeds)
    return (hist[-1], list(hist)) if return_rounds else hist[-1]


# The few settings measured by the CV simulation (not a sweep).
SETTINGS = {
    "hard_r1": dict(rounds=1, pos_k=0.005, neg_q=0.5),          # top 0.5% / bottom 50%, one round
    "wide_r2": dict(rounds=2, pos_k=0.015, neg_q=0.5),          # top 1.5% (half the base rate), two rounds
    "hard_r1+val": dict(rounds=1, pos_k=0.005, neg_q=0.5, val=True),
    "val_only": dict(rounds=1, pos_k=0.005, neg_q=0.5, val=True, use_target=False),
}
# Follow-ups (select with --only): the two settings positive on both seeds combined, and which half of
# the validation pseudo-labels (positives or negatives) carries the val_only gain.
EXTRA = {
    "wide_r2+val": dict(rounds=2, pos_k=0.015, neg_q=0.5, val=True),
    "val_pos_only": dict(rounds=1, pos_k=0.005, neg_q=0, val=True, use_target=False),
    "val_neg_only": dict(rounds=1, pos_k=0, neg_q=0.5, val=True, use_target=False),
}


def _fmt(m):
    return "  ".join(f"{k}={v:.3f}" if v < 1.5 else f"{k}={v:.1f}" for k, v in m.items())


def cv_simulation(X, p, Xval=None, seed=0, kind="lgbm", n_splits=5, settings=SETTINGS, folds=None):
    """Scores of the baseline and every setting on the held-out folds (only `folds` if given)."""
    y = p.is_suspicious.to_numpy()
    groups = cv_groups(p)
    names = ["baseline"] + [n for n, s in settings.items() if Xval is not None or not s.get("val")]
    scores = {n: np.zeros(len(y)) for n in names}
    diag = {n: [] for n in names[1:]}
    covered = np.zeros(len(y), bool)
    outer = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for k, (tr, te) in enumerate(outer.split(X, y, groups)):
        if folds is not None and k not in folds:
            continue
        t0 = time.time()
        covered[te] = True
        Xtr, ytr, Xte = X[tr], y[tr].astype(float), X[te]
        m0 = _fit(Xtr, ytr, None, kind, seed)                 # plain stage 1 = round 0 of every variant
        s_te = m0.predict_proba(Xte)[:, 1]
        s_ex = None if Xval is None else m0.predict_proba(Xval)[:, 1]
        del m0
        scores["baseline"][te] = s_te
        for n in names[1:]:
            cfg = dict(settings[n])
            use_val = cfg.pop("val", False)
            info = []
            hist = self_train_one(Xtr, ytr, Xte, seed, Xex=Xval if use_val else None, kind=kind,
                                  s_te=s_te, s_ex=s_ex if use_val else None, info=info, **cfg)
            scores[n][te] = hist[-1]
            for _, r, ix, yy in info:                 # pseudo-label quality (diagnostic only)
                pos, neg = ix[yy == 1], ix[yy == 0]
                diag[n].append((k, r, len(pos), y[te][pos].mean(), y[te][neg].sum(), y[te].sum()))
        print(f"  outer fold {k + 1}/{n_splits} done ({time.time() - t0:.0f}s)", flush=True)
    return scores, diag, covered


def report(scores, diag, p):
    base = metrics(scores["baseline"], p)
    print(f"[baseline] {_fmt(base)}")
    for n, s in scores.items():
        if n == "baseline":
            continue
        m = metrics(s, p)
        d = {k: m[k] - base[k] for k in ("AP", "AP_easy", "AP_medium", "AP_hard", "AP_relay_cohorts")}
        print(f"[{n}] {_fmt(m)}")
        print("    delta vs baseline: " + "  ".join(f"{k}={v:+.2f}" for k, v in d.items()))
        dd = np.asarray(diag.get(n, []), dtype=float)
        if len(dd):
            for r in np.unique(dd[:, 1]):
                q = dd[dd[:, 1] == r]
                print(f"    round {int(r)}: pseudo-pos {q[:, 2].mean():.0f}/fold, precision {q[:, 3].mean():.3f}; "
                      f"true positives among pseudo-neg {q[:, 4].mean():.1f} of {q[:, 5].mean():.0f}/fold")


if __name__ == "__main__":
    import argparse
    import gc

    from src.features import build_features
    from src.load import load_split

    ap = argparse.ArgumentParser()
    ap.add_argument("--features", nargs="+", default=FEATURES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-val", action="store_true", help="skip the variants using the validation split")
    ap.add_argument("--only", nargs="+", default=None, help="subset of SETTINGS to run")
    ap.add_argument("--folds", nargs="+", type=int, default=None, help="run only these outer folds (0-4)")
    ap.add_argument("--save", default=None, help="write scores/diagnostics of the run to this .npz")
    ap.add_argument("--report", nargs="+", default=None,
                    help="merge saved partial runs (.npz from --folds/--save) and print the report")
    a = ap.parse_args()

    if a.report:
        p = load_split("train")
        scores, diag, covered = {}, {}, np.zeros(len(p), bool)
        for path in a.report:
            z = np.load(path)
            c = z["covered"]
            covered |= c
            for n in z.files:
                if n.startswith("diag__"):
                    diag.setdefault(n[6:], []).extend(z[n].tolist())
                elif n != "covered":
                    scores.setdefault(n, np.zeros(len(p)))[c] = z[n][c]
        assert covered.all(), f"{(~covered).sum()} rows not covered by the saved runs"
        report(scores, diag, p)
        raise SystemExit

    Xval = None
    cols = None
    if not a.no_val:
        pv = load_split("validation")
        Xv = build_features("validation", a.features, pv)
        cols = list(Xv.columns)
        Xval = Xv.to_numpy(float)
        del pv, Xv
        gc.collect()
    p = load_split("train")
    Xdf = build_features("train", a.features, p)
    if cols is not None:
        Xdf = Xdf[cols]
    X = Xdf.to_numpy(float)
    del Xdf
    gc.collect()
    settings = {n: {**SETTINGS, **EXTRA}[n] for n in (a.only or SETTINGS)}
    print(f"train {X.shape}, validation {None if Xval is None else Xval.shape}, seed {a.seed}", flush=True)
    scores, diag, covered = cv_simulation(X, p, Xval, seed=a.seed, settings=settings, folds=a.folds)
    if a.save:
        np.savez(a.save, covered=covered, **scores,
                 **{f"diag__{n}": np.asarray(v, float).reshape(-1, 6) for n, v in diag.items()})
    if covered.all():
        report(scores, diag, p)
