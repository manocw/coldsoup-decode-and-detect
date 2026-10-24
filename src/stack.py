"""Two-stage scoring: share evidence between linked transactions.

Labels are pure per planted scenario, so a transaction whose linked neighbours look suspicious is
more likely suspicious itself. Stage 1 scores every tx from its own features. Stage 2 adds, for each
tx, aggregates of the stage-1 scores of its neighbours (same sender / same receiver close in time,
and time-respecting hand-offs: the receiver paying on shortly after, the sender funded shortly
before), then re-scores.

Leakage control: stage 2 is trained on OUT-OF-FOLD stage-1 scores. The CV of the whole two-stage
model is nested: for each outer fold, stage-1 OOF scores are computed on the remaining folds only.

    ./venv/bin/python -m src.stack --features starter funding relays ... [--cv]
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.evaluate import cv_groups, make_model, metrics
from src.features.groups import _components

WINDOWS = (600, 3600, 6 * 3600)          # 10 min, 1 h, 6 h
GROUP_SCALES = (120, 600, 1800)          # one-off-pair motif groups linked within 2 / 10 / 30 min


def _index(p, col):
    """wallet -> positions of its txs in p (p is time-sorted, so positions are time-sorted)."""
    return {a: np.asarray(ix) for a, ix in p.groupby(col, sort=False).indices.items()}


RELATIONS = ("same_sender", "same_receiver", "downstream", "upstream")
_NB_CACHE = {}


def _neighbourhoods(p):
    """For every window and relation, CSR arrays (indptr, idx) of each tx's neighbours (itself
    excluded). Depends only on the week's payments, so it is built once per split and reused for
    every score that gets pooled."""
    key = (len(p), p.hash.iat[0], p.hash.iat[-1], p.ts.iat[len(p) // 2])
    if key in _NB_CACHE:
        return _NB_CACHE[key]
    ts = p.ts.to_numpy()
    by_sender, by_receiver = _index(p, "sender"), _index(p, "receiver")
    S, R = p.sender.to_numpy(), p.receiver.to_numpy()
    empty = np.empty(0, dtype=np.int64)
    out = {}
    for w in WINDOWS:
        lists = {r: [] for r in RELATIONS}
        for i in range(len(p)):
            t = ts[i]
            ix = by_sender.get(S[i], empty)
            lo, hi = np.searchsorted(ts[ix], t - w, "left"), np.searchsorted(ts[ix], t + w, "right")
            lists["same_sender"].append(ix[lo:hi][ix[lo:hi] != i])
            ix = by_receiver.get(R[i], empty)
            lo, hi = np.searchsorted(ts[ix], t - w, "left"), np.searchsorted(ts[ix], t + w, "right")
            lists["same_receiver"].append(ix[lo:hi][ix[lo:hi] != i])
            ix = by_sender.get(R[i], empty)             # receiver pays on within w
            lo, hi = np.searchsorted(ts[ix], t, "left"), np.searchsorted(ts[ix], t + w, "right")
            lists["downstream"].append(ix[lo:hi][ix[lo:hi] != i])
            ix = by_receiver.get(S[i], empty)           # sender was funded within w before
            lo, hi = np.searchsorted(ts[ix], t - w, "left"), np.searchsorted(ts[ix], t, "right")
            lists["upstream"].append(ix[lo:hi][ix[lo:hi] != i])
        for r, l in lists.items():
            counts = np.fromiter((len(a) for a in l), dtype=np.int64, count=len(l))
            indptr = np.r_[0, np.cumsum(counts)]
            idx = np.concatenate(l) if indptr[-1] else empty
            out[(w, r)] = (indptr, idx, counts)
    _NB_CACHE.clear()
    _NB_CACHE[key] = out
    return out


def _pool(s, indptr, idx, counts):
    """max, mean and mean of the top two of s over each segment; 0 for empty segments."""
    n = len(counts)
    mx, mn, t2 = np.zeros(n), np.zeros(n), np.zeros(n)
    nz = counts > 0
    if not nz.any():
        return mx, mn, t2
    vals = s[idx]
    seg = np.repeat(np.arange(n), counts)
    order = np.lexsort((vals, seg))
    v = vals[order]
    last = indptr[1:][nz] - 1
    mx[nz] = v[last]
    mn[nz] = np.add.reduceat(vals, indptr[:-1][nz]) / counts[nz]
    two = counts >= 2
    t2[nz] = v[last]
    t2[two] = (v[indptr[1:][two] - 1] + v[indptr[1:][two] - 2]) / 2
    return mx, mn, t2


def neighbor_features(p, s):
    """Aggregates of scores s (aligned with p) over each tx's neighbourhoods, excluding itself."""
    nbh = _neighbourhoods(p)
    out = {}
    for w in WINDOWS:
        tag = f"{w // 60}m"
        for r in RELATIONS:
            mx, mn, t2 = _pool(s, *nbh[(w, r)])
            out[f"{r}_max_{tag}"], out[f"{r}_mean_{tag}"], out[f"{r}_top2_{tag}"] = mx, mn, t2
    # motif groups: one-off payments linked through shared wallets within dt (see features.groups)
    pair = p.sender + ">" + p.receiver
    one_off = (pair.map(pair.value_counts()) == 1).to_numpy()
    q = p[one_off]
    for dt in GROUP_SCALES:
        comp = _components(q.sender.to_numpy(), q.receiver.to_numpy(), q.ts.to_numpy(), dt)
        g = pd.DataFrame({"c": comp, "s": s[one_off]}).groupby("c").s
        size, total, top = g.transform("size"), g.transform("sum"), g.transform("max")
        others = (total - s[one_off]) / (size - 1).replace(0, np.nan)
        for name, vals in (("group_mean_others", others), ("group_max", top), ("group_size", size)):
            v = np.full(len(p), np.nan)
            v[one_off] = vals.to_numpy()
            out[f"{name}_{dt // 60}m"] = v
    f = pd.DataFrame(out, index=p.index)
    f["own"] = s
    return f


def _fit_scores(X, y, train_idx, pred_idx, kind, seed):
    m = make_model(kind, seed).fit(X[train_idx], y[train_idx])
    return m.predict_proba(X[pred_idx])[:, 1]


def oof_scores(X, y, groups, rows, kind="lgbm", seed=0, n_splits=5):
    """Out-of-fold stage-1 scores for the given rows only (folds formed inside them)."""
    s = np.zeros(len(y))
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=100 + seed)
    for tr, te in skf.split(X[rows], y[rows], groups[rows]):
        s[rows[te]] = _fit_scores(X, y, rows[tr], rows[te], kind, seed)
    return s


AUX_DEFAULT = ("relay", "motif")


def aux_targets(p, names=AUX_DEFAULT):
    """Auxiliary training targets from the annotations (targets only, never features): is the
    payment part of a relay cohort, and is it part of any non-relay planted motif (suspicious or
    legitimate look-alike). They have ~20k / ~2k positives, so stage 1 learns what relays and
    motifs look like far better than from the 1,250 suspicious labels alone. Optional "medium":
    is it a medium suspicious payment (225), so slow suspicious motifs get a scorer of their own
    instead of being drowned by the "fast = suspicious" pattern of the 800 easy ones."""
    relay = p.cohort_id.notna().to_numpy()
    motif = p.scenario_id.notna().to_numpy() & ~relay & (p.family != "ambiguous").to_numpy()
    medium = (p.is_suspicious == 1).to_numpy() & (p.difficulty == "medium").to_numpy()
    targets = {"relay": relay, "motif": motif, "medium": medium}
    return {n: targets[n].astype(int) for n in names}


def stage2_matrix(X, p, s, mode, aux=None):
    blocks = [neighbor_features(p, s).to_numpy()]
    for a in (aux or {}).values():
        blocks.append(neighbor_features(p, a).to_numpy())
    nb = np.hstack(blocks)
    return nb if mode == "scores" else np.hstack([X, nb])


def _aux_scores(X, targets, groups, tr, te, kind, seed):
    """OOF aux scores inside the training rows tr, full-fit scores for the rows te."""
    out = {}
    for name, ya in targets.items():
        a = oof_scores(X, ya, groups, tr, kind, seed)
        a[te] = _fit_scores(X, ya, tr, te, kind, seed)
        out[name] = a
    return out


def nested_cv(Xdf, p, kind="lgbm", seed=0, mode="full", n_splits=5, aux=False):
    """Honest CV of stage 1 and stage 2 with identical outer folds."""
    X = Xdf.to_numpy(float)
    y = p.is_suspicious.to_numpy()
    groups = cv_groups(p)
    targets = aux_targets(p, AUX_DEFAULT if aux is True else aux) if aux else {}
    s1 = np.zeros(len(y))
    s2 = np.zeros(len(y))
    outer = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for k, (tr, te) in enumerate(outer.split(X, y, groups)):
        # stage-1 scores: OOF inside the training folds, a full-train model for the held-out fold
        s_in = oof_scores(X, y, groups, tr, kind, seed)
        s_in[te] = _fit_scores(X, y, tr, te, kind, seed)
        s1[te] = s_in[te]
        a_in = _aux_scores(X, targets, groups, tr, te, kind, seed)
        Z = stage2_matrix(X, p, s_in, mode, a_in)
        s2[te] = make_model(kind, seed).fit(Z[tr], y[tr]).predict_proba(Z[te])[:, 1]
        print(f"  outer fold {k + 1}/{n_splits} done", flush=True)
    return s1, s2


def _pseudo(s, pos_k, neg_q):
    n = len(s)
    order = np.argsort(-s, kind="stable")
    return order[:int(round(pos_k * n))], order[n - int(round(neg_q * n)):]


def self_train(Ztr, y, Zte, s_te, kind, seed, rounds=2, pos_k=0.015, neg_q=0.5, Zex=None, s_ex=None):
    """Transductive self-training of the final scorer on the target split: its top pos_k scores
    become pseudo-positives and its bottom neg_q pseudo-negatives, the model is refit on train plus
    those rows and the target re-scored; repeated `rounds` times. The target's labels are never
    used. In the stage-1 CV simulation (src/selftrain.py, 2 seeds) this setting gained +1.0 AP.
    Zex/s_ex: an extra unlabelled split (the validation week), pseudo-labelled the same way."""
    for _ in range(rounds):
        pos, neg = _pseudo(s_te, pos_k, neg_q)
        Z = [Ztr, Zte[pos], Zte[neg]]
        yy = [y, np.ones(len(pos)), np.zeros(len(neg))]
        if Zex is not None:
            epos, eneg = _pseudo(s_ex, pos_k, neg_q)
            Z += [Zex[epos], Zex[eneg]]
            yy += [np.ones(len(epos)), np.zeros(len(eneg))]
        m = make_model(kind, seed).fit(np.vstack(Z), np.concatenate(yy))
        s_te = m.predict_proba(Zte)[:, 1]
        if Zex is not None:
            s_ex = m.predict_proba(Zex)[:, 1]
    return s_te


def fit_predict(Xtr_df, ptr, Xte_df, pte, kind="lgbm", seeds=(0, 1, 2), mode="full", aux=False,
                selftrain_rounds=0, Xex_df=None, pex=None):
    Xtr, Xte = Xtr_df.to_numpy(float), Xte_df.to_numpy(float)
    y = ptr.is_suspicious.to_numpy()
    groups = cv_groups(ptr)
    rows = np.arange(len(y))
    targets = aux_targets(ptr, AUX_DEFAULT if aux is True else aux) if aux else {}
    out = np.zeros(len(Xte))
    for seed in seeds:
        s_tr = oof_scores(Xtr, y, groups, rows, kind, seed)
        s_te = make_model(kind, seed).fit(Xtr, y).predict_proba(Xte)[:, 1]
        a_tr = {n: oof_scores(Xtr, ya, groups, rows, kind, seed) for n, ya in targets.items()}
        a_te = {n: make_model(kind, seed).fit(Xtr, ya).predict_proba(Xte)[:, 1] for n, ya in targets.items()}
        Ztr, Zte = stage2_matrix(Xtr, ptr, s_tr, mode, a_tr), stage2_matrix(Xte, pte, s_te, mode, a_te)
        m2 = make_model(kind, seed).fit(Ztr, y)
        s2 = m2.predict_proba(Zte)[:, 1]
        if selftrain_rounds:
            Zex = s2_ex = None
            if Xex_df is not None:
                Xex = Xex_df.to_numpy(float)
                s1_ex = make_model(kind, seed).fit(Xtr, y).predict_proba(Xex)[:, 1]
                a_ex = {n: make_model(kind, seed).fit(Xtr, ya).predict_proba(Xex)[:, 1] for n, ya in targets.items()}
                Zex = stage2_matrix(Xex, pex, s1_ex, mode, a_ex)
                s2_ex = m2.predict_proba(Zex)[:, 1]
            s2 = self_train(Ztr, y, Zte, s2, kind, seed, rounds=selftrain_rounds, Zex=Zex, s_ex=s2_ex)
        out += s2
    return out / len(seeds)


if __name__ == "__main__":
    import argparse

    from src.features import build_features
    from src.load import load_split

    ap = argparse.ArgumentParser()
    ap.add_argument("--features", nargs="+", required=True)
    ap.add_argument("--mode", default="full", choices=["full", "scores"])
    ap.add_argument("--model", default="lgbm")
    ap.add_argument("--aux", action="store_true", help="add auxiliary relay/motif targets to stage 1")
    a = ap.parse_args()
    p = load_split("train")
    X = build_features("train", a.features, p)
    s1, s2 = nested_cv(X, p, a.model, mode=a.mode, aux=a.aux)
    for name, s in (("stage1", s1), (f"stage2[{a.mode}]", s2)):
        m = metrics(s, p)
        print(f"[{name}] " + "  ".join(f"{k}={v:.3f}" if v < 1.5 else f"{k}={v:.1f}" for k, v in m.items()))
