"""Motif reconstruction: group transactions into candidate motif instances, score the group.

Labels are pure per planted scenario, so evidence aggregated over a reconstructed instance is
stronger than any single payment. Grouping is unsupervised and uses behaviour only:

* every wallet's timeline (its sends and receipts, time-ordered) links consecutive events that
  are at most dt apart: hand-offs (receive then send), fan-out (send, send), fan-in
  (receive, receive) and returns (send, receive);
* links are merged shortest-gap first (single linkage) with a size cap, so busy wallets
  cannot glue the week into one giant component;
* the longer scales only use first-time counterparty pairs (the pair occurs once in the
  week): planted motifs almost never reuse a pair, recurring commerce does, and dropping the
  recurring pairs is what stops long-dt linking from percolating.

Each transaction then gets its component's aggregates (size, wallets, span, hop delays,
star/chain shape, value retention through intermediaries, amount spread) and its position
inside the component, at three time scales.
"""
import numpy as np
import pandas as pd

CAP = 30
SCALES = (("s5", 300, False), ("s30", 1800, True), ("s120", 7200, True))


def _edges(sender, receiver, t, keep):
    """Consecutive events of each wallet among the kept transactions: (a, b, gap, type)."""
    idx = np.flatnonzero(keep)
    k = len(idx)
    w = np.concatenate([sender[idx], receiver[idx]])
    tt = np.concatenate([t[idx], t[idx]])
    ii = np.concatenate([idx, idx])
    role = np.concatenate([np.ones(k, np.int8), np.zeros(k, np.int8)])  # 1 = send, 0 = receive
    order = np.lexsort((ii, tt, w))
    w, tt, ii, role = w[order], tt[order], ii[order], role[order]
    same = w[1:] == w[:-1]
    a, b = ii[:-1][same], ii[1:][same]
    gap = (tt[1:] - tt[:-1])[same]
    typ = (role[:-1] * 2 + role[1:])[same]  # 0 in-in, 1 in-out (hand-off), 2 out-in, 3 out-out
    ok = a != b
    return a[ok], b[ok], gap[ok], typ[ok]


def _capped_components(n, a, b, gap, cap):
    """Single linkage in increasing-gap order, refusing merges that exceed `cap` transactions."""
    parent = np.arange(n)
    size = np.ones(n, np.int64)

    def find(x):
        r = x
        while parent[r] != r:
            r = parent[r]
        while parent[x] != r:
            parent[x], x = r, parent[x]
        return r

    for k in np.argsort(gap, kind="stable"):
        ra, rb = find(a[k]), find(b[k])
        if ra != rb and size[ra] + size[rb] <= cap:
            if size[ra] < size[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            size[ra] += size[rb]
    return np.array([find(x) for x in range(n)])


def _component_features(p, comp):
    n = len(p)
    d = pd.DataFrame({"c": comp, "t": p.ts.to_numpy(), "s": p.sender.to_numpy(),
                      "r": p.receiver.to_numpy(), "v": p.dst_usd.fillna(0).to_numpy(),
                      "i": np.arange(n)})
    g = d.groupby("c", sort=False)
    size = g.i.transform("size").to_numpy()
    tmin, tmax = g.t.transform("min").to_numpy(), g.t.transform("max").to_numpy()
    span = (tmax - tmin) / 60.0

    # gaps between consecutive transactions of the component
    ds = d.sort_values(["c", "t", "i"], kind="mergesort")
    gp = ds.t.diff().where(ds.c.eq(ds.c.shift())) / 60.0
    gstat = gp.groupby(ds.c).agg(["median", "mean", "std"])
    med_gap = d.c.map(gstat["median"]).to_numpy()
    cv_gap = d.c.map(gstat["std"] / (gstat["mean"] + 1e-9)).to_numpy()
    rank = ds.groupby("c").cumcount().reindex(d.index).to_numpy()  # ds index == original index
    pos = np.where(size > 1, rank / np.maximum(size - 1, 1), np.nan)

    # wallets: senders, receivers, intermediaries (both send and receive inside the component)
    outs = d.groupby(["c", "s"]).v.agg(["size", "sum"])
    ins = d.groupby(["c", "r"]).v.agg(["size", "sum"])
    outs.index.names = ins.index.names = ["c", "w"]
    both = outs.join(ins, how="outer", lsuffix="_o", rsuffix="_i").fillna(0)
    both["inter"] = (both.size_o > 0) & (both.size_i > 0)
    both["ratio"] = np.where(both.inter, both.sum_o / np.maximum(both.sum_i, 1e-9), np.nan)
    wg = both.groupby(level="c")
    n_wallets = d.c.map(wg.size()).to_numpy()
    n_inter = d.c.map(wg.inter.sum()).to_numpy()
    retention = d.c.map(wg.ratio.median()).to_numpy()
    max_out = d.c.map(outs["size"].groupby(level="c").max()).to_numpy()
    max_in = d.c.map(ins["size"].groupby(level="c").max()).to_numpy()

    vstat = g.v.agg(["mean", "std", "median"])
    amt_cv = d.c.map(vstat["std"] / (vstat["mean"] + 1e-9)).to_numpy()
    amt_med = d.c.map(vstat["median"]).to_numpy()

    # this transaction's own role in the component
    key_s = pd.MultiIndex.from_arrays([d.c, d.s])
    key_r = pd.MultiIndex.from_arrays([d.c, d.r])
    sender_received = both.size_i.reindex(key_s).to_numpy() > 0
    receiver_sends = both.size_o.reindex(key_r).to_numpy() > 0

    # how many multi-transaction components each wallet joins during the week
    multi = size >= 3
    part = pd.concat([d.loc[multi, ["c", "s"]].rename(columns={"s": "w"}),
                      d.loc[multi, ["c", "r"]].rename(columns={"r": "w"})]).drop_duplicates()
    wcount = part.w.value_counts()
    s_ncomp = d.s.map(wcount).fillna(0).to_numpy()
    r_ncomp = d.r.map(wcount).fillna(0).to_numpy()

    f = {
        "size": size,
        "n_wallets": n_wallets,
        "span_min": span,
        "med_gap_min": med_gap,
        "cv_gap": cv_gap,
        "rate": size / (span + 1.0),
        "inter_frac": n_inter / n_wallets,
        "max_out_share": max_out / size,
        "max_in_share": max_in / size,
        "retention": retention,
        "amt_cv": amt_cv,
        "log_amt_med": np.log1p(amt_med),
        "pos": pos,
        "sender_received": sender_received.astype(float),
        "receiver_sends": receiver_sends.astype(float),
        "sender_ncomp": s_ncomp,
        "receiver_ncomp": r_ncomp,
    }
    return pd.DataFrame(f)


def build(p):
    n = len(p)
    sender, receiver = p.sender.to_numpy(), p.receiver.to_numpy()
    t = p.ts.to_numpy()
    pair = pd.Series(sender) + ">" + pd.Series(receiver)
    novel = (pair.map(pair.value_counts()) == 1).to_numpy()

    out = [pd.DataFrame({"novel_pair": novel.astype(float)})]
    for name, dt, novel_only in SCALES:
        keep = novel if novel_only else np.ones(n, bool)
        a, b, gap, _ = _edges(sender, receiver, t, keep)
        m = gap <= dt
        comp = _capped_components(n, a[m], b[m], gap[m], CAP)
        f = _component_features(p, comp)
        if novel_only:
            f.loc[~novel] = np.nan
        f.columns = [f"{name}_{c}" for c in f.columns]
        out.append(f)
    return pd.concat(out, axis=1).astype(float)
