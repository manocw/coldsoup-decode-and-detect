"""Motif speed: how fast does the group of one-off payments around this payment move?

Planted motifs (suspicious and legitimate look-alikes) are made of one-off relationships: 95-97%
of their payments use a sender->receiver pair that occurs once in the week, against 28% of
background and 36% of relay traffic. Linking one-off payments that share a wallet and follow each
other within dt recovers motif instances: at dt = 10 min, 99% of each easy suspicious scenario
lands in one group at ~90% purity (train, labels used only for that check).

The decisive property is the group's internal pace. Easy suspicious motifs step every ~1 min
(median gap 60 s); the legitimate look-alikes that the per-payment model confuses with them step
every ~5 min, other look-alikes ~11 min, medium suspicious ~18 min but keep more value per hop.
So for several link scales each payment gets its group's size, wallets, span, median/max internal
gap, value spread and value retention through intermediaries, plus the median gap as a percentile
among the split's multi-payment groups (robust if a whole split runs faster or slower).
Non-one-off payments get NaN group features and a flag.
"""
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

SCALES = (60, 120, 300, 600, 1800)


def _components(sender, receiver, ts, dt):
    n = len(ts)
    w = np.concatenate([sender, receiver])
    t = np.concatenate([ts, ts])
    i = np.concatenate([np.arange(n), np.arange(n)])
    o = np.lexsort((t, w))
    w, t, i = w[o], t[o], i[o]
    link = (w[1:] == w[:-1]) & (np.diff(t) <= dt) & (i[1:] != i[:-1])
    a, b = i[:-1][link], i[1:][link]
    g = coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n))
    return connected_components(g, directed=False)[1]


def _retention(q, comp):
    """Median and min of out/in value at intermediaries: a payment received and the next one-off
    payment the receiver makes inside the same group."""
    d = pd.DataFrame({"c": comp, "s": q.sender.to_numpy(), "r": q.receiver.to_numpy(),
                      "t": q.ts.to_numpy(), "v": q.dst_usd.to_numpy()})
    outs = d.rename(columns={"s": "w", "t": "t_out", "v": "v_out"})[["c", "w", "t_out", "v_out"]]
    ins = d.rename(columns={"r": "w", "t": "t_in", "v": "v_in"})[["c", "w", "t_in", "v_in"]]
    m = ins.merge(outs, on=["c", "w"])
    m = m[m.t_out > m.t_in].sort_values("t_out").groupby(["c", "w", "t_in"], sort=False).first()
    m["ret"] = m.v_out / m.v_in
    agg = m.groupby(level="c").ret.agg(["median", "min", "size"])
    return agg


def build(p):
    p = p.reset_index(drop=True)
    pair = p.sender + ">" + p.receiver
    one_off = (pair.map(pair.value_counts()) == 1).to_numpy()
    q = p[one_off].reset_index(drop=True)
    S, R, T, V = q.sender.to_numpy(), q.receiver.to_numpy(), q.ts.to_numpy(), q.dst_usd.to_numpy()

    f = pd.DataFrame(index=p.index)
    f["one_off"] = one_off.astype(float)
    for dt in SCALES:
        tag = f"{dt // 60}m" if dt >= 60 else f"{dt}s"
        comp = _components(S, R, T, dt)
        d = pd.DataFrame({"c": comp, "t": T, "v": V, "lv": np.log(V)})
        d = d.sort_values(["c", "t"])
        d["gap"] = d.groupby("c").t.diff()
        g = d.groupby("c")
        agg = pd.DataFrame({
            "size": g.size(),
            "span_m": (g.t.max() - g.t.min()) / 60,
            "gap_med_s": g.gap.median(),
            "gap_max_s": g.gap.max(),
            "lv_std": g.lv.std(),
        })
        wallets = pd.concat([pd.Series(comp, name="c").to_frame().assign(w=S),
                             pd.Series(comp, name="c").to_frame().assign(w=R)]).drop_duplicates()
        agg["wallets"] = wallets.groupby("c").size()
        ret = _retention(q, comp)
        agg["ret_med"] = ret["median"]
        agg["ret_min"] = ret["min"]
        agg["handoffs"] = ret["size"]
        agg["handoffs"] = agg.handoffs.fillna(0) / agg["size"]
        multi = agg["size"] >= 3
        agg["gap_pct"] = np.nan
        agg.loc[multi, "gap_pct"] = agg.loc[multi, "gap_med_s"].rank(pct=True)
        per_tx = agg.reindex(comp).reset_index(drop=True)
        for col in per_tx.columns:
            v = np.full(len(p), np.nan)
            v[one_off] = per_tx[col].to_numpy()
            f[f"{col}_{tag}"] = v
    return f.astype(float)
