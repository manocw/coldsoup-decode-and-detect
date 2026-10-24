"""Medium-tier signatures found by the pattern hunters (notes/hunt_fans.md, notes/hunt_hops.md).

1. Hub-burst amount roughness. Payments on rarely used pairs (pair used <= 2 times in the week) are
   streamed per hub: (sender, source asset) and (receiver, destination asset); a burst ends at a gap
   > 60 min. Legitimate look-alike fans have SMOOTH amount sequences (fan_out nearly sorted ascending,
   fan_in / bipartite positively autocorrelated); medium suspicious ones come in random order. The von
   Neumann ratio VN = sum(diff(x)^2) / sum((x - mean)^2) on x = log USD in time order is ~2 for random
   order and well below 2 for smooth sequences (train: medium vs look-alike payment AUC 0.82 in fan
   families, 0.86 in multi_stage; identical distribution on train / validation / test).
2. Hub-burst gap regularity: medium hubs pay at near-constant intervals (+-20%), look-alike gaps look
   Poisson; reg = mean |diff log(gap_min + 0.25)| (low = regular), with a leave-one-out robust version.
3. One-off butterflies (stack width): for a one-off payment u->v, the number of (u2, v2) with u->v2,
   u2->v and u2->v2 all one-off payments within +-W. Suspicious stacks have 2+ middle wallets, look-alikes
   are mostly 2->1->2 (stack medium payment AUC 0.72-0.80 at W = 6-12 h).
"""
from collections import defaultdict

import numpy as np
import pandas as pd

GAP = 3600
MAX_PAIR = 2
K = ["vn", "vn_rob", "lag1", "reg", "reg_rob"]


def _vn(x):
    v = ((x - x.mean()) ** 2).sum()
    return (np.diff(x) ** 2).sum() / v if v > 0 else np.nan


def _gapreg(tt):
    g = np.diff(tt) / 60
    return np.abs(np.diff(np.log(g + 0.25))).mean() if len(g) >= 2 else np.nan


def _burst_stats(df):
    df = df.sort_values("ts")
    tt, la = df.ts.to_numpy(float), df.la.to_numpy()
    n = len(la)
    r = dict(vn=_vn(la), lag1=np.corrcoef(la[:-1], la[1:])[0, 1] if n >= 4 else np.nan, reg=_gapreg(tt))
    if n >= 5:
        r["vn_rob"] = np.nanmax([_vn(np.delete(la, i)) for i in range(n)])
        r["reg_rob"] = np.nanmin([_gapreg(np.delete(tt, i)) for i in range(n)])
    else:
        r["vn_rob"], r["reg_rob"] = r["vn"], r["reg"]
    return pd.Series(r)


def _bursts(p):
    q0 = p[["hash", "ts", "sender", "receiver", "src_asset", "dst_asset"]].copy()
    q0["la"] = np.log(p.dst_usd.clip(lower=1e-3))
    pair_n = q0.groupby(["sender", "receiver"]).ts.transform("size")
    F = pd.DataFrame(index=p.hash)
    for role, asset, pre in (("sender", "src_asset", "s_"), ("receiver", "dst_asset", "r_")):
        q = q0[pair_n <= MAX_PAIR].sort_values([role, asset, "ts"]).copy()
        key = (q[role] + "|" + q[asset]).to_numpy()
        t = q.ts.to_numpy(float)
        same = np.r_[False, key[1:] == key[:-1]]
        dt = np.r_[np.inf, np.diff(t)]
        q["bid"] = np.cumsum(~(same & (dt <= GAP)))
        q["bn"] = q.groupby("bid").ts.transform("size")
        big = q[q.bn >= 3]
        B = big.groupby("bid")[["ts", "la"]].apply(_burst_stats) if len(big) else pd.DataFrame(columns=K)
        q = q.join(B, on="bid")
        F = F.join(q.set_index("hash")[["bn"] + K].add_prefix(pre))
    hub_s = F.s_bn.fillna(0) >= F.r_bn.fillna(0)
    out = pd.DataFrame(index=F.index)
    for k in ["bn"] + K:
        out["h_" + k] = np.where(hub_s, F["s_" + k], F["r_" + k])
    return out


def _butterflies(p, W):
    S, R, t = p.sender.to_numpy(), p.receiver.to_numpy(), p.ts.to_numpy()
    pair = pd.Series(S) + ">" + pd.Series(R)
    novel = (pair.map(pair.value_counts()) == 1).to_numpy()
    idx = np.flatnonzero(novel)
    outs, ins = defaultdict(list), defaultdict(list)
    for i in idx:
        outs[S[i]].append(i)
        ins[R[i]].append(i)
    bf = np.full(len(p), np.nan)
    for i in idx:
        u, v, ti = S[i], R[i], t[i]
        vs = {R[j] for j in outs[u] if j != i and abs(t[j] - ti) <= W}
        if not vs:
            bf[i] = 0
            continue
        us = {S[j] for j in ins[v] if j != i and abs(t[j] - ti) <= W}
        bf[i] = sum(len({R[j] for j in outs[u2] if abs(t[j] - ti) <= W} & vs) for u2 in us)
    return bf


def build(p):
    p = p.reset_index(drop=True)
    f = _bursts(p).reindex(p.hash).reset_index(drop=True)
    f["bf_6h"] = _butterflies(p, 6 * 3600)
    f["bf_12h"] = _butterflies(p, 12 * 3600)
    return f.astype(float)
