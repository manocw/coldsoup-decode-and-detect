"""Round trips: time-respecting cycles that bring value back to the wallet that sent it.

For every payment u->v at time t we search forward in time from v for a path that returns to
u (each hop strictly later than the previous one, no wallet visited twice) inside a window.
Every edge on a found cycle is credited, so a transaction in the middle of a loop is marked
as well as the one that opened it. Searches are depth-first over each wallet's time-sorted
out-edges with a per-start cap, which keeps a split to a few seconds.

Features (all behavioural; no addresses or absolute times):
- n_<w>: how many cycles of the window pass through the transaction (log1p).
- cyc_*: shape of the fastest cycle (<= 6 h) through the transaction: duration, length,
  value retention last/first hop, mean/sd/min/max per-hop log ratio, share of rising hops,
  share of first-time pairs on the cycle, position of the tx, regularity of the hop gaps.
  cyc_minlen: the shortest cycle (<= 6 h) through the transaction.
- vc_*: same for "value-consistent" cycles (every hop keeps 50-160% of the previous hop's
  value) within 12 h, which removes chance loops through unrelated payments.
- vcx_score: best value-consistent cycle score = mean hop retention minus slowness.
- recip_next / recip_prev: delay to the next/previous payment in the reverse direction v->u.
- pair_count / rev_pair_count: how often the pair (and the reverse pair) occurs in the week.
"""
import bisect

import numpy as np
import pandas as pd

MIN = 60
HOUR = 3600


def _graph(p):
    codes, _ = pd.factorize(pd.concat([p.sender, p.receiver], ignore_index=True))
    n = len(p)
    return codes[:n].astype(np.int64), codes[n:].astype(np.int64), p.ts.to_numpy().astype(np.int64)


def enumerate_cycles(S, R, T, A, W, K, lo=0.0, hi=np.inf, cap_per_start=200):
    """All elementary time-respecting cycles opened by each edge, as tuples of row indices.

    Rows must be time-sorted. W: max seconds from the opening edge to the closing edge.
    K: max cycle length. lo/hi: optional bound on each hop's value relative to the previous.
    """
    n = len(S)
    N = int(max(S.max(), R.max())) + 1
    out = [[] for _ in range(N)]
    for i in range(n):
        out[S[i]].append(i)
    out_t = [[int(T[i]) for i in o] for o in out]
    Tl, Rl, Sl, Al = T.tolist(), R.tolist(), S.tolist(), A.tolist()
    check = lo > 0 or np.isfinite(hi)
    cycles = []
    for e in range(n):
        u, v, t = Sl[e], Rl[e], Tl[e]
        if u == v or (check and not Al[e] > 0):
            continue
        tend = t + W
        stack = [(v, e, (e,), (u, v))]
        found = 0
        while stack and found < cap_per_start:
            node, last, path, nodes = stack.pop()
            lst = out[node]
            ts = out_t[node]
            al = Al[last]
            j = bisect.bisect_left(ts, Tl[last])
            m = len(lst)
            while j < m and ts[j] <= tend:
                f = lst[j]
                j += 1
                if f <= last:
                    continue
                if check and not (lo * al <= Al[f] <= hi * al):
                    continue
                w = Rl[f]
                if w == u:
                    cycles.append(path + (f,))
                    found += 1
                    continue
                if len(path) + 1 >= K or w in nodes:
                    continue
                stack.append((w, f, path + (f,), nodes + (w,)))
    return cycles


def _counts(cycles, n):
    cnt = np.zeros(n)
    for c in cycles:
        for x in c:
            cnt[x] += 1
    return np.log1p(cnt)


def _best_attrs(cycles, T, amt, pair_cnt, n, prefix, f):
    """Attributes of the fastest cycle through each transaction."""
    best = {}
    for c in cycles:
        dur = T[c[-1]] - T[c[0]]
        for k, x in enumerate(c):
            b = best.get(x)
            if b is None or dur < b[0]:
                best[x] = (dur, c, k)
    names = ["dur", "len", "ret", "geo", "up", "sd", "novel", "pos", "gapcv", "hopmin", "hopmax"]
    cols = {k: np.full(n, np.nan) for k in names}
    for x, (dur, c, k) in best.items():
        c = list(c)
        a = amt[c]
        lr = np.diff(np.log(a))
        gaps = np.diff(T[c]).astype(float)
        cols["dur"][x] = np.log1p(dur)
        cols["len"][x] = len(c)
        cols["ret"][x] = np.log(a[-1] / a[0])
        cols["geo"][x] = np.nanmean(lr)
        cols["up"][x] = np.mean(lr > 0)
        cols["sd"][x] = np.nanstd(lr)
        cols["hopmin"][x] = np.nanmin(lr)
        cols["hopmax"][x] = np.nanmax(lr)
        cols["novel"][x] = np.mean(pair_cnt[c] == 1)
        cols["pos"][x] = k / (len(c) - 1)
        cols["gapcv"][x] = gaps.std() / gaps.mean() if gaps.mean() > 0 else 0
    for k, v in cols.items():
        f[f"{prefix}_{k}"] = v


def build(p):
    S, R, T = _graph(p)
    n = len(p)
    amt = p.dst_usd.to_numpy(float)
    amt = np.where(np.isfinite(amt) & (amt > 0), amt, np.nan)
    A = np.nan_to_num(amt, nan=0.0)
    pair = pd.Series(list(zip(S, R)))
    vc = pair.value_counts()
    pair_cnt = pair.map(vc).to_numpy()
    f = pd.DataFrame(index=p.index)

    short = []
    for name, W, K in [("15m", 15 * MIN, 8), ("2h", 2 * HOUR, 8), ("6h", 6 * HOUR, 8),
                       ("12h", 12 * HOUR, 5), ("3d", 72 * HOUR, 3)]:
        cyc = enumerate_cycles(S, R, T, A, W, K)
        f[f"n_{name}"] = _counts(cyc, n)
        if name in ("15m", "2h", "6h"):
            short += cyc
    _best_attrs(short, T, amt, pair_cnt, n, "cyc", f)
    minlen = np.full(n, np.nan)
    for c in short:
        for x in c:
            if not minlen[x] <= len(c):
                minlen[x] = len(c)
    f["cyc_minlen"] = minlen

    # value-consistent cycles: every hop keeps 50%-160% of the previous hop's value
    f["n_vc2h"] = _counts(enumerate_cycles(S, R, T, A, 2 * HOUR, 8, 0.5, 1.6), n)
    cyc = enumerate_cycles(S, R, T, A, 12 * HOUR, 8, 0.5, 1.6)
    f["n_vc12h"] = _counts(cyc, n)
    _best_attrs(cyc, T, amt, pair_cnt, n, "vc", f)
    score = np.full(n, np.nan)
    for c in cyc:
        c = list(c)
        sc = np.mean(np.diff(np.log(amt[c]))) - 0.1 * np.log1p((T[c[-1]] - T[c[0]]) / 60)
        for x in c:
            if not score[x] >= sc:
                score[x] = sc
    f["vcx_score"] = score

    # reciprocal payments v->u around this one
    key = {}
    for i in range(n):
        key.setdefault((S[i], R[i]), []).append(i)
    nxt = np.full(n, np.nan)
    prv = np.full(n, np.nan)
    for i in range(n):
        lst = key.get((R[i], S[i]))
        if not lst:
            continue
        ts = T[lst]
        j = np.searchsorted(ts, T[i], "right")
        if j < len(ts):
            nxt[i] = np.log1p(ts[j] - T[i])
        j2 = np.searchsorted(ts, T[i], "left")
        if j2 > 0:
            prv[i] = np.log1p(T[i] - ts[j2 - 1])
    f["recip_next"] = nxt
    f["recip_prev"] = prv
    f["pair_count"] = pair_cnt
    f["rev_pair_count"] = pd.Series(list(zip(R, S))).map(vc).fillna(0).to_numpy()
    return f.astype(float)
