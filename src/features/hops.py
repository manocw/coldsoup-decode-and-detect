"""Fast hops / money trail tracker.

A "hop" links payment i to a later payment j sent by i's receiver (S[j] == R[i],
t_i < t_j <= t_i + window). A hop "matches" when j moves on roughly the value i delivered
(out/in ratio in [0.6, 1.25]). From the hop graph every transaction gets:

  local   - delay and value retention (log out/in) of its fastest matched hop forward /
            backward (within 6h)
  chain   - longest matched chain through it within 10m, 1h, 6h
  trail   - walking the longest 1h / 6h chain through it: median / max hop delay, median /
            min hop retention, cumulative skim, wallets touched, revisits (cycles), in-degree
            of the wallet the trail ends at (cash-out hub), out-degree of the trail origin
            (degrees relative to the split's median wallet)
  comp    - connected component of the hop graph (all hops within 10m; matched hops within 1h):
            size, wallets, span, median hop delay / retention, end-to-end retention
            (value leaving the component's sinks over value entering from its sources)
  par     - stack-like layering: distinct intermediaries m between the same start a and end z
            (a -> m -> z, second hop within 1h / 6h), distinct 2-hop ends / starts, and for the
            widest such (a, z) group: value retention (sum out of the m's / sum into them) and
            median hop delay
  flow    - money the receiver passes on within 1h / 6h after, money the sender got within
            1h / 6h before, receiver pass-through ratio around the payment

Everything is behavioural: wallets are integer codes local to the split, times are relative.
Lookups use sorted keys + searchsorted; the chain DP is one pass over the edge list.
"""
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

MIN, HOUR = 60, 3600
WMAX = 6 * HOUR
LR_LO, LR_HI = np.log(0.6), np.log(1.25)      # matched hop: out/in value ratio band
CHAIN_WINDOWS = {"10m": 10 * MIN, "1h": HOUR, "6h": 6 * HOUR}
TRAIL_WINDOWS = ("1h", "6h")
PAR_WINDOWS = {"1h": HOUR, "6h": 6 * HOUR}
FLOW_WINDOWS = {"1h": HOUR, "6h": 6 * HOUR}
MAX_WALK = 40


def _keyed(W, T, base, big):
    return W.astype(np.int64) * big + (T - base)


def _edges(t, S, R, wmax):
    """All hops (i, j): S[j] == R[i] and t[i] < t[j] <= t[i] + wmax. Vectorised."""
    base, big = t.min(), np.int64(t.max() - t.min() + wmax + 10)
    order = np.lexsort((t, S))
    k = _keyed(S[order], t[order], base, big)
    lo = np.searchsorted(k, _keyed(R, t, base, big), "right")
    hi = np.searchsorted(k, _keyed(R, t + wmax, base, big), "right")
    cnt = hi - lo
    src = np.repeat(np.arange(len(t)), cnt)
    off = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    return src, order[np.repeat(lo, cnt) + off]


def _window_sums(W, T, V):
    """Per-wallet range sums: q(wallets, lo, hi, sides) -> sum of V with T in the range."""
    base, big = T.min(), np.int64(T.max() - T.min() + 30 * 86400)
    o = np.lexsort((T, W))
    k = _keyed(W[o], T[o], base, big)
    cs = np.concatenate([[0.0], np.cumsum(V[o])])

    def q(wq, lo, hi, side_lo, side_hi):
        a = np.searchsorted(k, _keyed(wq, lo, base, big), side_lo)
        b = np.searchsorted(k, _keyed(wq, hi, base, big), side_hi)
        return cs[b] - cs[a]
    return q


def _dp(n, src, dst, score, forward):
    """Longest matched path from each node (forward: via successors, else predecessors).

    t[src] < t[dst] and rows are time-sorted, so processing edges by src descending
    (forward) / dst ascending (backward) finalises a node before it is read. Ties on
    depth go to the higher edge score. Returns depth and the chosen neighbour (-1)."""
    if forward:
        idx = np.lexsort((-dst, -src))
        a, b = src[idx], dst[idx]
    else:
        idx = np.lexsort((src, dst))
        a, b = dst[idx], src[idx]
    dl = [0] * n
    bl = [-1] * n
    bs = [-np.inf] * n
    for u, v, s in zip(a.tolist(), b.tolist(), score[idx].tolist()):
        d = dl[v] + 1
        if d > dl[u] or (d == dl[u] and s > bs[u]):
            dl[u] = d
            bl[u] = v
            bs[u] = s
    return np.array(dl, float), np.array(bl)


def _trail(n, t, S, R, vin, vout, nf, nb, in_deg, out_deg, w):
    """Walk the chosen backward and forward pointers through every tx and describe the trail."""
    cols = {k: np.full(n, np.nan) for k in
            ("med_delay_min", "max_delay_min", "med_lr", "min_lr", "cum_lr")}
    wallets = np.full(n, 2.0)
    revisits = np.zeros(n)
    end_in = in_deg[R].astype(float)
    start_out = out_deg[S].astype(float)
    nfl, nbl, tl, Sl, Rl = nf.tolist(), nb.tolist(), t.tolist(), S.tolist(), R.tolist()
    lvin, lvout = np.log(vin).tolist(), np.log(vout).tolist()
    for i in np.flatnonzero((nf >= 0) | (nb >= 0)).tolist():
        chain = [i]
        k = i
        while nbl[k] >= 0 and len(chain) < MAX_WALK:
            k = nbl[k]
            chain.append(k)
        chain.reverse()
        k = i
        steps = 0
        while nfl[k] >= 0 and steps < MAX_WALK:
            k = nfl[k]
            chain.append(k)
            steps += 1
        dts = np.diff([tl[c] for c in chain])
        lrs = [lvout[chain[q + 1]] - lvin[chain[q]] for q in range(len(chain) - 1)]
        cols["med_delay_min"][i] = np.median(dts) / MIN
        cols["max_delay_min"][i] = dts.max() / MIN
        cols["med_lr"][i] = np.median(lrs)
        cols["min_lr"][i] = min(lrs)
        cols["cum_lr"][i] = lvin[chain[-1]] - lvout[chain[0]]
        ws = [Sl[c] for c in chain] + [Rl[chain[-1]]]
        wallets[i] = len(set(ws))
        revisits[i] = len(ws) - len(set(ws))
        end_in[i] = in_deg[Rl[chain[-1]]]
        start_out[i] = out_deg[Sl[chain[0]]]
    out = {f"trail_{k}_{w}": v for k, v in cols.items()}
    out.update({f"trail_wallets_{w}": wallets, f"trail_revisits_{w}": revisits,
                f"trail_end_indeg_{w}": end_in, f"trail_start_outdeg_{w}": start_out})
    return out


def _components(n, t, S, R, vin, vout, a, b, dt, lr, tag):
    """Connected components of the hop graph restricted to edges (a, b): flow-unit stats."""
    g = coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n))
    nc, lab = connected_components(g, directed=False)
    gb = pd.Series(t).groupby(lab)
    size = np.bincount(lab)[lab].astype(float)
    span = ((gb.max() - gb.min()) / MIN).to_numpy()[lab]
    wal = (pd.concat([pd.Series(S), pd.Series(R)]).groupby(np.concatenate([lab, lab]))
           .nunique().to_numpy()[lab].astype(float))
    e = pd.DataFrame({"c": lab[a], "dt": dt / MIN, "lr": lr}).groupby("c")
    med_lr = e.lr.median().reindex(range(nc)).to_numpy()[lab]
    med_dt = e.dt.median().reindex(range(nc)).to_numpy()[lab]
    # sources: payments whose sender receives nothing inside the component; sinks: the reverse
    recv_in = pd.MultiIndex.from_arrays([lab, R])
    send_in = pd.MultiIndex.from_arrays([lab, S])
    first = ~send_in.isin(recv_in)
    last = ~recv_in.isin(send_in)
    fi = np.bincount(lab, weights=np.where(first, vout, 0.0), minlength=nc)
    la = np.bincount(lab, weights=np.where(last, vin, 0.0), minlength=nc)
    with np.errstate(divide="ignore", invalid="ignore"):
        e2e = np.log(la / fi)[lab]
    e2e[~np.isfinite(e2e)] = np.nan
    single = size == 1
    for arr in (med_lr, med_dt, e2e):
        arr[single] = np.nan
    return {f"comp_size_{tag}": size, f"comp_span_min_{tag}": span, f"comp_wallets_{tag}": wal,
            f"comp_med_lr_{tag}": med_lr, f"comp_med_delay_min_{tag}": med_dt,
            f"comp_e2e_lr_{tag}": e2e}


def _parallel(n, S, R, vin, vout, src, dst, dt, tag):
    """Stack-like layering over 2-hop paths a -> m -> z (z != a) within the window."""
    a_, m_, z_ = S[src], R[src], R[dst]
    keep = a_ != z_
    e = pd.DataFrame({"i": src[keep], "j": dst[keep], "a": a_[keep], "m": m_[keep],
                      "z": z_[keep], "dt": dt[keep] / MIN})
    g = e.groupby(["a", "z"])
    grp = pd.DataFrame({"par": g.m.nunique(), "dtmed": g.dt.median()})
    ii = e.drop_duplicates(["a", "z", "i"])
    jj = e.drop_duplicates(["a", "z", "j"])
    vi = pd.Series(vout[ii.i.to_numpy()], index=pd.MultiIndex.from_frame(ii[["a", "z"]]))
    vj = pd.Series(vin[jj.j.to_numpy()], index=pd.MultiIndex.from_frame(jj[["a", "z"]]))
    grp["vin"] = vi.groupby(level=[0, 1]).sum()
    grp["lr"] = np.log(vj.groupby(level=[0, 1]).sum() / grp.vin)
    e = e.join(grp, on=["a", "z"])
    out = {}
    pi = e.groupby("i").par.max().reindex(range(n)).fillna(0).to_numpy()
    pj = e.groupby("j").par.max().reindex(range(n)).fillna(0).to_numpy()
    out[f"par_{tag}"] = np.maximum(pi, pj)
    out[f"par_ends_{tag}"] = e.groupby("i").z.nunique().reindex(range(n)).fillna(0).to_numpy()
    out[f"par_starts_{tag}"] = e.groupby("j").a.nunique().reindex(range(n)).fillna(0).to_numpy()
    # retention / delay of the widest (a, z) group the tx belongs to (as first or second hop)
    w = e[e.par >= 2]
    both = pd.concat([w[["i", "par", "vin", "lr", "dtmed"]].rename(columns={"i": "k"}),
                      w[["j", "par", "vin", "lr", "dtmed"]].rename(columns={"j": "k"})])
    best = (both.sort_values(["k", "par", "vin"], ascending=[True, False, False])
            .drop_duplicates("k").set_index("k"))
    out[f"par_lr_{tag}"] = best.lr.reindex(range(n)).to_numpy()
    out[f"par_delay_min_{tag}"] = best.dtmed.reindex(range(n)).to_numpy()
    return out


def build(p):
    n = len(p)
    t = p.ts.to_numpy(np.int64)
    codes, uniq = pd.factorize(pd.concat([p.sender, p.receiver], ignore_index=True))
    S, R = codes[:n], codes[n:]
    nw = len(uniq)
    vin = p.dst_usd.to_numpy(float)                       # what the receiver got
    vout = p.src_usd.to_numpy(float)                      # what the sender actually paid
    vin = np.where(np.isfinite(vin) & (vin > 0), vin, np.nanmedian(vin))
    vout = np.where(np.isfinite(vout) & (vout > 0), vout, vin)

    src, dst = _edges(t, S, R, WMAX)
    dt = (t[dst] - t[src]).astype(float)
    lr = np.log(vout[dst] / vin[src])
    match = (lr >= LR_LO) & (lr <= LR_HI)
    # distinct counterparties, relative to the split's median wallet (split-size free)
    in_deg = pd.Series(S).groupby(R).nunique().reindex(range(nw)).fillna(0).to_numpy()
    out_deg = pd.Series(R).groupby(S).nunique().reindex(range(nw)).fillna(0).to_numpy()
    in_deg = in_deg / np.median(in_deg[in_deg > 0])
    out_deg = out_deg / np.median(out_deg[out_deg > 0])

    f = {}
    # ---- local: fastest matched hop forward / backward ----------------------------------
    ms, md, mdt, mlr = src[match], dst[match], dt[match], lr[match]
    for name, a in (("fwd", ms), ("bwd", md)):
        o = np.lexsort((mdt, a))
        first = np.ones(len(o), bool)
        first[1:] = a[o][1:] != a[o][:-1]
        sel = o[first]
        delay = np.full(n, np.nan)
        ratio = np.full(n, np.nan)
        delay[a[sel]] = mdt[sel] / MIN
        ratio[a[sel]] = mlr[sel]
        f[f"{name}_delay_min"] = delay
        f[f"{name}_lr"] = ratio

    # ---- chain length and trails ---------------------------------------------------------
    hop_score = -np.abs(lr) - dt / WMAX                     # prefer close value, then fast
    for w, sec in CHAIN_WINDOWS.items():
        keep = match & (dt <= sec)
        a, b, s = src[keep], dst[keep], hop_score[keep]
        df_, nf = _dp(n, a, b, s, True)
        db_, nb = _dp(n, a, b, s, False)
        f[f"chain_len_{w}"] = df_ + db_ + 1
        if w in TRAIL_WINDOWS:
            f.update(_trail(n, t, S, R, vin, vout, nf, nb, in_deg, out_deg, w))

    # ---- hop-graph components ------------------------------------------------------------
    for tag, keep in (("10m", dt <= 10 * MIN), ("1hm", match & (dt <= HOUR))):
        f.update(_components(n, t, S, R, vin, vout, src[keep], dst[keep], dt[keep], lr[keep], tag))

    # ---- stack-like layering -------------------------------------------------------------
    for tag, sec in PAR_WINDOWS.items():
        keep = dt <= sec
        f.update(_parallel(n, S, R, vin, vout, src[keep], dst[keep], dt[keep], tag))

    # ---- aggregate flow around the payment -------------------------------------------------
    sends = _window_sums(S, t, vout)
    gets = _window_sums(R, t, vin)
    for tag, sec in FLOW_WINDOWS.items():
        r_out = sends(R, t, t + sec, "right", "right")        # receiver passes on after
        s_in = gets(S, t - sec, t, "left", "left")            # sender was funded before
        r_in = gets(R, t - sec, t, "left", "right")           # receiver's inflow incl. this
        # log out/in ratios only where there is flow; "no flow" is its own indicator, so a tiny
        # payment with nothing passed on cannot look like a full pass-through
        with np.errstate(divide="ignore", invalid="ignore"):
            f[f"r_out_after_{tag}"] = np.where(r_out > 0, np.log(r_out / vin), np.nan)
            f[f"s_in_before_{tag}"] = np.where(s_in > 0, np.log(s_in / vout), np.nan)
        f[f"r_has_out_after_{tag}"] = (r_out > 0).astype(float)
        f[f"s_has_in_before_{tag}"] = (s_in > 0).astype(float)
        f[f"r_passthru_{tag}"] = np.log1p(r_out) - np.log1p(r_in)

    return pd.DataFrame(f, index=p.index).astype(float)
