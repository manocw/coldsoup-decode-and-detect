"""Relay structure: hub wallets that collect then pass on value in repeated rounds.

The aggregate/distributed relay look-alikes are the largest source of false positives: a hub
receives from a few wallets, and a few hours later pays a few others, in three or four rounds
spread over the week. This family reconstructs that structure from behaviour only:

1. Edge recurrence: relay counterparty pairs repeat in 2-4 separate rounds (tens of hours apart),
   unlike one-shot motif edges and unlike high-frequency commercial relationships.
2. Hub bursts: a wallet's receipts of one asset (and its payments out of that same asset) are
   grouped into bursts (consecutive events < 1.2 h apart). A hub's in-burst is linked to an
   out-burst of the same wallet-asset when the idle gap between them matches a relay hop
   (3.35-6.1 h single hub, 1.6-3.25 h per hop of a two-hub chain). "Forward" = receive then pay
   on; "refill" = pay out then get topped up. The log ratio of the two burst values measures how
   much of the value is passed on.
3. Reconstructed relays: bursts are joined across rounds when they reuse a recurring
   counterparty pair, when an in-burst and out-burst are each other's only link, or when a lone
   recurring payment is the only chain-timed link at both of its ends (hub -> hub). Per joined
   component we summarise direction (share of forward vs refill links) and value conservation
   (share of forward hops passing on the value within +-0.55 log, median |log ratio|).

Measured on train (scenario-level, labels used only for analysis): all 12 planted relays are
forward and ~95% of their hops pass on the value within |log ratio| < 0.55; legit relays are
half refills (594 of 1188 forward), and in legit forward single-hub relays only ~36% of hops do.
Each transaction gets the summary of the component on the receiver side (it is a receipt there)
and on the sender side (it is a payment there), plus a "hub side" pick of the two.
No addresses, labels, or absolute times are used; only gaps between this split's own timestamps.
"""
import numpy as np
import pandas as pd

H = 3600.0
BURST_GAP = 1.2                                # h, max gap inside one in- or out-burst
LAGS = {"a": (3.35, 6.1), "d": (1.6, 3.25)}    # h, idle gap from one burst to the burst it feeds
RARE = (2, 4)                                  # transactions per recurring relay edge
TOL = 0.55                                     # |log(out/in)| counted as "value passed on"
SESSION_GAP = 6.5                              # h, idle gap that closes a wallet-asset session


def _edges(p):
    e = pd.Series(p.sender.to_numpy() + ">" + p.receiver.to_numpy())
    return e, e.map(e.value_counts()).to_numpy()


def _bursts(key, t):
    """Group events by key into time bursts; returns the burst id of each event (input order)."""
    n = len(key)
    o = np.lexsort((t, key))
    ks, ts = key[o], t[o]
    new = np.r_[True, ks[1:] != ks[:-1]] | (np.r_[np.inf, np.diff(ts)] > BURST_GAP)
    b = np.empty(n, dtype=np.int64)
    b[o] = np.cumsum(new) - 1
    return b


def _reconstruct(p, t, v):
    kin = p.receiver.to_numpy() + "#" + p.dst_asset.to_numpy()     # receiver holding what it was paid
    kout = p.sender.to_numpy() + "#" + p.src_asset.to_numpy()      # sender spending from that holding
    bi, bo = _bursts(kin, t), _bursts(kout, t)
    Bi = pd.DataFrame({"b": bi, "k": kin, "t": t, "v": v}).groupby("b").agg(
        k=("k", "first"), t0=("t", "min"), t1=("t", "max"), n=("t", "size"), s=("v", "sum"))
    Bo = pd.DataFrame({"b": bo, "k": kout, "t": t, "v": v}).groupby("b").agg(
        k=("k", "first"), t0=("t", "min"), t1=("t", "max"), n=("t", "size"), s=("v", "sum"))
    L = Bi.reset_index().rename(columns={"b": "bi"}).merge(
        Bo.reset_index().rename(columns={"b": "bo"}), on="k", suffixes=("_i", "_o"))
    L["lag_f"] = L.t0_o - L.t1_i
    L["lag_r"] = L.t0_i - L.t1_o
    parts = []
    for typ, (lo, hi) in LAGS.items():
        for d in ("f", "r"):
            m = L[(L[f"lag_{d}"] >= lo) & (L[f"lag_{d}"] <= hi)]
            parts.append(pd.DataFrame({"bi": m.bi.to_numpy(), "bo": m.bo.to_numpy(), "typ": typ, "dirn": d,
                                       "alr": np.abs(np.log((m.s_o.to_numpy() + 1) / (m.s_i.to_numpy() + 1)))}))
    return bi, Bi, bo, Bo, pd.concat(parts, ignore_index=True)


def _components(n_nodes, a, b):
    par = np.arange(n_nodes)

    def find(x):
        r = x
        while par[r] != r:
            r = par[r]
        while par[x] != r:
            par[x], x = r, par[x]
        return r

    for x, y in zip(a, b):
        rx, ry = find(x), find(y)
        if rx != ry:
            par[rx] = ry
    return np.fromiter((find(x) for x in range(n_nodes)), dtype=np.int64, count=n_nodes)


def _relay_components(bi, bo, nbi, nbo, L, e, en):
    """Join bursts across rounds: shared recurring counterparty pair, or mutual unique hop link."""
    A, B = [], []
    d = pd.DataFrame({"bi": bi, "bo": bo + nbi, "e": e.to_numpy()})[(en >= RARE[0]) & (en <= RARE[1])]
    for col in ("bi", "bo"):
        for bs in d.groupby("e")[col].unique():
            if len(bs) > 1:
                A += [bs[0]] * (len(bs) - 1)
                B += list(bs[1:])
    ci, co = L.groupby("bi").size(), L.groupby("bo").size()
    m = L[(L.bi.map(ci) == 1) & (L.bo.map(co) == 1)]
    A += list(m.bi)
    B += list(m.bo + nbi)
    # chain hop (hub -> hub): a lone recurring payment that is the only chain-timed link of both its
    # sender's out-burst and its receiver's in-burst, in the same direction
    Ld = L[L.typ == "d"]
    one = lambda s: s.iloc[0] if len(s) == 1 else "x"
    xi = pd.Series(bi).map(Ld.groupby("bi").dirn.agg(one))
    xo = pd.Series(bo).map(Ld.groupby("bo").dirn.agg(one))
    lone = (pd.Series(bi).map(pd.Series(bi).value_counts()) == 1) & (pd.Series(bo).map(pd.Series(bo).value_counts()) == 1)
    ok = (xi.notna() & (xi == xo) & (xi != "x") & lone).to_numpy() & (en >= RARE[0]) & (en <= RARE[1])
    A += list(bi[ok])
    B += list(bo[ok] + nbi)
    return _components(nbi + nbo, A, B)


def _burst_table(L, nbi, nbo):
    T = pd.DataFrame(index=np.arange(nbi + nbo))
    for typ in ("a", "d"):
        for dirn in ("f", "r"):
            T[f"h{dirn}{typ}"] = 0.0
            T[f"a{dirn}{typ}"] = np.nan
            m = L[(L.typ == typ) & (L.dirn == dirn)]
            for col, off in (("bi", 0), ("bo", nbi)):
                best = m.groupby(col).alr.min()
                idx = best.index.to_numpy() + off
                T.loc[idx, f"h{dirn}{typ}"] = 1.0
                T.loc[idx, f"a{dirn}{typ}"] = best.to_numpy()
    T["is_in"] = np.r_[np.ones(nbi), np.zeros(nbo)]
    return T


def _component_summary(T, comp):
    T = T.assign(comp=comp)
    g = T.groupby("comp")
    C = pd.DataFrame({"nb": g.size(), "nin": g.is_in.sum()})
    C["nout"] = C.nb - C.nin
    for typ in ("a", "d"):
        C[f"fs_{typ}"] = g[f"hf{typ}"].mean()
        C[f"rs_{typ}"] = g[f"hr{typ}"].mean()
        C[f"amed_{typ}"] = g[f"af{typ}"].median()
        C[f"amax_{typ}"] = g[f"af{typ}"].max()
        ok = (T[f"af{typ}"] < TOL).astype(float).where(T[f"af{typ}"].notna())
        C[f"aok_{typ}"] = ok.groupby(T.comp).mean()
    use_d = (C.fs_d + C.rs_d) > (C.fs_a + C.rs_a)       # chain-hop timing dominates this component
    out = pd.DataFrame({"nb": C.nb, "nin": C.nin, "nout": C.nout, "is_chain": use_d.astype(float)}, index=C.index)
    for k in ("fs", "rs", "amed", "amax", "aok"):
        out[k] = np.where(use_d, C[f"{k}_d"], C[f"{k}_a"])
    out["dir"] = out.fs - out.rs
    return out


def _window_counts(key_q, t, book, lo, hi):
    """count of events in book[key] within [t+lo, t+hi] for each query."""
    n = len(key_q)
    c = np.zeros(n)
    for k, q in pd.Series(np.arange(n)).groupby(key_q).groups.items():
        tt = book.get(k)
        if tt is None:
            continue
        q = np.asarray(q)
        c[q] = np.searchsorted(tt, t[q] + hi, "right") - np.searchsorted(tt, t[q] + lo, "left")
    return c


def _sessions(p, t, v, kin, kout):
    """Wallet-asset sessions: receipts of asset A and payments out of A, split at idle gaps > SESSION_GAP.

    Returns per-transaction session summaries for the receiver side (tx is a receipt) and the
    sender side (tx is a payment), plus wallet-asset level summaries over all its two-sided sessions.
    """
    n = len(p)
    ev = pd.DataFrame({"k": np.r_[kin, kout], "t": np.r_[t, t], "v": np.r_[v, v],
                       "out": np.r_[np.zeros(n), np.ones(n)], "row": np.r_[np.arange(n), np.arange(n)]})
    ev = ev.sort_values(["k", "t"], kind="mergesort").reset_index(drop=True)
    k = ev.k.to_numpy()
    new = np.r_[True, k[1:] != k[:-1]] | (np.r_[np.inf, np.diff(ev.t.to_numpy())] > SESSION_GAP)
    ev["sid"] = np.cumsum(new) - 1
    ev["vin"] = ev.v * (1 - ev.out)
    ev["vout"] = ev.v * ev.out
    ev["tin"] = ev.t.where(ev.out == 0)
    ev["tout"] = ev.t.where(ev.out == 1)
    g = ev.groupby("sid")
    S = pd.DataFrame({"k": g.k.first(), "n_out": g.out.sum(), "n": g.out.size(), "sum_in": g.vin.sum(), "sum_out": g.vout.sum(),
                      "t_in": g.tin.mean(), "t_out": g.tout.mean(), "last_in": g.tin.max(), "first_in": g.tin.min(),
                      "last_out": g.tout.max(), "first_out": g.tout.min(), "t0": g.t.min(), "t1": g.t.max()})
    S["n_in"] = S.n - S.n_out
    both = (S.n_in > 0) & (S.n_out > 0)
    S["order"] = (S.t_out - S.t_in).where(both)                         # > 0: receive first, pay on later
    S["lr"] = np.log((S.sum_out + 1) / (S.sum_in + 1)).where(both)
    S["cfwd"] = (S.last_in < S.first_out).astype(float).where(both)
    S["crev"] = (S.last_out < S.first_in).astype(float).where(both)
    S["dur"] = S.t1 - S.t0
    Sb = S[both]
    gk = Sb.groupby("k")
    K = pd.DataFrame({"k_sess": gk.size(), "k_fwd_frac": gk.cfwd.mean(), "k_rev_frac": gk.crev.mean(),
                      "k_order_mean": np.sign(Sb.order).groupby(Sb.k).mean(),
                      "k_abslr_max": Sb.lr.abs().groupby(Sb.k).max(), "k_abslr_mean": Sb.lr.abs().groupby(Sb.k).mean()})
    res = {}
    for side, tag in ((0, "r"), (1, "s")):
        e = ev[ev.out == side].sort_values("row")
        s = S.loc[e.sid.to_numpy()].reset_index(drop=True)
        d = pd.DataFrame({"n_in": s.n_in, "n_out": s.n_out, "order": s.order, "lr": s.lr,
                          "cfwd": s.cfwd, "crev": s.crev, "dur": s.dur})
        kk = K.reindex(s.k.to_numpy()).reset_index(drop=True)
        res[tag] = pd.concat([d, kk], axis=1)
    return res


def build(p):
    t = p.ts.to_numpy().astype(float) / H
    v = np.nan_to_num(p.dst_usd.to_numpy().astype(float))
    e, en = _edges(p)
    f = pd.DataFrame(index=p.index)

    # 1. edge recurrence across rounds
    f["edge_n"] = en
    g = pd.DataFrame({"e": e.to_numpy(), "t": t}).groupby("e").t
    prev, nxt = t - g.shift(1).to_numpy(), g.shift(-1).to_numpy() - t
    f["edge_mingap_h"] = np.fmin(prev, nxt)
    f["edge_span_h"] = (g.transform("max") - g.transform("min")).to_numpy()

    # 2. asset-matched hop timing around this payment (receiver as hub / sender as hub)
    kin = p.receiver.to_numpy() + "#" + p.dst_asset.to_numpy()
    kout = p.sender.to_numpy() + "#" + p.src_asset.to_numpy()
    out_book = {k: np.sort(t[np.asarray(ii)]) for k, ii in pd.Series(np.arange(len(p))).groupby(kout).groups.items()}
    in_book = {k: np.sort(t[np.asarray(ii)]) for k, ii in pd.Series(np.arange(len(p))).groupby(kin).groups.items()}
    for typ, (lo, hi) in LAGS.items():
        f[f"r_pass_on_{typ}"] = _window_counts(kin, t, out_book, lo, hi + BURST_GAP)
        f[f"r_paid_before_{typ}"] = _window_counts(kin, t, out_book, -hi - BURST_GAP, -lo)
        f[f"s_funded_before_{typ}"] = _window_counts(kout, t, in_book, -hi - BURST_GAP, -lo)
        f[f"s_refill_after_{typ}"] = _window_counts(kout, t, in_book, lo, hi + BURST_GAP)
    for c in [c for c in f.columns if c.startswith(("r_pass_on", "r_paid_before", "s_funded_before", "s_refill_after"))]:
        f[f"edge_{c}"] = f[c].groupby(e.to_numpy()).transform("mean").to_numpy()

    # 3. burst links and reconstructed relay components
    bi, Bi, bo, Bo, L = _reconstruct(p, t, v)
    nbi, nbo = len(Bi), len(Bo)
    comp = _relay_components(bi, bo, nbi, nbo, L, e, en)
    T = _burst_table(L, nbi, nbo)
    C = _component_summary(T, comp)
    sides = {}
    for side, nodes, b, B in (("r", bi, bi, Bi), ("s", bo + nbi, bo, Bo)):
        tb = T.loc[nodes]
        f[f"{side}_burst_n"] = B.n.reindex(b).to_numpy()
        f[f"{side}_hop_fwd"] = np.maximum(tb.hfa, tb.hfd).to_numpy()
        f[f"{side}_hop_refill"] = np.maximum(tb.hra, tb.hrd).to_numpy()
        f[f"{side}_hop_alr"] = np.fmin(tb.afa, tb.afd).to_numpy()
        for c in ("hop_fwd", "hop_refill", "hop_alr"):    # same counterparty pair across its rounds
            f[f"edge_{side}_{c}"] = f[f"{side}_{c}"].groupby(e.to_numpy()).transform("mean").to_numpy()
        cc = C.loc[comp[nodes]].reset_index(drop=True)
        for c in C.columns:
            f[f"{side}_relay_{c}"] = cc[c].to_numpy()
        sides[side] = cc

    # 4. hub side: the side whose component shows more hop structure
    hub_r = ((sides["r"].fs + sides["r"].rs) * np.log1p(sides["r"].nb)).to_numpy()
    hub_s = ((sides["s"].fs + sides["s"].rs) * np.log1p(sides["s"].nb)).to_numpy()
    use_r = hub_r >= hub_s
    for c in ("nb", "fs", "rs", "dir", "aok", "amed", "amax"):
        f[f"hub_{c}"] = np.where(use_r, sides["r"][c], sides["s"][c])
    f["hub_strength"] = np.maximum(hub_r, hub_s)
    f["hub_pass_on_score"] = (f.hub_dir.fillna(0) + f.hub_aok.fillna(0) - f.hub_amed.fillna(3).clip(upper=3) / 3)

    # 5. wallet-asset sessions (coarser view of the same receive / pay-on cycle)
    for side, d in _sessions(p, t, v, kin, kout).items():
        for c in d.columns:
            f[f"{side}_sess_{c}"] = d[c].to_numpy()
    return f.astype(float)
