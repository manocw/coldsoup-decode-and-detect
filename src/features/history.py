"""Whole-week wallet history and how far one payment deviates from it.

Normal traffic is made of recurring relationships (scheduled pair payments with a fixed period
and a stable amount), one-off commerce and relay rounds. Planted motifs reuse ordinary wallets,
so wallet identity says nothing; what can carry signal is how a payment sits against the
week-long history of its pair, its sender and its receiver:

* pair history: how regular the pair's schedule is (inter-arrival CV), whether this payment
  breaks it (gap to its neighbours vs the pair's median period), its amount and asset against
  the pair's own;
* wallet roles: whole-week in/out counts, distinct counterparties, share of one-off and
  scheduled relationships, value balance, spread over the week (as percentiles among the
  split's wallets so they transfer across splits);
* deviation: the amount's percentile among the sender's payments and receipts (and the
  receiver's), its share of the wallet's weekly volume, how usual the asset is for each side,
  and how familiar the two wallets are (common counterparties).

Measured on train (grouped CV, labels only for analysis; notes/history.md): over all 7 earlier
families (3 seeds) AP 59.5 -> 60.8, AP_medium 6.0 -> 6.9; the wallet-role group carries most of
it. Local activity against the wallet's weekly rate (+-1 h / +-6 h bursts, one-off bursts) was
dropped: redundant with starter/funding windows and it cost ~1 AP. Wallet temporal profiles
(burstiness, sessions), counterparty roles, slow-scale one-off excess, relay-round density,
rate-normalised gaps and fan-out episode recurrence were tried and did not help.

Only this split's own transactions are used, with times relative to each other.
"""
import numpy as np
import pandas as pd

H = 3600.0


def _pct_within(keys, vals):
    """percentile rank of vals within groups of keys (average ties, in (0, 1])."""
    return pd.Series(vals).groupby(keys).rank(pct=True).to_numpy()


def _frac_le(book_keys, book_vals, q_keys, q_vals):
    """for each query, fraction of the book entries of the same key with value <= query value."""
    out = np.full(len(q_keys), np.nan)
    order = np.lexsort((book_vals, book_keys))
    bk, bv = book_keys[order], book_vals[order]
    cut = np.r_[0, np.flatnonzero(bk[1:] != bk[:-1]) + 1, len(bk)]
    start = dict(zip(bk[cut[:-1]], cut[:-1]))
    stop = dict(zip(bk[cut[:-1]], cut[1:]))
    qs = pd.Series(np.arange(len(q_keys))).groupby(q_keys).indices
    for k, ix in qs.items():
        if k not in start:
            continue
        v = bv[start[k]:stop[k]]
        out[ix] = np.searchsorted(v, q_vals[ix], "right") / len(v)
    return out


def build(p):
    p = p.reset_index(drop=True)
    n = len(p)
    t = p.ts.to_numpy().astype(np.int64)
    th = (t - t.min()) / H
    span_h = max(th.max(), 1.0)
    codes, uniq = pd.factorize(np.concatenate([p.sender.to_numpy(), p.receiver.to_numpy()]))
    S, R = codes[:n].astype(np.int64), codes[n:].astype(np.int64)
    W = len(uniq)
    v = p.dst_usd.to_numpy(dtype=float)
    lv = np.log(np.clip(np.nan_to_num(v, nan=1e-6), 1e-6, None))
    assets = pd.factorize(np.concatenate([p.src_asset.to_numpy(), p.dst_asset.to_numpy()]))[0]
    sa, da = assets[:n].astype(np.int64), assets[n:].astype(np.int64)   # one code space for both sides
    A = int(assets.max()) + 1
    f = pd.DataFrame(index=p.index)

    # ---------------- pair history ----------------
    pair = S * W + R
    order = np.lexsort((np.arange(n), t, pair))
    ps, pt = pair[order], th[order]
    same_prev = np.r_[False, ps[1:] == ps[:-1]]
    gp = np.where(same_prev, pt - np.r_[pt[0], pt[:-1]], np.nan)
    gap_prev = np.empty(n); gap_prev[order] = gp
    same_next = np.r_[ps[:-1] == ps[1:], False]
    gn = np.where(same_next, np.r_[pt[1:], pt[-1]] - pt, np.nan)
    gap_next = np.empty(n); gap_next[order] = gn
    gs = pd.Series(gap_prev).groupby(pair)
    pair_n = pd.Series(pair).map(pd.Series(pair).value_counts()).to_numpy()
    g_med = gs.transform("median").to_numpy()
    g_mean = gs.transform("mean").to_numpy()
    g_std = gs.transform("std").to_numpy()
    f["pair_gap_cv"] = np.where(pair_n >= 3, g_std / np.maximum(g_mean, 1e-9), np.nan)
    f["pair_gap_med_h"] = g_med
    dev = np.fmin(np.abs(gap_prev - g_med), np.abs(gap_next - g_med))
    f["pair_gap_break"] = np.where(pair_n >= 3, dev / np.maximum(g_med, 1e-9), np.nan)
    lvs = pd.Series(lv).groupby(pair)
    f["pair_amt_dev"] = np.where(pair_n >= 2, lv - lvs.transform("median").to_numpy(), np.nan)
    f["pair_amt_std"] = np.where(pair_n >= 2, lvs.transform("std").to_numpy(), np.nan)
    pa = pd.Series(pair * A + da)
    f["pair_asset_share"] = np.where(pair_n >= 2, pa.map(pa.value_counts()).to_numpy() / pair_n, np.nan)
    periodic = (pair_n >= 3) & (np.nan_to_num(f.pair_gap_cv.to_numpy(), nan=9) < 0.1)
    oneoff = pair_n == 1

    # ---------------- wallet roles (whole week) ----------------
    def wallet_stat(keys, vals, how):
        s = pd.Series(vals).groupby(keys).agg(how)
        return s.reindex(np.arange(W)).to_numpy()

    out_n = np.bincount(S, minlength=W).astype(float)
    in_n = np.bincount(R, minlength=W).astype(float)
    out_deg = wallet_stat(S, R, "nunique")
    in_deg = wallet_stat(R, S, "nunique")
    out_usd = np.bincount(S, weights=np.nan_to_num(v), minlength=W)
    in_usd = np.bincount(R, weights=np.nan_to_num(v), minlength=W)
    oneoff_out = np.bincount(S, weights=oneoff, minlength=W) / np.maximum(out_n, 1)
    oneoff_in = np.bincount(R, weights=oneoff, minlength=W) / np.maximum(in_n, 1)
    per_out = np.bincount(S, weights=periodic, minlength=W) / np.maximum(out_n, 1)
    per_in = np.bincount(R, weights=periodic, minlength=W) / np.maximum(in_n, 1)
    hours = np.concatenate([S, R]) * 200 + np.concatenate([th, th]).astype(np.int64) // 6
    active = pd.Series(np.concatenate([S, R])).groupby(pd.Series(hours)).first().value_counts()
    active = active.reindex(np.arange(W)).fillna(0).to_numpy() / max(span_h / 6, 1)

    def wpct(x):  # percentile among the split's wallets that appear
        return pd.Series(x).rank(pct=True).to_numpy()

    role = {
        "out_n": wpct(out_n), "in_n": wpct(in_n),
        "out_deg": wpct(np.nan_to_num(out_deg)), "in_deg": wpct(np.nan_to_num(in_deg)),
        "deg_per_tx": (np.nan_to_num(out_deg) + np.nan_to_num(in_deg)) / np.maximum(out_n + in_n, 1),
        "oneoff_out": oneoff_out, "oneoff_in": oneoff_in, "periodic_out": per_out, "periodic_in": per_in,
        "balance": np.log1p(in_usd) - np.log1p(out_usd), "spread": active,
        "out_in_ratio": np.log1p(out_n) - np.log1p(in_n),
    }
    for k, arr in role.items():
        f[f"s_{k}"] = arr[S]
        f[f"r_{k}"] = arr[R]

    # ---------------- deviation of this payment from each wallet's history ----------------
    f["s_amt_pct_out"] = _pct_within(S, lv)
    f["r_amt_pct_in"] = _pct_within(R, lv)
    f["s_amt_pct_in"] = _frac_le(R, lv, S, lv)     # vs what the sender receives
    f["r_amt_pct_out"] = _frac_le(S, lv, R, lv)    # vs what the receiver pays out
    f["s_amt_share_out"] = np.nan_to_num(v) / np.maximum(out_usd[S], 1e-9)
    f["r_amt_share_in"] = np.nan_to_num(v) / np.maximum(in_usd[R], 1e-9)
    sk, rk = pd.Series(S * A + sa), pd.Series(R * A + da)
    sout, rin = sk.value_counts(), rk.value_counts()
    f["s_asset_share_out"] = sk.map(sout).to_numpy() / out_n[S]
    f["r_asset_share_in"] = rk.map(rin).to_numpy() / in_n[R]
    # does the sender also receive the asset it pays with; does the receiver also spend what it got
    f["s_asset_share_in"] = sk.map(rin).fillna(0).to_numpy() / np.maximum(in_n[S], 1)
    f["r_asset_share_out"] = rk.map(sout).fillna(0).to_numpy() / np.maximum(out_n[R], 1)

    # familiarity: counterparties the two wallets share over the week (undirected, excluding each other)
    a = np.minimum(S, R); b = np.maximum(S, R)
    und = pd.DataFrame({"a": np.r_[a, b], "b": np.r_[b, a]}).drop_duplicates()
    nb = und.groupby("a").b.apply(lambda s: set(s.to_numpy())).to_dict()
    cn = np.zeros(n)
    jac = np.zeros(n)
    cache = {}
    for i in range(n):
        key = (S[i], R[i]) if S[i] < R[i] else (R[i], S[i])
        if key not in cache:
            x = nb.get(key[0], set()) - {key[1]}
            y = nb.get(key[1], set()) - {key[0]}
            c = len(x & y)
            cache[key] = (c, c / max(len(x | y), 1))
        cn[i], jac[i] = cache[key]
    f["common_cp"] = cn
    f["common_cp_jac"] = jac
    return f.astype(float)
