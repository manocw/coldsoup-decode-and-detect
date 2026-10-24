"""Transaction-level features computed from the whole supplied window of one split.

Every feature is computed inside the split being scored (train, validation or test),
never across splits, because wallets do not repeat between splits.
"""
from collections import defaultdict
import numpy as np
import pandas as pd

H = 3600
WINDOWS = {"1h": H, "6h": 6 * H, "24h": 24 * H, "72h": 72 * H}


def _index(df):
    """Per-account sorted event arrays for sends and receipts."""
    sends, gets = {}, {}
    for a, g in df.groupby("src", sort=False):
        o = np.argsort(g.ts.values, kind="stable")
        sends[a] = (g.ts.values[o], g.usd.values[o], g.dst.values[o], g.index.values[o])
    for a, g in df.groupby("dst", sort=False):
        o = np.argsort(g.ts.values, kind="stable")
        gets[a] = (g.ts.values[o], g.usd.values[o], g.src.values[o], g.index.values[o])
    return sends, gets


_EMPTY = (np.array([], dtype=np.int64), np.array([]), np.array([], dtype=object), np.array([], dtype=np.int64))


def _slice(ev, lo, hi, left_closed=True, right_closed=True):
    ts = ev[0]
    a = np.searchsorted(ts, lo, "left" if left_closed else "right")
    b = np.searchsorted(ts, hi, "right" if right_closed else "left")
    return a, b


def account_features(df):
    """Whole-window behaviour of each account."""
    out = df.groupby("src").agg(out_n=("usd", "size"), out_sum=("usd", "sum"), out_uniq=("dst", "nunique"),
                                out_conv=("conversion", "mean"), out_med=("usd", "median"),
                                out_cv=("usd", lambda v: v.std() / v.mean() if len(v) > 1 else 0),
                                out_assets=("src_asset", "nunique"))
    inn = df.groupby("dst").agg(in_n=("usd", "size"), in_sum=("usd", "sum"), in_uniq=("src", "nunique"),
                                in_conv=("conversion", "mean"), in_med=("usd", "median"),
                                in_assets=("dst_asset", "nunique"))
    a = out.join(inn, how="outer").fillna(0)
    a["pass_through"] = np.minimum(a.out_sum, a.in_sum) / np.maximum(a.out_sum, a.in_sum).clip(lower=1e-9)
    a["out_repeat"] = a.out_n / a.out_uniq.clip(lower=1)          # recurring relationships -> high
    a["in_repeat"] = a.in_n / a.in_uniq.clip(lower=1)
    a["deg_ratio"] = (a.out_uniq + 1) / (a.in_uniq + 1)
    a["n_total"] = a.out_n + a.in_n
    # share of counterparties seen only once (changing counterparties)
    pair_out = df.groupby(["src", "dst"]).size()
    pair_in = df.groupby(["dst", "src"]).size()
    a["out_once_share"] = (pair_out == 1).groupby(level=0).mean().reindex(a.index).fillna(0)
    a["in_once_share"] = (pair_in == 1).groupby(level=0).mean().reindex(a.index).fillna(0)
    # burstiness: median gap between consecutive sends / receipts
    for col, key in [("src", "out"), ("dst", "in")]:
        gaps = df.sort_values("ts").groupby(col).ts.apply(lambda t: np.median(np.diff(t.values)) if len(t) > 1 else 7 * 86400)
        a[f"{key}_gap_med"] = np.log1p(gaps.reindex(a.index).fillna(7 * 86400))
    return a


def tx_features(df):
    df = df.reset_index(drop=True)
    sends, gets = _index(df)
    acc = account_features(df)
    n = len(df)
    F = defaultdict(lambda: np.zeros(n))

    pair_count = df.groupby(["src", "dst"]).size()
    rev_pair = set(zip(df.dst, df.src))

    for i, (s, d, t, v) in enumerate(zip(df.src.values, df.dst.values, df.ts.values, df.usd.values)):
        ds, dg = sends.get(d, _EMPTY), gets.get(d, _EMPTY)
        ss, sg = sends.get(s, _EMPTY), gets.get(s, _EMPTY)
        for name, W in WINDOWS.items():
            # receiver forwards after this payment
            a, b = _slice(ds, t, t + W, left_closed=False)
            F[f"d_fwd_n_{name}"][i] = b - a
            F[f"d_fwd_ratio_{name}"][i] = ds[1][a:b].sum() / v
            F[f"d_fwd_uniq_{name}"][i] = len(set(ds[2][a:b]))
            # receiver collected before this payment (fan-in build-up)
            a2, b2 = _slice(dg, t - W, t, right_closed=False)
            F[f"d_in_n_{name}"][i] = b2 - a2
            F[f"d_in_uniq_{name}"][i] = len(set(dg[2][a2:b2]))
            # balance: what d forwards after vs what it collected before (incl. this one)
            F[f"d_balance_{name}"][i] = ds[1][a:b].sum() / (dg[1][a2:b2].sum() + v)
            # sender received just before (passing along)
            a3, b3 = _slice(sg, t - W, t, right_closed=False)
            F[f"s_got_n_{name}"][i] = b3 - a3
            F[f"s_got_ratio_{name}"][i] = sg[1][a3:b3].sum() / v
            # sender burst around this payment (fan-out)
            a4, b4 = _slice(ss, t - W, t + W)
            F[f"s_burst_n_{name}"][i] = b4 - a4
            F[f"s_burst_uniq_{name}"][i] = len(set(ss[2][a4:b4]))
            amts = ss[1][a4:b4]
            F[f"s_burst_cv_{name}"][i] = amts.std() / amts.mean() if len(amts) > 1 else -1
            # receiver burst around (fan-in)
            a5, b5 = _slice(dg, t - W, t + W)
            F[f"d_burst_n_{name}"][i] = b5 - a5
            F[f"d_burst_uniq_{name}"][i] = len(set(dg[2][a5:b5]))

        # time until receiver next sends / since sender last received
        j = np.searchsorted(ds[0], t, "right")
        F["d_next_send_h"][i] = (ds[0][j] - t) / H if j < len(ds[0]) else 999
        j = np.searchsorted(sg[0], t, "left") - 1
        F["s_last_get_h"][i] = (t - sg[0][j]) / H if j >= 0 else 999

        # 2-hop: does someone the receiver pays within 24h pay on within 24h?
        a, b = _slice(ds, t, t + 24 * H, left_closed=False)
        hop2 = 0; ret = 0
        for k in range(a, b):
            e, te = ds[2][k], ds[0][k]
            if e == s:
                ret = 1
            es = sends.get(e, _EMPTY)
            a6, b6 = _slice(es, te, te + 24 * H, left_closed=False)
            hop2 += b6 - a6
            if b6 > a6 and s in set(es[2][a6:b6]):
                ret = 1
        F["d_hop2_n_24h"][i] = hop2
        F["returns_to_s_48h"][i] = ret

        # pair history
        sd = ss[2] == d
        F["pair_before"][i] = np.sum(sd & (ss[0] < t))
        F["pair_after"][i] = np.sum(sd & (ss[0] > t))

    X = pd.DataFrame(F)
    X["pair_total"] = [pair_count[(s, d)] for s, d in zip(df.src, df.dst)]
    X["reciprocal"] = [(s, d) in rev_pair for s, d in zip(df.src, df.dst)]
    X["log_usd"] = np.log1p(df.usd)
    X["conversion"] = df.conversion.astype(float)
    X["n_legs"] = df.n_legs
    X["spread"] = df.spread
    X["src_is_xrp"] = df.src_is_xrp.astype(float)
    X["dst_is_xrp"] = df.dst_is_xrp.astype(float)
    X["hour"] = (df.ts % 86400) / H
    X["frac_week"] = (df.ts - df.ts.min()) / max(1, df.ts.max() - df.ts.min())

    for side, col in [("s", "src"), ("d", "dst")]:
        A = acc.reindex(df[col]).reset_index(drop=True)
        A.columns = [f"{side}_{c}" for c in A.columns]
        X = pd.concat([X, A], axis=1)
    # relative size of this payment for the sender / receiver
    X["v_rel_s_out"] = df.usd / X.s_out_med.clip(lower=1e-9)
    X["v_rel_d_in"] = df.usd / X.d_in_med.clip(lower=1e-9)
    X.insert(0, "hash", df.hash.values)
    return X
