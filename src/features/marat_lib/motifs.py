"""Motif reconstruction: link transactions into flow episodes and describe each episode.

Two transactions are linked when money plausibly flows through an account
(payment into account A, then a payment out of A within W seconds), or when they
are near-simultaneous siblings (same sender, or same receiver, within W/4).
Connected groups of linked transactions approximate the planted scenarios
(chains, cycles, fan-outs, stacks...). Each transaction gets features of its group.
"""
from collections import defaultdict
import numpy as np
import pandas as pd


class DSU:
    def __init__(self, n):
        self.p = np.arange(n)

    def find(self, x):
        p = self.p
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[max(a, b)] = min(a, b)


def _flow_links(df, W, nearest=False, band=None):
    """(i, j): tx i pays into account A, tx j pays out of A, 0 <= t_j - t_i <= W, i before j.
    nearest=True keeps only the latest inflow before each outflow and the first outflow after
    each inflow, which stops busy accounts from merging everything into one giant episode."""
    ins = defaultdict(list); outs = defaultdict(list); usd = df.usd.values
    ok = (lambda i, j: band[0] <= usd[j] / usd[i] <= band[1]) if band else (lambda i, j: True)
    for i, (s, d, t) in enumerate(zip(df.src.values, df.dst.values, df.ts.values)):
        ins[d].append((t, i)); outs[s].append((t, i))
    links = []
    for a, o in outs.items():
        if a not in ins:
            continue
        it = np.array([x[0] for x in ins[a]]); ii = np.array([x[1] for x in ins[a]])
        ot = np.array([x[0] for x in o]); oi = np.array([x[1] for x in o])
        if not nearest:
            for tj, j in o:
                lo = np.searchsorted(it, tj - W, "left"); hi = np.searchsorted(it, tj, "right")
                for k in range(lo, hi):
                    if ii[k] < j:
                        links.append((ii[k], j))
            continue
        for tj, j in o:                      # latest inflow before each outflow
            hi = np.searchsorted(it, tj, "right") - 1
            while hi >= 0 and ii[hi] > j:
                hi -= 1
            if hi >= 0 and tj - it[hi] <= W and ok(ii[hi], j):
                links.append((ii[hi], j))
        for ti, i in ins[a]:                 # first outflow after each inflow
            lo = np.searchsorted(ot, ti, "left")
            while lo < len(ot) and oi[lo] < i:
                lo += 1
            if lo < len(ot) and ot[lo] - ti <= W and ok(i, oi[lo]):
                links.append((i, oi[lo]))
    return list(set(links))


def _sibling_links(df, col, W):
    links = []
    for _, g in df.groupby(col, sort=False):
        idx = g.index.values; t = g.ts.values
        o = np.argsort(t, kind="stable"); idx, t = idx[o], t[o]
        close = np.diff(t) <= W
        links += list(zip(idx[:-1][close], idx[1:][close]))
    return links


def motif_features(df, scales=(600, 3600, 6 * 3600), nearest=False, prefix="", siblings=True, band=None,
                   sib_div=4, gap_stats=False):
    """df must be sorted by time with a 0..n-1 index (load_split does this)."""
    df = df.reset_index(drop=True)
    n = len(df)
    out = {}
    usd = df.usd.values; ts = df.ts.values
    for W in scales:
        tag = f"{prefix}{W // 60}m"
        flow = _flow_links(df, W, nearest, band)
        sib = (_sibling_links(df, "src", W // sib_div) + _sibling_links(df, "dst", W // sib_div)) if siblings else []

        # longest flow chain through each tx (links always go forward in index order)
        fwd_adj = defaultdict(list); bwd_adj = defaultdict(list)
        for i, j in flow:
            fwd_adj[i].append(j); bwd_adj[j].append(i)
        back = np.zeros(n, dtype=int)
        for j in range(n):
            if bwd_adj[j]:
                back[j] = 1 + max(back[i] for i in bwd_adj[j])
        ahead = np.zeros(n, dtype=int)
        for i in range(n - 1, -1, -1):
            if fwd_adj[i]:
                ahead[i] = 1 + max(ahead[j] for j in fwd_adj[i])
        out[f"chain_len_{tag}"] = back + ahead + 1
        out[f"chain_pos_{tag}"] = back / np.maximum(1, back + ahead)
        out[f"n_in_links_{tag}"] = np.array([len(bwd_adj[j]) for j in range(n)])
        out[f"n_out_links_{tag}"] = np.array([len(fwd_adj[i]) for i in range(n)])

        # value carried along best link: ratio of this tx to its largest predecessor
        pred_ratio = np.full(n, -1.0)
        for j in range(n):
            if bwd_adj[j]:
                pred_ratio[j] = usd[j] / max(usd[i] for i in bwd_adj[j])
        out[f"pred_ratio_{tag}"] = pred_ratio

        # connected episodes
        dsu = DSU(n)
        for i, j in flow + sib:
            dsu.union(i, j)
        comp = np.array([dsu.find(i) for i in range(n)])
        g = pd.DataFrame({"c": comp, "src": df.src.values, "dst": df.dst.values, "usd": usd, "ts": ts,
                          "conv": df.conversion.values.astype(float), "depth": back + ahead + 1})
        gc = g.groupby("c")
        agg = gc.agg(size=("usd", "size"), ts_min=("ts", "min"), ts_max=("ts", "max"),
                     usd_mean=("usd", "mean"), usd_std=("usd", "std"),
                     usd_sum=("usd", "sum"), conv=("conv", "mean"), depth=("depth", "max"),
                     n_src=("src", "nunique"), n_dst=("dst", "nunique"))
        agg["span"] = agg.ts_max - agg.ts_min
        agg["usd_cv"] = (agg.usd_std / agg.usd_mean).fillna(0)
        # accounts that both send and receive inside the episode (intermediaries), and
        # whether one of them receives after it first sent (money comes back: a cycle)
        first_send = g.groupby(["c", "src"]).ts.min().rename_axis(["c", "a"])
        last_get = g.groupby(["c", "dst"]).ts.max().rename_axis(["c", "a"])
        both = pd.concat([first_send.rename("fs"), last_get.rename("lg")], axis=1, join="inner")
        inter = both.groupby(level="c").size().reindex(agg.index).fillna(0)
        agg["intermediaries"] = inter
        agg["accounts"] = agg.n_src + agg.n_dst - inter
        agg["inter_share"] = inter / agg.accounts
        agg["tx_per_acct"] = agg["size"] / agg.accounts
        agg["log_span"] = np.log1p(agg.span)
        agg["rate"] = agg["size"] / (agg.span / 60 + 1)
        agg["origins"] = agg.n_src - inter          # send-only accounts
        agg["sinks"] = agg.n_dst - inter            # receive-only accounts
        agg["has_cycle"] = (both.lg > both.fs).groupby(level="c").any().reindex(agg.index).fillna(False).astype(float)
        for c in ["size", "log_span", "usd_cv", "conv", "depth", "n_src", "n_dst", "accounts",
                  "intermediaries", "inter_share", "tx_per_acct", "rate", "origins", "sinks", "has_cycle"]:
            out[f"ep_{c}_{tag}"] = agg[c].reindex(comp).values
        out[f"ep_log_usd_{tag}"] = np.log1p(agg.usd_sum.reindex(comp).values)
        if gap_stats:   # timing rhythm inside the episode: median gap between consecutive payments
            gg = g.sort_values(["c", "ts"])
            gap = gg.groupby("c").ts.diff()
            gs = gap.groupby(gg.c).agg(["median", "mean", "std", "max"])
            out[f"ep_gap_med_{tag}"] = np.log1p(gs["median"].reindex(comp).fillna(-1).values.clip(min=0))
            out[f"ep_gap_max_{tag}"] = np.log1p(gs["max"].reindex(comp).fillna(-1).values.clip(min=0))
            out[f"ep_gap_cv_{tag}"] = (gs["std"] / gs["mean"]).reindex(comp).fillna(-1).values
        out[f"ep_size_capped_{tag}"] = np.minimum(out[f"ep_size_{tag}"], 200)
    X = pd.DataFrame(out)
    X.insert(0, "hash", df.hash.values)
    return X


def episode_ids(df, W=600):
    """Episode id per transaction at the fast 10-minute scale (same links as motif_features)."""
    df = df.reset_index(drop=True)
    links = _flow_links(df, W) + _sibling_links(df, "src", W // 4) + _sibling_links(df, "dst", W // 4)
    dsu = DSU(len(df))
    for i, j in links:
        dsu.union(i, j)
    return np.array([dsu.find(i) for i in range(len(df))])


def newpair_features(df, scales=(300, 600, 1200, 2400, 4800), max_pair=1, slow_scales=()):
    """Episode features built only from first-time sender-receiver pairs.

    Ordinary traffic repeats the same pairs through the week; planted patterns and their
    lookalikes mostly use pairs seen once. Restricting episodes to those payments removes
    the noise from busy accounts, so wider windows (needed for slow patterns) stay clean.
    Payments on repeated pairs get -1.
    """
    df = df.reset_index(drop=True)
    pair_n = df.groupby(["src", "dst"]).hash.transform("size").values
    keep = pair_n <= max_pair
    sub = df[keep].reset_index(drop=True)
    F = motif_features(sub, scales=scales, prefix="np", sib_div=1, gap_stats=True).drop(columns="hash")
    out = pd.DataFrame(-1.0, index=range(len(df)), columns=F.columns)
    out.loc[np.where(keep)[0], :] = F.values
    if slow_scales:   # slow patterns: nearest links only, so wide windows do not merge everything
        S = motif_features(sub, scales=slow_scales, nearest=True, prefix="nps", sib_div=2, gap_stats=True).drop(columns="hash")
        S2 = pd.DataFrame(-1.0, index=range(len(df)), columns=S.columns)
        S2.loc[np.where(keep)[0], :] = S.values
        out = pd.concat([out, S2], axis=1)
    out["np_is_new_pair"] = keep.astype(float)
    # local new-pair bursts around each payment (fan-out / fan-in rhythm)
    for W in (120, 300, 600, 1200):
        for col, name in (("src", "s"), ("dst", "d")):
            cnt = np.zeros(len(df))
            for a, g in sub.groupby(col, sort=False):
                t = np.sort(g.ts.values)
                idx_all = np.where((df[col].values == a))[0]
                tt = df.ts.values[idx_all]
                cnt[idx_all] = np.searchsorted(t, tt + W, "right") - np.searchsorted(t, tt - W, "left")
            out[f"np_{name}_burst_{W // 60}m"] = cnt
    out.insert(0, "hash", df.hash.values)
    return out
