"""Who funded it / mule network finder.

Behavioural features about where a payment's money came from and where it goes next:

- funding: how recently the sender was funded (by anyone / by a one-off payer), how big that
  funding was relative to this payment, whether funding arrives before (pass-through) or only
  after (replenishment) the payment, the last funder's fan-out to wallets that pay onward
  (mules), and how much of the funding the sender splits out again;
- fan-out / fan-in: distinct counterparties of the sender and the receiver in short windows,
  receivers that pay onward to a common target, senders that share a recent funder, and
  many-to-many (bipartite) co-payers;
- one-off payments: a sender->receiver relationship used exactly once in the week is the edge
  of a mule network. We count such payments around the sender and receiver, against the
  wallet's own weekly rate, the 1- and 2-hop neighbourhood they form, and the time to the
  nearest one-off payment sharing a wallet (planted motifs fire them seconds apart).

Everything is computed from one split's transactions relative to each payment's own time, so
nothing depends on specific addresses, labels or absolute timestamps.
"""
from collections import defaultdict

import numpy as np
import pandas as pd

M5, M10, M30, H1, H2, H3, H6 = 300, 600, 1800, 3600, 7200, 10800, 21600
WEEK_H = 168.0


def _index(t, keys):
    """wallet -> (sorted times, row indices in time order)."""
    out = {}
    for a, ix in pd.Series(keys).groupby(keys, sort=False).indices.items():
        ix = ix[np.argsort(t[ix], kind="stable")]
        out[a] = (t[ix], ix)
    return out


def build(p):
    p = p.reset_index(drop=True)
    n = len(p)
    t = p.ts.to_numpy().astype(np.int64)
    v = np.nan_to_num(p.dst_usd.to_numpy(dtype=float))
    S, R = p.sender.to_numpy(), p.receiver.to_numpy()
    sends, gets = _index(t, S), _index(t, R)
    E = (np.empty(0, np.int64), np.empty(0, np.int64))

    def rng(d, a, lo, hi, lo_incl=True, hi_incl=True):
        tt, ix = d.get(a, E)
        i0 = np.searchsorted(tt, lo, "left" if lo_incl else "right")
        i1 = np.searchsorted(tt, hi, "right" if hi_incl else "left")
        return ix[i0:i1]

    pair = pd.Series(S) + ">" + pd.Series(R)
    novel = (pair.map(pair.value_counts()) == 1).to_numpy()  # one-off relationship this week

    # one-off payments indexed by both wallets they touch (the mule network's edges)
    nv = np.flatnonzero(novel)
    tw = pd.DataFrame({"w": np.concatenate([S[nv], R[nv]]), "i": np.concatenate([nv, nv])})
    tw["t"] = t[tw.i.to_numpy()]
    tw = tw.sort_values(["t", "i"], kind="mergesort")
    touch = {a: (g.t.to_numpy(), g.i.to_numpy()) for a, g in tw.groupby("w", sort=False)}
    # each wallet's weekly rate of one-off payments out / in (per hour)
    span_h = max((t.max() - t.min()) / 3600, 1.0)
    rate_out = pd.Series(novel).groupby(S).sum() / span_h
    rate_in = pd.Series(novel).groupby(R).sum() / span_h

    cols = defaultdict(lambda: np.full(n, np.nan))
    for i in range(n):
        s, r, ti, vi = S[i], R[i], t[i], v[i]

        # --- who funded the sender (before) and is it replenished instead (after)
        before = rng(gets, s, ti - H6, ti, hi_incl=False)
        after = rng(gets, s, ti, ti + H6, lo_incl=False)
        b2 = before[t[before] >= ti - H2]
        cols["s_in_gap_h"][i] = (ti - t[before[-1]]) / 3600 if len(before) else 6.0
        nb = before[novel[before]]
        cols["s_novel_in_gap_h"][i] = (ti - t[nb[-1]]) / 3600 if len(nb) else 6.0
        cols["s_in_2h_ratio"][i] = np.log1p(v[b2].sum()) - np.log1p(vi)
        cols["s_fund_order_6h"][i] = np.log1p(v[before].sum()) - np.log1p(v[after].sum())
        if len(b2):
            j = b2[-1]  # the most recent funding payment
            F, tj = S[j], t[j]
            fr = set(R[rng(sends, F, tj - H2, tj + H2)])
            cols["s_funder_fanout_2h"][i] = len(fr)
            cols["s_funder_mules_2h"][i] = sum(
                1 for x in fr if len(rng(sends, x, tj - M30, tj + H2)) > 0)
            cols["s_split_ratio"][i] = v[rng(sends, s, tj, tj + H2)].sum() / max(v[j], 1e-9)

        # --- sender fan-out, receiver fan-in
        for w, name in ((M5, "5m"), (M30, "30m"), (H2, "2h")):
            cols[f"s_distinct_recv_{name}"][i] = len(set(R[rng(sends, s, ti - w, ti + w)]))
            cols[f"r_distinct_send_{name}"][i] = len(set(S[rng(gets, r, ti - w, ti + w)]))

        # --- one-off payments around the sender and receiver
        for w, name in ((M10, "10m"), (H1, "1h"), (H3, "3h")):
            so = novel[rng(sends, s, ti - w, ti + w)].sum()
            ri = novel[rng(gets, r, ti - w, ti + w)].sum()
            cols[f"s_novel_out_{name}"][i] = so
            cols[f"r_novel_in_{name}"][i] = ri
            if w >= H1:  # against the wallet's own weekly rate of one-off payments
                cols[f"s_novel_out_excess_{name}"][i] = so - rate_out.get(s, 0.0) * 2 * w / 3600
                cols[f"r_novel_in_excess_{name}"][i] = ri - rate_in.get(r, 0.0) * 2 * w / 3600
        for w, name in ((M30, "30m"), (H3, "3h")):
            cols[f"r_novel_out_after_{name}"][i] = novel[rng(sends, r, ti, ti + w, lo_incl=False)].sum()
            cols[f"s_novel_in_before_{name}"][i] = novel[rng(gets, s, ti - w, ti, hi_incl=False)].sum()

        # --- mules: the sender's receivers paying onward, to a common target; co-payers
        recs = set(R[rng(sends, s, ti - H2, ti + H2)])
        onward, coparents, n_onward = defaultdict(int), defaultdict(int), 0
        for x in recs:
            xo = rng(sends, x, ti - H2, ti + 2 * H2)
            if len(xo):
                n_onward += 1
                for y in set(R[xo]):
                    onward[y] += 1
            for y in set(S[rng(gets, x, ti - H2, ti + H2)]):
                if y != s:
                    coparents[y] += 1
        cols["s_recv_onward_2h"][i] = n_onward
        cols["s_recv_common_target_2h"][i] = max(onward.values(), default=0)
        cols["bipartite_copayers_2h"][i] = sum(1 for c in coparents.values() if c >= 2)

        # --- collector: do the receiver's senders share a recent funder; does it pass money on
        funders = defaultdict(int)
        for x in set(S[rng(gets, r, ti - H2, ti + H2)]):
            for y in set(S[rng(gets, x, ti - 2 * H2, ti + H2)]):
                if y != r:
                    funders[y] += 1
        cols["r_common_funder_2h"][i] = max(funders.values(), default=0)
        ra = rng(sends, r, ti, ti + H6, lo_incl=False)
        rb = rng(sends, r, ti - H6, ti, hi_incl=False)
        cols["r_fwd_order_6h"][i] = np.log1p(v[ra].sum()) - np.log1p(v[rb].sum())

        # --- seconds to the nearest one-off payments sharing a wallet (within 2h)
        near = np.concatenate([np.abs(t[rng(touch, s, ti - H2, ti + H2)] - ti),
                               np.abs(t[rng(touch, r, ti - H2, ti + H2)] - ti)])
        own = 2 if novel[i] else 0  # this payment itself appears once per wallet
        near = np.sort(near)[own:] if own else np.sort(near)
        cols["novel_gap1_min"][i] = near[0] / 60 if len(near) > 0 else 120.0
        cols["novel_gap2_min"][i] = near[1] / 60 if len(near) > 1 else 120.0

    # --- local mule network: one-off payments within +-W sharing a wallet with this payment
    # (1 hop) or with any wallet reached that way (2 hops)
    for w, name in ((M10, "10m"), (M30, "30m"), (H1, "1h"), (H3, "3h")):
        for i in range(n):
            lo, hi = t[i] - w, t[i] + w
            hop1 = set(rng(touch, S[i], lo, hi)) | set(rng(touch, R[i], lo, hi))
            hop1.add(i)
            h1 = list(hop1)
            hop2 = set(hop1)
            for x in set(S[h1]) | set(R[h1]):
                hop2.update(rng(touch, x, lo, hi))
            cols[f"net1_{name}"][i] = len(hop1)
            cols[f"net2_{name}"][i] = len(hop2)

    return pd.DataFrame(dict(cols)).astype(float)
