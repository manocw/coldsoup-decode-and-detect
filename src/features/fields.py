"""Raw-field forensics distilled into generic per-transaction context features.

Forensics on train (see the fields report) found that the raw XRPL fields themselves carry no
label signal: SOURCE_FEE is i.i.d. uniform over 10/12/15 drops, LAST_LEDGER_SEQUENCE is always
ledger + 20, FLAGS is 0, SendMax is exactly 1.01 x the first leg's cost whenever (and only when)
the payment converts, Paths is present iff the payment is a 3-leg XRP bridge (always [[XRP]]),
Amount == DeliverMax == delivered amount, the liquidity provider and its spread are random,
TRANSACTION_INDEX is a random order within the ledger, blocks close every 4 s with TOTAL_COINS
falling by exactly the fees burned, and timestamps have no sub-ledger structure. SEQUENCE is the
sender's running payment count in the split (1, 2, 3, ... with no gaps).

What does carry signal is the context those fields encode, described here relative to each
account's own behaviour and to the split's own volume (so it transfers across splits with
different wallets, issuers, start days and transaction counts):

* account cadence: seconds since/until the same sender's previous/next send and the same
  receiver's previous/next receipt (SEQUENCE order), and since/until the wallet's previous/next
  appearance in any role;
* counterparty novelty: how many times this sender->receiver pair occurs (before and in total)
  and whether the reverse pair exists;
* amount against the account's own history: log USD minus the sender's (receiver's) median;
* network load: ledger-window transaction counts around the payment, as a percentile within the
  split (planted motifs are placed uniformly in time, background and relay traffic is diurnal /
  in rounds, so quiet periods are enriched for planted motifs - suspicious and look-alike).
"""
import numpy as np
import pandas as pd

# ledger windows (1 ledger = 4 s): +-1 minute, +-10 minutes, +-1 hour
LOAD_WINDOWS = (("1m", 15), ("10m", 150), ("1h", 900))


def _neighbour_gaps(t, key, tiebreak):
    """log1p seconds to the previous / next row with the same key (NaN at the ends)."""
    o = np.lexsort((tiebreak, t, key))
    k, ts = key[o], t[o]
    same_prev = np.r_[False, k[1:] == k[:-1]]
    same_next = np.r_[k[:-1] == k[1:], False]
    prev = np.where(same_prev, ts - np.r_[ts[0], ts[:-1]], np.nan)
    nxt = np.where(same_next, np.r_[ts[1:], ts[-1]] - ts, np.nan)
    a, b = np.empty(len(t)), np.empty(len(t))
    a[o], b[o] = prev, nxt
    return np.log1p(a), np.log1p(b)


def _any_role_gaps(t, s_code, r_code):
    """log1p seconds since / until the wallet's previous / next appearance in any role."""
    n = len(t)
    w = np.concatenate([s_code, r_code])
    tt = np.concatenate([t, t])
    row = np.concatenate([np.arange(n), np.arange(n)])
    prev, nxt = _neighbour_gaps(tt, w, row)
    return prev[:n], nxt[:n], prev[n:], nxt[n:]


def build(p):
    p = p.reset_index(drop=True)
    n = len(p)
    f = pd.DataFrame(index=p.index)
    t = p.ts.to_numpy().astype(np.int64)
    codes = pd.factorize(np.concatenate([p.sender.to_numpy(), p.receiver.to_numpy()]))[0]
    s_code, r_code = codes[:n], codes[n:]
    seq = p.sequence.to_numpy()

    # account cadence (SEQUENCE orders a sender's same-ledger payments)
    f["snd_prev_gap"], f["snd_next_gap"] = _neighbour_gaps(t, s_code, seq)
    f["rcv_prev_gap"], f["rcv_next_gap"] = _neighbour_gaps(t, r_code, np.arange(n))
    (f["snd_any_prev"], f["snd_any_next"],
     f["rcv_any_prev"], f["rcv_any_next"]) = _any_role_gaps(t, s_code, r_code)

    # counterparty novelty
    pair = s_code.astype(np.int64) * (codes.max() + 1) + r_code
    rev = r_code.astype(np.int64) * (codes.max() + 1) + s_code
    pair_counts = pd.Series(pair).value_counts()
    f["pair_seq"] = pd.Series(pair).groupby(pair).cumcount().to_numpy()
    f["pair_total"] = pd.Series(pair).map(pair_counts).to_numpy()
    f["rev_pair_total"] = pd.Series(rev).map(pair_counts).fillna(0).to_numpy()

    # amount against the account's own history
    lv = pd.Series(np.log(np.clip(p.dst_usd.to_numpy(dtype=float), 1e-6, None)))
    s_med = lv.groupby(s_code).transform("median")
    s_std = lv.groupby(s_code).transform("std")
    f["amt_vs_sender_med"] = (lv - s_med).to_numpy()
    f["amt_vs_sender_z"] = ((lv - s_med) / s_std.where(s_std > 0)).to_numpy()
    f["amt_vs_receiver_med"] = (lv - lv.groupby(r_code).transform("median")).to_numpy()

    # network load as a within-split percentile: monotone, so lossless for trees on one split,
    # and it removes the split-level level shift (test raw load runs ~10% higher than train)
    li = p.ledger.to_numpy().astype(np.int64)
    led = np.sort(li)
    for name, w in LOAD_WINDOWS:
        c = np.searchsorted(led, li + w, "right") - np.searchsorted(led, li - w, "left")
        f[f"load_{name}"] = pd.Series(c).rank(pct=True).to_numpy()
    return f.astype(float)
