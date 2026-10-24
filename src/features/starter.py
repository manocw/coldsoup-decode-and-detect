"""The organiser starter notebook's 12 rules, re-implemented on the shared table.

Kept as the reference baseline: every new feature family is measured as a gain over this.
"""
import numpy as np
import pandas as pd

HOUR, TEN_MINUTES = 3600, 600


def _by_account(p, col):
    return {a: (g.ts.to_numpy(), g.dst_usd.to_numpy()) for a, g in p.groupby(col, sort=False)}


def build(p):
    t = p.ts.to_numpy()
    v = p.dst_usd.to_numpy()
    sends = _by_account(p, "sender")
    gets = _by_account(p, "receiver")
    empty = (np.empty(0, np.int64), np.empty(0))

    def window(d, a, ti, before, after):
        times, _ = d.get(a, empty)
        return int(np.searchsorted(times, ti + after, "right") - np.searchsorted(times, ti - before, "left"))

    def fwd_similar(a, ti, amt):
        times, vals = sends.get(a, empty)
        lo, hi = np.searchsorted(times, ti, "right"), np.searchsorted(times, ti + HOUR, "right")
        return bool(np.any(np.abs(vals[lo:hi] - amt) <= 0.1 * amt))

    def got_similar(a, ti, amt):
        times, vals = gets.get(a, empty)
        lo, hi = np.searchsorted(times, ti - HOUR, "left"), np.searchsorted(times, ti, "left")
        return bool(np.any(np.abs(vals[lo:hi] - amt) <= 0.1 * amt))

    def minutes_to_next_send(a, ti):
        times, _ = sends.get(a, empty)
        i = np.searchsorted(times, ti, "right")
        return min(60.0, (times[i] - ti) / 60) if i < len(times) else 60.0

    S, R = p.sender.to_numpy(), p.receiver.to_numpy()
    f = pd.DataFrame(index=p.index)
    f["log_value"] = np.log1p(v)
    f["sender_payments"] = p.sender.map(p.sender.value_counts())
    f["receiver_payments"] = p.receiver.map(p.receiver.value_counts())
    f["receiver_forwards_1h"] = [window(sends, a, ti, 0, HOUR) > 0 for a, ti in zip(R, t)]
    f["sender_received_1h_before"] = [window(gets, a, ti, HOUR, 0) > 0 for a, ti in zip(S, t)]
    f["sender_burst_10m"] = [window(sends, a, ti, TEN_MINUTES, TEN_MINUTES) for a, ti in zip(S, t)]
    f["receiver_burst_10m"] = [window(gets, a, ti, TEN_MINUTES, TEN_MINUTES) for a, ti in zip(R, t)]
    f["receiver_forwards_similar_1h"] = [fwd_similar(a, ti, x) for a, ti, x in zip(R, t, v)]
    f["sender_received_similar_1h"] = [got_similar(a, ti, x) for a, ti, x in zip(S, t, v)]
    f["receiver_minutes_to_forward"] = [minutes_to_next_send(a, ti) for a, ti in zip(R, t)]
    sent_total = p.groupby("sender").dst_usd.sum()
    got_total = p.groupby("receiver").dst_usd.sum()
    pass_through = (sent_total / got_total.reindex(sent_total.index)).clip(0, 3)
    f["sender_pass_through"] = p.sender.map(pass_through).fillna(0)
    f["receiver_pass_through"] = p.receiver.map(pass_through).fillna(0)
    return f.astype(float)
