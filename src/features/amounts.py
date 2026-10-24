"""Amount forensics: does the money a payment moves line up with money moving around it?

Static amount properties carry no label signal in this data (measured on train, see the
family report): no round numbers in units or USD, uniform 5-6 decimal places, no excess
mass just under 100/500/1000/3000/5000/10000 thresholds (amounts top out near 4.6k USD),
no repeated near-identical amounts per sender or per receiver, conversion cost vs the
same-pair same-hour median, cross-issuer swaps, currency switches and liquidity-provider
choice all match the legit rates, and the amount relative to the sender's / receiver's
typical amount adds nothing over the starter's log value.

What does carry signal is the amount *relationship* between a payment and the payments
around it (value retention and conservation through a wallet):

  b_dlog / b_dt     paid amount vs the sender's best-matching receipt in the 6 h before
                    (log ratio) and how long ago that receipt arrived (hours)
  f_dlog / f_dt     the receiver's best-matching onward payment in the 6 h after vs the
                    delivered amount (log ratio) and how long after it left (hours)
  b_vs_in           paid amount vs everything the sender received in the 6 h before
                    (how much of the recent inflow this payment passes on)
  f_vs_out          everything the receiver pays on in the 6 h after vs the delivered amount
  s_post            what the sender receives in the 6 h AFTER vs its +-1 h payment burst
                    (fronting: the wallet paid first and is reimbursed later, a legit-hub habit)
  r_pre             what the receiver paid out in the 6 h BEFORE vs its +-1 h receipt burst

Burst-vs-window conservation ratios (sender's +-1 h out-burst / prior 6 h receipts and the
receiver mirror) were measured and dropped: leave-one-out showed them redundant (CV AP 48.0
with, 48.7 / 47.8 without on two seed pairs).

The sender side uses src_usd (what the sender actually spent, first executed leg) and the
receiver side dst_usd (delivered), both at the hourly reference rate, so conversions compare
like with like. Every quantity is relative to the payment's own time and amounts; nothing
depends on addresses, issuers or absolute timestamps. Missing context (no receipt / no
onward payment in the window) is NaN.
"""
import numpy as np
import pandas as pd

HOUR = 3600
WINDOW = 6 * HOUR      # look-back / look-ahead for matches and sums (best of 3 h / 6 h / 12 h in CV)
BURST = 1 * HOUR       # half-width of the burst a payment belongs to


def _index(keys, ts, logv):
    """account -> (sorted times, log values, cumulative linear sums with a leading 0)."""
    order = np.lexsort((ts, keys))
    k = keys[order]
    cut = np.r_[0, np.flatnonzero(k[1:] != k[:-1]) + 1, len(k)]
    out = {}
    for a, b in zip(cut[:-1], cut[1:]):
        ix = order[a:b]
        out[k[a]] = (ts[ix], logv[ix], np.r_[0.0, np.cumsum(np.exp(logv[ix]))])
    return out


def _sum(entry, lo, hi):
    """Sum of an account's values with time in [lo, hi]."""
    ts, _, cs = entry
    return cs[np.searchsorted(ts, hi, "right")] - cs[np.searchsorted(ts, lo, "left")]


def build(p):
    p = p.reset_index(drop=True)
    n = len(p)
    t = p.ts.to_numpy().astype(np.int64)
    vin = p.dst_usd.to_numpy(dtype=float)
    vout = p.src_usd.to_numpy(dtype=float)
    vout = np.where(np.isfinite(vout) & (vout > 0), vout, vin)
    ok = np.isfinite(vin) & (vin > 0) & np.isfinite(vout) & (vout > 0)
    lin = np.log(np.where(ok, vin, 1.0))
    lout = np.log(np.where(ok, vout, 1.0))
    S, R = p.sender.to_numpy(), p.receiver.to_numpy()
    gets = _index(R[ok], t[ok], lin[ok])      # account -> its receipts (delivered USD)
    sends = _index(S[ok], t[ok], lout[ok])    # account -> its payments (spent USD)
    empty = (np.empty(0, np.int64), np.empty(0), np.zeros(1))

    names = ["b_dlog", "b_dt", "f_dlog", "f_dt", "b_vs_in", "f_vs_out", "s_post", "r_pre"]
    o = {c: np.full(n, np.nan) for c in names}
    for i in np.flatnonzero(ok):
        ti = t[i]
        s_get, s_send = gets.get(S[i], empty), sends.get(S[i], empty)
        r_get, r_send = gets.get(R[i], empty), sends.get(R[i], empty)
        out_burst = _sum(s_send, ti - BURST, ti + BURST)    # includes this payment
        in_burst = _sum(r_get, ti - BURST, ti + BURST)      # includes this payment

        # backward: the sender's receipts in [t - W, t)
        ts_, lv_, cs = s_get
        a, b = np.searchsorted(ts_, ti - WINDOW, "left"), np.searchsorted(ts_, ti, "left")
        if b > a:
            d = lout[i] - lv_[a:b]
            j = int(np.argmin(np.abs(d)))
            o["b_dlog"][i] = d[j]
            o["b_dt"][i] = (ti - ts_[a + j]) / HOUR
            o["b_vs_in"][i] = lout[i] - np.log(cs[b] - cs[a])
        # fronting: the sender's receipts in (t, t + W]
        later = _sum(s_get, ti + 1, ti + WINDOW)
        o["s_post"][i] = np.log((later + 1.0) / (out_burst + 1.0))

        # forward: the receiver's payments in (t, t + W]
        ts_, lv_, cs = r_send
        a, b = np.searchsorted(ts_, ti, "right"), np.searchsorted(ts_, ti + WINDOW, "right")
        if b > a:
            d = lv_[a:b] - lin[i]
            j = int(np.argmin(np.abs(d)))
            o["f_dlog"][i] = d[j]
            o["f_dt"][i] = (ts_[a + j] - ti) / HOUR
            o["f_vs_out"][i] = np.log(cs[b] - cs[a]) - lin[i]
        # fronting: the receiver's payments in [t - W, t)
        earlier = _sum(r_send, ti - WINDOW, ti - 1)
        o["r_pre"][i] = np.log((earlier + 1.0) / (in_burst + 1.0))

    f = pd.DataFrame(o, index=p.index)
    return f.replace([np.inf, -np.inf], np.nan).astype(float)
