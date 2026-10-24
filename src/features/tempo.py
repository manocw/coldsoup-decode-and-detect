"""Motif tempo and hop-sign signatures of one-off-pair groups.

Two generator differences measured on train (true scenarios, analysis only):
- tempo: medium suspicious motifs are SLOWER than their legitimate look-alikes in all 9 motif families
  (median step 17.8 vs 9.5 min, minimum step 9.4 vs 4.4 min; scenario AUC 0.70), while easy
  suspicious motifs step every ~1 min. A model trained mostly on easy cases learns "fast = suspicious"
  and ranks medium below the look-alikes, so the group's pace is given explicitly here.
- hop sign: look-alikes lose value at every hand-off (no rising hops), suspicious motifs random-walk
  (share of rising hand-offs, scenario AUC 0.73 over chain / cycle / fan_out_in / random).

Groups: payments on sender->receiver pairs that occur once in the week, linked when they share a
wallet and are at most dt apart (single linkage through each wallet's time-ordered events), with the
link scales extended to 45 min because medium steps run up to ~40 min. Hand-offs inside a group: a
payment received by w and the next group payment w sends, matched on the smallest |log value ratio|.
"""
import numpy as np
import pandas as pd

from src.features.groups import _components

SCALES = (600, 1800, 2700)
LR_DECAY = (np.log(0.60), np.log(0.92))     # the look-alike per-hop band (lose 8-40%)


def _handoffs(q, comp):
    d = pd.DataFrame({"c": comp, "s": q.sender.to_numpy(), "r": q.receiver.to_numpy(), "t": q.ts.to_numpy(),
                      "vin": q.dst_usd.to_numpy(), "vout": q.src_usd.to_numpy(), "i": np.arange(len(q))})
    ins = d[["c", "r", "t", "vin", "i"]].rename(columns={"r": "w", "t": "t_in", "i": "i_in"})
    outs = d[["c", "s", "t", "vout"]].rename(columns={"s": "w", "t": "t_out"})
    m = ins.merge(outs, on=["c", "w"])
    m = m[m.t_out > m.t_in]
    if m.empty:
        return pd.DataFrame(columns=["up_share", "decay_all", "lr_max", "lr_min", "n_hand"])
    m["lr"] = np.log(m.vout / m.vin)
    m["alr"] = m.lr.abs()
    best = m.sort_values("alr").groupby("i_in", sort=False).first()
    g = best.groupby("c").lr
    out = pd.DataFrame({"up_share": g.apply(lambda s: float((s > 0).mean())),
                        "decay_all": g.apply(lambda s: float(((s >= LR_DECAY[0]) & (s <= LR_DECAY[1])).all())),
                        "lr_max": g.max(), "lr_min": g.min(), "n_hand": g.size()})
    return out


def build(p):
    p = p.reset_index(drop=True)
    pair = p.sender + ">" + p.receiver
    one_off = (pair.map(pair.value_counts()) == 1).to_numpy()
    q = p[one_off].reset_index(drop=True)
    S, R, T = q.sender.to_numpy(), q.receiver.to_numpy(), q.ts.to_numpy()
    f = pd.DataFrame(index=p.index)
    for dt in SCALES:
        tag = f"{dt // 60}m"
        comp = _components(S, R, T, dt)
        d = pd.DataFrame({"c": comp, "t": T}).sort_values(["c", "t"])
        d["gap"] = d.groupby("c").t.diff() / 60
        g = d.groupby("c")
        agg = pd.DataFrame({"n": g.size(), "gap_min": g.gap.min(), "gap_med": g.gap.median(),
                            "gap_cv": g.gap.std() / g.gap.mean(), "span_m": (g.t.max() - g.t.min()) / 60})
        agg.loc[agg.n > 30, ["gap_min", "gap_med", "gap_cv", "span_m"]] = np.nan     # giant: not a motif
        agg["tempo"] = np.log(agg.gap_med.clip(lower=0.1) / 10.0)
        agg["slow_band"] = ((agg.gap_med >= 12) & (agg.gap_med <= 45) & (agg.gap_min >= 6) & (agg.n >= 3)).astype(float)
        agg = agg.join(_handoffs(q, comp))
        per_tx = agg.reindex(comp).reset_index(drop=True)
        for col in ["n", "gap_min", "gap_med", "gap_cv", "span_m", "tempo", "slow_band",
                    "up_share", "decay_all", "lr_max", "lr_min", "n_hand"]:
            v = np.full(len(p), np.nan)
            v[one_off] = per_tx[col].to_numpy(dtype=float)
            f[f"{col}_{tag}"] = v
    return f.astype(float)
