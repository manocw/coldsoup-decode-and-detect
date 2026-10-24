"""Teammate Marat's flow-episode features (team repo: xrpl_hackathon/marat_model), as a family.

His feature code is vendored unchanged in marat_lib/. This wrapper feeds it our per-transaction
table renamed to his schema and returns exactly the feature set of his predict.py (site 66.8),
minus the episode id (smoothing only). Two variants are exposed via DROP_SHIFTING: his asset /
conversion / clock columns can be dropped because the conversion mix shifts between splits.
"""
import numpy as np
import pandas as pd

from src.features.marat_lib.features import tx_features
from src.features.marat_lib.motifs import motif_features, newpair_features

DEPENDS_ON = ["src.features.marat_lib.features", "src.features.marat_lib.motifs"]
SHIFTING = ["conversion", "n_legs", "spread", "src_is_xrp", "dst_is_xrp", "hour", "frac_week"]
DROP_SHIFTING = False


def to_marat_schema(p):
    df = pd.DataFrame({
        "hash": p.hash.to_numpy(), "ts": p.ts.to_numpy(), "ledger": p.ledger.to_numpy(),
        "tx_index": p.tx_index.to_numpy(), "src": p.sender.to_numpy(), "dst": p.receiver.to_numpy(),
        "fee_drops": p.fee_drops.to_numpy(), "seq": p.sequence.to_numpy(),
        "has_sendmax": p.has_sendmax.to_numpy(), "has_paths": p.has_paths.to_numpy(),
        "n_legs": p.n_legs.to_numpy(), "src_asset": p.src_asset.to_numpy(),
        "dst_asset": p.dst_asset.to_numpy(), "src_usd": p.src_usd.to_numpy(), "dst_usd": p.dst_usd.to_numpy(),
    })
    df["usd"] = df.dst_usd
    df["spread"] = (df.src_usd - df.dst_usd) / df.src_usd.clip(lower=1e-9)
    df["conversion"] = df.src_asset != df.dst_asset
    df["dst_is_xrp"] = df.dst_asset == "XRP"
    df["src_is_xrp"] = df.src_asset == "XRP"
    return df


def build(p):
    df = to_marat_schema(p.reset_index(drop=True))     # p is already time-sorted
    parts = [
        tx_features(df),
        motif_features(df, scales=(600,)).drop(columns="hash"),
        motif_features(df, scales=(300, 1200, 3600, 4 * 3600), nearest=True, prefix="nn").drop(columns="hash"),
        motif_features(df, scales=(1800, 3600, 2 * 3600, 4 * 3600), nearest=True, prefix="vf",
                       siblings=False, band=(0.5, 2.0)).drop(columns="hash"),
        newpair_features(df, scales=(300, 600, 1200), slow_scales=(1800, 2700, 3600, 5400)).drop(columns="hash"),
    ]
    X = pd.concat(parts, axis=1)
    assert (X.hash.to_numpy() == df.hash.to_numpy()).all()
    X = X.drop(columns="hash")
    if DROP_SHIFTING:
        X = X.drop(columns=[c for c in SHIFTING if c in X.columns])
    return X.astype(float).replace([np.inf, -np.inf], np.nan)
