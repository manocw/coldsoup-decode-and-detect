"""Load one split into a single per-transaction table.

One row per transaction. Money truth is the delivered amount (destination side) and the
first executed transfer leg (source side: the sender's actual cost). Every asset is keyed
as "XRP" or "issuer:currency" and valued in reference USD with the hourly market rate.
"""
import json
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
XRP_SENTINEL = "0x1111222211112222111122221111222211112222"
SPLITS = ("train", "validation", "test")


def split_dir(split):
    return DATA / split


def _asset_key(token_address, symbol):
    return np.where(token_address == XRP_SENTINEL, "XRP", token_address + ":" + symbol)


def _decode_delivered(text):
    wire = json.loads(text)
    if isinstance(wire, str):
        return "XRP", int(wire) / 1e6
    return f"{wire['issuer']}:{wire['currency']}", float(Decimal(wire["value"]))


def _rates(directory):
    r = pd.read_csv(directory / "market_rates.csv")
    r["price"] = r.PRICE_MICRO_USD / 1e6
    return r.set_index(["TIMESTAMP", "ASSET"]).price


def _usd(rates, ts, asset, units):
    hour = (np.asarray(ts, dtype=np.int64) // 3600) * 3600
    idx = pd.MultiIndex.from_arrays([hour, np.asarray(asset)])
    return np.asarray(units, dtype=float) * rates.reindex(idx).to_numpy()


def load_split(split, with_labels=True):
    """Return the per-transaction table for a split, sorted by time (stable order)."""
    d = split_dir(split)
    tx = pd.read_parquet(d / "transactions.parquet")
    tr = pd.read_parquet(d / "transfers.parquet")
    rates = _rates(d)

    p = pd.DataFrame({
        "hash": tx.TRANSACTION_HASH.to_numpy(),
        "ts": tx.TIMESTAMP.astype("int64").to_numpy(),
        "ledger": tx.LEDGER_INDEX.astype("int64").to_numpy(),
        "tx_index": tx.TRANSACTION_INDEX.astype("int64").to_numpy(),
        "sender": tx.ACCOUNT.to_numpy(),
        "fee_drops": tx.SOURCE_FEE.astype("int64").to_numpy(),
        "sequence": tx.SEQUENCE.astype("int64").to_numpy(),
        "last_ledger_offset": (tx.LAST_LEDGER_SEQUENCE.astype("int64")
                               - tx.LEDGER_INDEX.astype("int64")).to_numpy(),
    })
    extra = [json.loads(x) for x in tx.EXTRA_TRANSACTION_FIELDS]
    p["receiver"] = [e["Destination"] for e in extra]
    p["has_sendmax"] = [("SendMax" in e) for e in extra]
    p["has_paths"] = [("Paths" in e) for e in extra]

    dlv = [_decode_delivered(x) for x in tx.DELIVERED_AMOUNT]
    p["dst_asset"] = [a for a, _ in dlv]
    p["dst_units"] = [u for _, u in dlv]

    # transfer legs: first leg = sender's actual cost, legs in between = liquidity providers
    tr = tr.assign(idx=tr.TRANSFER_INDEX.astype("int64"),
                   asset=_asset_key(tr.TOKEN_ADDRESS.to_numpy(), tr.SYMBOL.to_numpy()))
    tr = tr.sort_values(["TRANSACTION_HASH", "idx"])
    g = tr.groupby("TRANSACTION_HASH", sort=False)
    first = g.first()
    legs = g.size().rename("n_legs")
    lps = g.TO_ADDRESS.apply(lambda s: "|".join(s.iloc[:-1])).rename("lps")
    p = p.join(first[["asset", "VALUE"]].rename(columns={"asset": "src_asset", "VALUE": "src_units"}), on="hash")
    p = p.join(legs, on="hash").join(lps, on="hash")
    p["is_conversion"] = p.n_legs > 1

    p["dst_usd"] = _usd(rates, p.ts, p.dst_asset, p.dst_units)
    p["src_usd"] = _usd(rates, p.ts, p.src_asset, p.src_units)
    # what the sender paid over the reference value delivered (spread + rounding)
    p["conv_cost"] = np.where(p.dst_usd > 0, p.src_usd / p.dst_usd - 1, 0.0)

    if with_labels and (d / "labels.csv").exists():
        lab = pd.read_csv(d / "labels.csv").rename(columns={"TRANSACTION_HASH": "hash"})
        p = p.merge(lab, on="hash", how="left")

    p = p.sort_values(["ts", "ledger", "tx_index", "hash"], kind="mergesort").reset_index(drop=True)
    return p


if __name__ == "__main__":
    import sys
    for s in sys.argv[1:] or SPLITS:
        t = load_split(s)
        print(s, t.shape, "nan usd:", int(t.dst_usd.isna().sum()), int(t.src_usd.isna().sum()))
        print(t.head(3).T)
