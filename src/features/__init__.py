"""Feature families. Each module exposes build(p) -> DataFrame row-aligned with p.

p is the unlabelled per-transaction table of ONE split (src.load.load_split minus label
columns). Features may use the whole week of that split (offline investigation is allowed),
but never labels, split identity or specific wallet/issuer addresses (they differ per split).

build_features caches each family per split under cache/, keyed by the module's source hash.
"""
import hashlib
import importlib
import inspect

import pandas as pd

from src.evaluate import unlabelled
from src.load import ROOT, load_split

CACHE = ROOT / "cache"


def _module(name):
    return importlib.import_module(f"src.features.{name}")


def _source_at_import(mod):
    """Source as it was when this process imported the module (an edit on disk afterwards must
    not relabel features computed by the old code)."""
    if not hasattr(mod, "_source_at_import"):
        mod._source_at_import = inspect.getsource(mod)
    return mod._source_at_import


def _src_hash(mod):
    deps = [_source_at_import(mod)] + [_source_at_import(importlib.import_module(d))
                                       for d in getattr(mod, "DEPENDS_ON", [])]
    return hashlib.sha1("".join(deps).encode()).hexdigest()[:12]


def family(name, split, p=None, use_cache=True):
    mod = _module(name)
    path = CACHE / split / f"{name}-{_src_hash(mod)}.parquet"
    if use_cache and path.exists():
        return pd.read_parquet(path)
    if p is None:
        p = load_split(split)
    f = mod.build(unlabelled(p))
    assert len(f) == len(p), f"{name}: {len(f)} rows for {len(p)} transactions"
    f = f.reset_index(drop=True).astype(float)
    f.columns = [f"{name}.{c}" for c in f.columns]
    path.parent.mkdir(parents=True, exist_ok=True)
    for old in path.parent.glob(f"{name}-*.parquet"):
        old.unlink()
    f.to_parquet(path)
    return f


def build_features(split, names, p=None, use_cache=True):
    if p is None:
        p = load_split(split)
    return pd.concat([family(n, split, p, use_cache) for n in names], axis=1)
