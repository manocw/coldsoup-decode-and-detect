"""Export the transaction graph + node features for the GNN (src/gnn.py, colab/gnn_colab.ipynb).

    ./venv/bin/python -m src.gnn_export                  # -> colab_bundle/, colab_bundle.zip, notebook
    ./venv/bin/python -m src.gnn_export --notebook-only  # only regenerate colab/gnn_colab.ipynb

Nodes are the transactions of one split (row order = src.load.load_split order, i.e. time order).
A directed edge j -> i carries j's state to i when the two payments share a wallet close in time.
The 8 edge types are (role of the shared wallet in i, role in j, j before/after i):

  0 funded_by      j paid i's sender, before i       (j.receiver == i.sender)   window: handoff
  1 refill_after   j paid i's sender, after i        (j.receiver == i.sender)   window: handoff
  2 pays_on        i's receiver pays j, after i      (j.sender == i.receiver)   window: handoff
  3 paid_before    i's receiver paid j, before i     (j.sender == i.receiver)   window: handoff
  4 sender_prev / 5 sender_next       same sender, j before / after i           window: same
  6 receiver_prev / 7 receiver_next   same receiver, j before / after i         window: same

"Before/after" is strict row order (same-ledger payments are ordered by tx_index). Per receiving
node and type only the k nearest in time are kept, so busy wallets cannot blow up the graph. The
windows exported here are the widest the model may use; src/gnn.py filters edges down to narrower
windows at load time (the raw signed time gap is stored), so window ablations need no re-export.

Edge attributes (behaviour only, never identities): signed time gap (seconds), nearness rank,
log value ratio v_j / v_i (USD delivered), and flags: amounts match within 1%, same asset handed on,
closing pair (the other ends coincide too: round trip for hand-offs, repeated pair otherwise).

Node features: the cached feature families (default: the v3 nine), transformed with statistics
fitted on train only: binary columns -> 0/1, others -> quantile-normal (sklearn), NaN -> 0 plus one
missing indicator per distinct train NaN pattern; plus log1p candidate counts per edge type before
the cap. Stored as float16 (quantile-normal values need no more precision).

Labels and annotation metadata go ONLY into train_meta.npz: y (target), fold (the outer folds of
src.evaluate: StratifiedGroupKFold(5, shuffle=True, random_state=0) over cv_groups), group (for
grouped inner early-stopping splits) and difficulty / relay codes (metric breakdowns only).
Hashes go into *_graph.npz under `hash`, used only to write the output CSVs.

stage1.npz (skip with --no-stage1): LightGBM stage-1 scores on the same features, with the nested
leakage control of src/stack.py (seed 0, same outer folds): fold[k] = OOF inside the other four
folds for their rows and a model fit on them for fold k's rows; train_full = OOF over all of
train; test = model fit on all of train; oof = fold[k] on fold k's rows (= stage-1 CV scores).
36 LightGBM fits: run with XRPL_NJOBS=3 on a shared machine (~5 min).
"""
import argparse
import json
import shutil
import time
import zipfile

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import QuantileTransformer

from src.evaluate import cv_groups, make_model, unlabelled
from src.features import build_features
from src.load import ROOT, load_split
from src.stack import oof_scores

BUNDLE = ROOT / "colab_bundle"
ZIP = ROOT / "colab_bundle.zip"
NOTEBOOK = ROOT / "colab" / "gnn_colab.ipynb"
GNN_SRC = ROOT / "src" / "gnn.py"

V3_FAMILIES = ["starter", "funding", "relays", "fields", "motifs", "hops", "amounts", "history", "cycles"]
EDGE_TYPES = ["funded_by", "refill_after", "pays_on", "paid_before",
              "sender_prev", "sender_next", "receiver_prev", "receiver_next"]
FLAG_BITS = {"amount_match": 1, "same_asset": 2, "closing_pair": 4}
DIFFICULTY_CODES = {"normal": 0, "easy": 1, "medium": 2, "hard": 3}


# ---------------------------------------------------------------- edges

def _index(col):
    """wallet -> row positions (ascending; rows are time-sorted, so times ascend too)."""
    return {a: np.asarray(ix) for a, ix in pd.Series(col).groupby(col, sort=False).indices.items()}


def build_edges(p, handoff=6 * 3600, same=3600, k=8):
    """Directed edges j -> i of the 8 types above, the k nearest per (i, type). Uses wallets only to
    find neighbours; nothing identifying is written into the attributes."""
    n = len(p)
    ts = p.ts.to_numpy(np.int64)
    S, R = p.sender.to_numpy(), p.receiver.to_numpy()
    by_s, by_r = _index(S), _index(R)
    empty = np.empty(0, dtype=np.int64)
    src, dst, typ, rank = [], [], [], []
    counts = np.zeros((n, len(EDGE_TYPES)), dtype=np.int32)
    for i in range(n):
        t = ts[i]
        specs = ((by_r.get(S[i], empty), handoff), (by_s.get(R[i], empty), handoff),
                 (by_s[S[i]], same), (by_r[R[i]], same))
        for pair, (ix, w) in enumerate(specs):
            tix = ts[ix]
            # before: positions < i with t_j >= t - w, nearest first
            lo, hi = np.searchsorted(tix, t - w, "left"), np.searchsorted(ix, i, "left")
            before = ix[lo:hi][::-1]
            # after: positions > i with t_j <= t + w, nearest first
            lo, hi = np.searchsorted(ix, i, "right"), np.searchsorted(tix, t + w, "right")
            after = ix[lo:hi]
            # type order inside a role pair: (before, after) for pairs 0, 2, 3 and (after, before) for 1
            sides = (before, after) if pair != 1 else (after, before)
            for side, nb in enumerate(sides):
                ty = 2 * pair + side
                counts[i, ty] = len(nb)
                nb = nb[:k]
                if len(nb):
                    src.append(nb)
                    dst.append(np.full(len(nb), i, dtype=np.int64))
                    typ.append(np.full(len(nb), ty, dtype=np.int8))
                    rank.append(np.arange(len(nb), dtype=np.uint8))
    src, dst = np.concatenate(src), np.concatenate(dst)
    typ, rank = np.concatenate(typ), np.concatenate(rank)

    dt = (ts[src] - ts[dst]).astype(np.float32)              # signed: j's time minus i's time
    v = p.dst_usd.to_numpy(float)
    ok = (v[src] > 0) & (v[dst] > 0) & np.isfinite(v[src]) & np.isfinite(v[dst])
    lr = np.zeros(len(src))
    lr[ok] = np.log(v[src][ok] / v[dst][ok])
    amount_match = ok & (np.abs(lr) < 0.01)
    lr = np.clip(lr, -20, 20).astype(np.float16)

    dst_a, src_a = p.dst_asset.to_numpy(), p.src_asset.to_numpy()
    cross_in = typ <= 1                                        # j.receiver == i.sender
    cross_out = (typ == 2) | (typ == 3)                        # j.sender == i.receiver
    same_role = typ >= 4
    same_asset = np.where(cross_in, dst_a[src] == src_a[dst],
                          np.where(cross_out, dst_a[dst] == src_a[src], dst_a[src] == dst_a[dst]))
    closing = np.select(
        [cross_in, cross_out, (typ == 4) | (typ == 5)],
        [S[src] == R[dst], R[src] == S[dst], R[src] == R[dst]],
        default=S[src] == S[dst])
    assert same_role.sum() + cross_in.sum() + cross_out.sum() == len(typ)
    flags = (amount_match * FLAG_BITS["amount_match"] + same_asset * FLAG_BITS["same_asset"]
             + closing * FLAG_BITS["closing_pair"]).astype(np.uint8)
    edges = {"edge_index": np.vstack([src, dst]).astype(np.int32), "edge_type": typ,
             "edge_dt": dt, "edge_rank": rank, "edge_logratio": lr, "edge_flags": flags}
    return edges, counts


# ---------------------------------------------------------------- node features

class NodeTransform:
    """Fitted on train only, applied unchanged to every split."""

    def fit(self, X):
        A = X.to_numpy(np.float64)
        nun = X.nunique()
        self.columns = list(X.columns)
        self.binary = [c for c in self.columns if nun[c] <= 2]
        self.cont = [c for c in self.columns if nun[c] > 2]
        b = X[self.binary]
        self.b_lo, self.b_hi = b.min().to_numpy(), b.max().to_numpy()
        self.qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal",
                                      subsample=10**9, random_state=0).fit(X[self.cont].to_numpy())
        # one missing indicator per distinct NaN pattern (many columns go missing together)
        nan = np.isnan(A)
        rate = nan.mean(0)
        self.nan_cols, seen = [], {}
        for j in np.where((rate > 0.005) & (rate < 0.995))[0]:
            key = np.packbits(nan[:, j]).tobytes()
            if key not in seen:
                seen[key] = self.columns[j]
                self.nan_cols.append(self.columns[j])
        return self

    def transform(self, X, counts):
        X = X[self.columns]
        b = X[self.binary].to_numpy(np.float64)
        span = np.where(self.b_hi > self.b_lo, self.b_hi - self.b_lo, 1.0)
        b = np.clip((b - self.b_lo) / span, -1, 2)
        q = self.qt.transform(X[self.cont].to_numpy(np.float64))
        m = np.isnan(X[self.nan_cols].to_numpy(np.float64)).astype(np.float64)
        c = np.log1p(counts.astype(np.float64)) / 2.0
        out = np.nan_to_num(np.hstack([b, q, m, c]), nan=0.0)
        names = ([f"bin:{n}" for n in self.binary] + [f"qn:{n}" for n in self.cont]
                 + [f"nan:{n}" for n in self.nan_cols] + [f"cand:{t}" for t in EDGE_TYPES])
        return out.astype(np.float16), names


# ---------------------------------------------------------------- bundle

def train_meta(p):
    """Targets, folds and metric-breakdown codes. Never node or edge features."""
    y = p.is_suspicious.to_numpy().astype(np.int8)
    groups = cv_groups(p)
    fold = np.full(len(p), -1, dtype=np.int8)
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)   # = src.evaluate, seed 0
    for k, (_, te) in enumerate(skf.split(np.zeros((len(p), 1)), y, groups)):
        fold[te] = k
    assert (fold >= 0).all()
    return {"y": y, "fold": fold, "group": groups.astype(np.int32),
            "difficulty": p.difficulty.map(DIFFICULTY_CODES).fillna(0).to_numpy().astype(np.int8),
            "relay": p.cohort_id.notna().to_numpy().astype(np.int8)}


def stage1_scores(Xtr, y, groups, fold, Xte, seed=0):
    """Nested out-of-fold LightGBM scores (see module docstring); mirrors src.stack.nested_cv."""
    n = len(y)
    by_fold = np.zeros((fold.max() + 1, n), dtype=np.float32)
    for k in range(fold.max() + 1):
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        s = oof_scores(Xtr, y, groups, tr, "lgbm", seed)
        s[te] = make_model("lgbm", seed).fit(Xtr[tr], y[tr]).predict_proba(Xtr[te])[:, 1]
        by_fold[k] = s
        print(f"  stage 1: outer fold {k + 1}/{fold.max() + 1}", flush=True)
    full = oof_scores(Xtr, y, groups, np.arange(n), "lgbm", seed)
    test = make_model("lgbm", seed).fit(Xtr, y).predict_proba(Xte)[:, 1]
    oof = by_fold[fold, np.arange(n)]
    return {"fold": by_fold, "train_full": full.astype(np.float32), "test": test.astype(np.float32),
            "oof": oof.astype(np.float32)}


def export(families, handoff, same, k, with_stage1=True, out=BUNDLE):
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    (out / ".gitignore").write_text("*\n")          # derived competition data: never commit
    summary = {}
    tf = None
    for split in ("train", "test"):
        p = load_split(split)
        X = build_features(split, families, p)        # cached per split; built on unlabelled(p)
        edges, counts = build_edges(unlabelled(p), handoff, same, k)
        if tf is None:
            tf = NodeTransform().fit(X)
        x, names = tf.transform(X, counts)
        np.savez_compressed(out / f"{split}_graph.npz", x=x, hash=np.array(p.hash.tolist(), dtype="S"), **edges)
        if split == "train":
            meta = train_meta(p)
            np.savez_compressed(out / "train_meta.npz", **meta)
            Xtr = X.to_numpy(np.float64)
        else:
            Xte = X[tf.columns].to_numpy(np.float64)
        per_type = np.bincount(edges["edge_type"], minlength=len(EDGE_TYPES))
        summary[split] = {"nodes": len(p), "node_features": x.shape[1], "edges": int(per_type.sum()),
                          "edges_per_type": dict(zip(EDGE_TYPES, per_type.tolist())),
                          "isolated_nodes": int(len(p) - len(np.unique(edges["edge_index"][1])))}
        if split == "train":
            summary[split]["positives"] = int(meta["y"].sum())
        print(f"{split}: {summary[split]}", flush=True)
        del p, X, edges, counts, x
    if with_stage1:
        from sklearn.metrics import average_precision_score
        s1 = stage1_scores(Xtr, meta["y"], meta["group"], meta["fold"], Xte)
        np.savez_compressed(out / "stage1.npz", **s1)
        summary["stage1_oof_AP"] = 100 * average_precision_score(meta["y"], s1["oof"])
        print(f"stage 1 OOF AP {summary['stage1_oof_AP']:.1f}", flush=True)
    elif (out / "stage1.npz").exists():
        (out / "stage1.npz").unlink()
    del Xtr, Xte
    manifest = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "families": families, "handoff_window_s": handoff, "same_window_s": same, "k_nearest": k,
        "edge_types": EDGE_TYPES, "edge_flag_bits": FLAG_BITS, "difficulty_codes": DIFFICULTY_CODES,
        "node_feature_names": names,
        "transform": {"binary": len(tf.binary), "quantile_normal": len(tf.cont),
                      "nan_indicators": len(tf.nan_cols), "candidate_counts": len(EDGE_TYPES),
                      "fitted_on": "train"},
        "folds": "StratifiedGroupKFold(5, shuffle=True, random_state=0) over src.evaluate.cv_groups",
        "stage1": ("LightGBM (src.evaluate defaults, seed 0), nested OOF as src.stack" if with_stage1 else None),
        "summary": summary,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    shutil.copy(GNN_SRC, out / "gnn.py")
    if ZIP.exists():
        ZIP.unlink()
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_STORED) as z:   # npz members are already deflated
        for f in sorted(out.iterdir()):
            if f.name != ".gitignore":
                z.write(f, f"colab_bundle/{f.name}")
    print(f"wrote {out}/ and {ZIP} ({ZIP.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.0f}s")


# ---------------------------------------------------------------- notebook

def _cell(kind, text):
    lines = text.strip("\n").split("\n")
    src = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    c = {"cell_type": kind, "metadata": {}, "source": src}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


NB_INTRO = """
# Decode & Detect: transaction-graph GNN (Colab)

Node classification on the transaction graph of each split (nodes = payments, edges = payments that
share a wallet close in time, 8 directed relation types). For every variant in `VARIANTS` it runs
grouped 5-fold CV on train (same outer folds as `src.evaluate`), trains test models on all of
train, and reports the AP of rank blends with the LightGBM stage-1 OOF scores shipped in the bundle.

Outputs in `/content/gnn_out/<variant>/`: `gnn_oof_train.csv` (out-of-fold train scores),
`gnn_test.csv` (seed-averaged models on all of train), `gnn_test_foldavg.csv` (CV fold models on
test), `gnn_report.json`, `gnn_log.txt`; plus `summary.json` and, for the best variant by OOF AP,
top-level `gnn_oof_train.csv` / `gnn_test.csv`.

**Steps:** Runtime > Change runtime type > **T4 GPU**. Runtime > Run all. When the upload cell
asks, choose `colab_bundle.zip` (or put it on Drive and set `USE_DRIVE = True`). The last cell
downloads `gnn_results.zip`; bring that file back.
"""

NB_SETUP = """
# Dependencies: torch, numpy, pandas and scikit-learn are preinstalled on Colab; this only fills gaps.
import importlib.util, subprocess, sys
for mod, pkg in [("torch", "torch"), ("sklearn", "scikit-learn"), ("pandas", "pandas"), ("numpy", "numpy")]:
    if importlib.util.find_spec(mod) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
import torch
print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only - switch the runtime to a GPU")
"""

NB_UPLOAD = """
# Get the bundle: upload colab_bundle.zip, or mount Drive (USE_DRIVE = True) and read it from there.
import json, os, zipfile
BUNDLE_ZIP = "/content/colab_bundle.zip"
OUT = "/content/gnn_out"
USE_DRIVE = False
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    BUNDLE_ZIP = "/content/drive/MyDrive/colab_bundle.zip"
    OUT = "/content/drive/MyDrive/gnn_out"         # results survive a runtime disconnect
if not os.path.exists(BUNDLE_ZIP):
    from google.colab import files
    uploaded = files.upload()                      # choose colab_bundle.zip
    BUNDLE_ZIP = os.path.join("/content", next(iter(uploaded)))
with zipfile.ZipFile(BUNDLE_ZIP) as z:
    z.extractall("/content")
BUNDLE = "/content/colab_bundle"
manifest = json.load(open(f"{BUNDLE}/manifest.json"))
print(json.dumps(manifest["summary"], indent=1))
"""

NB_CONFIG = """
import dataclasses, importlib, sys
sys.path.insert(0, "/content")
import gnn
importlib.reload(gnn)
cfg = gnn.Config()          # defaults; override fields here, e.g. cfg.hidden = 128
# Variants: each one is a full CV + test run. {} = defaults (stage-1 input, hand-off <= 6h,
# same wallet <= 1h). Delete entries to save time; add your own overrides as new entries.
VARIANTS = {
    "s1_wide": {},
    "s1_narrow": {"handoff_window": 3600, "same_window": 600},
    "pure_wide": {"stage1": False},
}
print(cfg)
"""

NB_RUNTIME = """
**Expected runtime on a T4 (defaults):** about 10 min per variant (5 folds x 3 seeds of
early-stopped full-batch training plus 5 test seeds), so about 30 min for the three variants.
Peak GPU memory is a few GB. Each finished variant is written to disk immediately, so a
disconnect only loses the variant in progress (use `USE_DRIVE = True` to keep files across sessions).
"""

NB_RUN = """
import shutil
os.makedirs(OUT, exist_ok=True)
summary = []
for name, over in VARIANTS.items():
    c = dataclasses.replace(cfg, **over)
    d = f"{OUT}/{name}"
    rep = gnn.run_cv(BUNDLE, d, c)
    gnn.run_test(BUNDLE, d, c, epochs=rep["final_epochs"])
    row = {"variant": name, "override": over, "final_epochs": rep["final_epochs"], **rep["metrics"]}
    if os.path.exists(f"{BUNDLE}/stage1.npz"):
        blends = gnn.blend_report(BUNDLE, f"{d}/gnn_oof_train.csv", "stage1")
        best = max(blends, key=lambda r: r["AP"])
        row.update(blend_with_stage1_best_w=best["w_a"], blend_with_stage1_best_AP=best["AP"],
                   stage1_AP=blends[0]["AP"])
    summary.append(row)
    json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1, default=float)
    print(f"== {name}: OOF AP {row['AP']:.1f}", flush=True)
best = max(summary, key=lambda r: r["AP"])
for f in ("gnn_oof_train.csv", "gnn_test.csv"):
    shutil.copy(f"{OUT}/{best['variant']}/{f}", f"{OUT}/{f}")
print("best variant:", best["variant"])
print(json.dumps(summary, indent=1, default=float))
"""

NB_DOWNLOAD = """
# Bring back gnn_results.zip (all variants' CSVs, reports and logs, plus summary.json).
import shutil
from google.colab import files
shutil.make_archive("/content/gnn_results", "zip", OUT)
files.download("/content/gnn_results.zip")
"""


def write_notebook():
    code = GNN_SRC.read_text()
    cells = [_cell("markdown", NB_INTRO), _cell("code", NB_SETUP), _cell("code", NB_UPLOAD),
             _cell("markdown", "Model code (identical to `src/gnn.py` in the repo; written to `/content/gnn.py`)."),
             _cell("code", "%%writefile /content/gnn.py\n" + code), _cell("code", NB_CONFIG),
             _cell("markdown", NB_RUNTIME), _cell("code", NB_RUN), _cell("code", NB_DOWNLOAD)]
    nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                       "kernelspec": {"display_name": "Python 3", "name": "python3"},
                       "language_info": {"name": "python"}}}
    NOTEBOOK.parent.mkdir(exist_ok=True)
    NOTEBOOK.write_text(json.dumps(nb, indent=1))
    print(f"wrote {NOTEBOOK}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", nargs="+", default=V3_FAMILIES)
    ap.add_argument("--handoff", type=int, default=6 * 3600, help="widest hand-off window, seconds")
    ap.add_argument("--same", type=int, default=3600, help="widest same-wallet window, seconds")
    ap.add_argument("--k", type=int, default=8, help="nearest neighbours kept per node and edge type")
    ap.add_argument("--no-stage1", action="store_true", help="skip the LightGBM stage-1 scores")
    ap.add_argument("--notebook-only", action="store_true")
    a = ap.parse_args()
    write_notebook()
    if not a.notebook_only:
        export(a.features, a.handoff, a.same, a.k, with_stage1=not a.no_stage1)


if __name__ == "__main__":
    main()
