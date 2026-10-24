"""Transaction-graph neural network: node classification on the bundle written by src.gnn_export.

Self-contained (numpy, pandas, scikit-learn, torch; no torch_geometric, no project imports), so the
same file runs in Colab (colab/gnn_colab.ipynb embeds it) and locally:

    python gnn.py --bundle colab_bundle --out gnn_out                 # grouped CV + test (GPU)
    python gnn.py --bundle colab_bundle --out gnn_out --set hidden=128 handoff_window=3600
    python gnn.py --bundle colab_bundle --out /tmp/smoke --smoke      # 3,000 nodes, 2 epochs, CPU
    python gnn.py --bundle colab_bundle --out /tmp/bench --bench      # seconds per epoch
    python gnn.py --bundle colab_bundle --blend gnn_oof_train.csv stage1   # AP of rank blends
    python -m src.gnn ...                                             # same, from the repo root

Inputs (see src/gnn_export.py): per split x (node features, already scaled on train statistics),
directed edges j -> i of 8 types (shared-wallet role pair x before/after) with signed time gap,
nearness rank, log value ratio and flags. train_meta.npz holds y (target only), fold (the outer
folds of src.evaluate), group (for grouped inner splits) and difficulty / relay codes (metrics
only). Nothing from train_meta ever enters the model's inputs.

Optional residual stage 2 (Config.stage1, stage1.npz): LightGBM stage-1 scores are appended to the
node features (so message passing shares neighbours' scores, as src/stack.py does by hand) AND the
output is logit = a * logit(s1) + b + correction, with the correction head zero-initialised: training
starts exactly at stage 1 and early stopping keeps only graph corrections that help the inner fold.
Leakage control as in src/stack.py. In outer fold k every train node gets the
score from stage1.npz `fold[k]`: out-of-fold within the other four folds for their nodes, and a
model trained on those four folds for fold k's nodes, so no label of fold k reaches any input.
Test-time models use `train_full` (OOF over all of train) and `test` (model on all of train).

Model: input MLP -> L relational message-passing layers -> jumping-knowledge MLP head.
  message   m_ji = GELU(A h_j + B h_i + C e_ji)          (e = type one-hot + edge attributes)
  aggregate per edge type: attention-weighted sum (softmax over i's neighbours of that type)
            and elementwise max, plus log in-degree per type
  update    h_i <- LayerNorm(h_i + MLP([h_i, typed attention sums, typed maxes, log degrees]))
Training: full batch, BCE with pos_weight, AdamW, DropEdge + input-feature dropout, EMA weights,
early stopping on AP of a grouped inner validation fold carved from the outer training folds.
Outputs: gnn_oof_train.csv (out-of-fold, seed-averaged), gnn_test.csv (models trained on all of
train for the median early-stopping epoch, seed-averaged), gnn_test_foldavg.csv (average of the
CV fold models on test), gnn_report.json, gnn_log.txt.
"""
import argparse
import copy
import dataclasses
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

N_TYPES = 8                      # edge types 0-3 are hand-offs, 4-7 same-wallet (see gnn_export)
FLAG_BITS = (1, 2, 4)            # amount_match, same_asset, closing_pair
DIFFICULTY = {1: "easy", 2: "medium", 3: "hard"}


@dataclass
class Config:
    # graph: edges are filtered down from the exported (widest) windows at load time
    handoff_window: int = 6 * 3600   # seconds, edge types 0-3
    same_window: int = 3600          # seconds, edge types 4-7
    k: int = 8                       # nearest neighbours kept per node and type (<= exported k)
    stage1: bool = True              # residual stage 2 on nested-OOF LightGBM scores (stage1.npz)
    # model
    hidden: int = 96
    layers: int = 3
    use_max: bool = True             # typed max aggregation next to typed attention
    dropout: float = 0.3
    feat_drop: float = 0.1           # dropout on the input features
    edge_drop: float = 0.2           # DropEdge rate during training
    # optimisation
    lr: float = 2e-3
    weight_decay: float = 1e-3
    pos_weight: float = 5.0          # train prevalence is 3.1% (neg/pos = 31); sqrt-ish weighting
    ema: float = 0.99                # EMA of weights, used for validation and prediction
    grad_clip: float = 1.0
    max_epochs: int = 800            # full-batch: one optimiser step per epoch
    min_epochs: int = 60
    eval_every: int = 5
    patience: int = 120              # epochs without inner-val AP improvement
    # protocol
    seeds: tuple = (0, 1, 2)         # per outer fold (each with its own inner split)
    test_seeds: tuple = (0, 1, 2, 3, 4)
    inner_splits: int = 5            # inner val = 1 of this many grouped folds of the outer train
    epoch_scale: float = 1.0         # test models train for median(best epochs) * this
    test_from_folds: bool = True     # also write gnn_test_foldavg.csv from the CV fold models
    device: str = "auto"


def _device(cfg):
    if cfg.device != "auto":
        return torch.device(cfg.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Log:
    def __init__(self, path):
        self.path = Path(path)

    def __call__(self, *msg):
        line = " ".join(str(m) for m in msg)
        print(line, flush=True)
        with open(self.path, "a") as f:
            f.write(line + "\n")


# ---------------------------------------------------------------- data

class Graph:
    """One split as tensors on `device`. nodes: optional subset of row positions (smoke tests)."""

    def __init__(self, path, cfg, device, nodes=None):
        z = np.load(path)
        x = z["x"].astype(np.float32)
        ei = z["edge_index"].astype(np.int64)
        et = z["edge_type"].astype(np.int64)
        dt = z["edge_dt"].astype(np.float32)
        rank = z["edge_rank"].astype(np.int64)
        lr = z["edge_logratio"].astype(np.float32)
        flags = z["edge_flags"].astype(np.int64)
        self.hash = z["hash"].astype(str)
        win = np.where(et < 4, cfg.handoff_window, cfg.same_window).astype(np.float32)
        keep = (np.abs(dt) <= win) & (rank < cfg.k)
        if nodes is not None:
            nodes = np.asarray(nodes)
            inside = np.zeros(len(x), bool)
            inside[nodes] = True
            remap = np.full(len(x), -1, dtype=np.int64)
            remap[nodes] = np.arange(len(nodes))
            keep &= inside[ei[0]] & inside[ei[1]]
            ei = remap[ei]
            x, self.hash = x[nodes], self.hash[nodes]
        ei, et, dt, rank, lr, flags, win = (a[..., keep] for a in (ei, et, dt, rank, lr, flags, win))
        attr = np.column_stack([
            np.eye(N_TYPES, dtype=np.float32)[et],
            np.sign(dt) * np.log1p(np.abs(dt)) / np.log1p(86400.0),   # signed log time gap
            np.abs(dt) / win,                                          # gap as a fraction of the window
            rank / 8.0,                                                # nearness rank (0 = nearest)
            np.clip(lr, -20, 20) / 5.0,                                # log(v_j / v_i), USD delivered
            np.minimum(np.abs(lr), 10) / 5.0,
            *[(flags & b) > 0 for b in FLAG_BITS],                     # amount match, same asset, closing pair
        ]).astype(np.float32)
        self.nodes = nodes
        self.n = len(x)
        self.n_edges = len(et)
        self.edge_dim = attr.shape[1]
        self.x_base = torch.from_numpy(x).to(device)
        self.x = self.x_base
        self.s1_logit = None
        self.src = torch.from_numpy(ei[0]).to(device)
        self.dst = torch.from_numpy(ei[1]).to(device)
        self.slot = torch.from_numpy(ei[1] * N_TYPES + et).to(device)    # (receiving node, type)
        self.eattr = torch.from_numpy(attr).to(device)
        self.type_counts = np.bincount(et, minlength=N_TYPES).tolist()

    def set_stage1(self, s):
        """Append stage-1 scores (aligned with the full split) as two input columns."""
        if self.nodes is not None:
            s = s[self.nodes]
        s = np.clip(np.asarray(s, dtype=np.float64), 1e-6, 1 - 1e-6)
        logit = np.log(s / (1 - s))
        cols = np.column_stack([logit / 4.0, s]).astype(np.float32)
        dev = self.x_base.device
        self.x = torch.cat([self.x_base, torch.from_numpy(cols).to(dev)], 1)
        self.s1_logit = torch.from_numpy(logit.astype(np.float32)).to(dev)


def load_stage1(bundle, cfg):
    if not cfg.stage1:
        return None
    path = Path(bundle) / "stage1.npz"
    if not path.exists():
        raise SystemExit(f"{path} missing: re-export with stage 1 or set stage1=False")
    z = np.load(path)
    return {k: z[k] for k in ("fold", "train_full", "test", "oof")}


def load_meta(bundle, nodes=None):
    z = np.load(Path(bundle) / "train_meta.npz")
    meta = {k: z[k] for k in ("y", "fold", "group", "difficulty", "relay")}
    if nodes is not None:
        meta = {k: v[nodes] for k, v in meta.items()}
    return meta


# ---------------------------------------------------------------- model

class Layer(nn.Module):
    def __init__(self, h, edge_dim, dropout, use_max):
        super().__init__()
        self.h, self.use_max = h, use_max
        self.w_src = nn.Linear(h, h)
        self.w_dst = nn.Linear(h, h, bias=False)
        self.w_edge = nn.Linear(edge_dim, h, bias=False)
        self.att = nn.Linear(h, 1, bias=False)
        n_in = h + N_TYPES * h * (2 if use_max else 1) + N_TYPES
        self.update = nn.Sequential(nn.Linear(n_in, 2 * h), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * h, h))
        self.norm = nn.LayerNorm(h)

    def forward(self, h, src, dst, slot, slot_h, eattr, log_deg):
        n, H = h.size(0), self.h
        m = F.gelu(self.w_src(h)[src] + self.w_dst(h)[dst] + self.w_edge(eattr))       # (E, H)
        # attention: softmax of the edge logits within each (receiving node, type) slot
        a = self.att(m).squeeze(-1)
        a_max = torch.full((n * N_TYPES,), float("-inf"), device=h.device).scatter_reduce(
            0, slot, a.detach(), "amax", include_self=True)
        w = torch.exp(a - a_max[slot])
        den = torch.zeros(n * N_TYPES, device=h.device).index_add(0, slot, w)
        w = w / den[slot].clamp_min(1e-12)
        parts = [h, torch.zeros(n * N_TYPES, H, device=h.device).index_add(0, slot, m * w[:, None]).view(n, -1)]
        if self.use_max:
            mx = torch.zeros(n * N_TYPES, H, device=h.device).scatter_reduce(
                0, slot_h, m, "amax", include_self=False)
            parts.append(mx.view(n, -1))
        parts.append(log_deg)
        return self.norm(h + self.update(torch.cat(parts, 1)))


class Net(nn.Module):
    def __init__(self, n_feat, edge_dim, cfg):
        super().__init__()
        H = cfg.hidden
        self.feat_drop = nn.Dropout(cfg.feat_drop)
        self.enc = nn.Sequential(nn.Linear(n_feat, H), nn.LayerNorm(H), nn.GELU(), nn.Dropout(cfg.dropout),
                                 nn.Linear(H, H), nn.LayerNorm(H))
        self.layers = nn.ModuleList([Layer(H, edge_dim, cfg.dropout, cfg.use_max) for _ in range(cfg.layers)])
        self.head = nn.Sequential(nn.Linear(H * (cfg.layers + 1), H), nn.GELU(), nn.Dropout(cfg.dropout),
                                  nn.Linear(H, 1))
        self.residual = cfg.stage1
        if self.residual:                     # start exactly at the stage-1 logit
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)
            self.s1_scale = nn.Parameter(torch.tensor(1.0))
            self.s1_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, g, keep=None):
        src, dst, slot, ea = g.src, g.dst, g.slot, g.eattr
        if keep is not None:
            src, dst, slot, ea = src[keep], dst[keep], slot[keep], ea[keep]
        H = self.enc[0].out_features
        deg = torch.zeros(g.n * N_TYPES, device=src.device).index_add(
            0, slot, torch.ones_like(slot, dtype=torch.float32))
        log_deg = torch.log1p(deg).view(g.n, N_TYPES) / 2.0
        slot_h = slot[:, None].expand(-1, H)
        h = self.enc(self.feat_drop(g.x))
        hs = [h]
        for layer in self.layers:
            h = layer(h, src, dst, slot, slot_h, ea, log_deg)
            hs.append(h)
        out = self.head(torch.cat(hs, 1)).squeeze(-1)
        if self.residual:
            out = out + self.s1_scale * g.s1_logit + self.s1_bias
        return out


@torch.no_grad()
def predict(model, g):
    model.eval()
    return torch.sigmoid(model(g)).float().cpu().numpy()


@torch.no_grad()
def _ema_update(ema, model, decay):
    for pe, pm in zip(ema.parameters(), model.parameters()):
        pe.mul_(decay).add_(pm.detach(), alpha=1 - decay)


def fit(g, y, train_rows, val_rows, cfg, seed, epochs=None, log=print):
    """Train on train_rows (loss only there; the whole graph is visible as unlabelled context).
    With val_rows: early stopping on their AP, returns the best EMA model. Without: `epochs` steps."""
    torch.manual_seed(seed)
    dev = g.x.device
    model = Net(g.x.shape[1], g.edge_dim, cfg).to(dev)
    ema = copy.deepcopy(model)
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    yt = torch.as_tensor(y, dtype=torch.float32, device=dev)
    tr = torch.as_tensor(train_rows, device=dev)
    pw = torch.tensor(cfg.pos_weight, device=dev)
    can_validate = val_rows is not None and y[val_rows].sum() > 0
    n_epochs = epochs if epochs is not None else cfg.max_epochs
    best_ap, best_ep, best_state = -1.0, n_epochs, None
    for ep in range(1, n_epochs + 1):
        model.train()
        keep = (torch.rand(g.n_edges, device=dev) >= cfg.edge_drop) if cfg.edge_drop > 0 else None
        loss = F.binary_cross_entropy_with_logits(model(g, keep)[tr], yt[tr], pos_weight=pw)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        _ema_update(ema, model, min(cfg.ema, (1 + ep) / (10 + ep)))
        if can_validate and (ep % cfg.eval_every == 0 or ep == n_epochs):
            ap = average_precision_score(y[val_rows], predict(ema, g)[val_rows])
            if ap > best_ap:
                best_ap, best_ep = ap, ep
                best_state = {k: v.detach().clone() for k, v in ema.state_dict().items()}
            elif ep >= cfg.min_epochs and ep - best_ep >= cfg.patience:
                break
    if best_state is not None:
        ema.load_state_dict(best_state)
    return ema, best_ep, best_ap


def inner_split(rows, y, group, seed, n_splits):
    """One grouped, stratified fold of `rows` as the early-stopping set."""
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=1000 + seed)
    a, b = next(skf.split(np.zeros(len(rows)), y[rows], group[rows]))
    return rows[a], rows[b]


# ---------------------------------------------------------------- metrics and output

def _ap(y, s):
    return 100 * average_precision_score(y, s) if 0 < y.sum() < len(y) else float("nan")


def metrics(s, meta, hashes):
    """Same definitions as src.evaluate.metrics."""
    y, diff, relay = meta["y"], meta["difficulty"], meta["relay"].astype(bool)
    out = {"AP": _ap(y, s)}
    k = max(1, int(round(0.01 * len(y))))
    top = np.lexsort((hashes, -s))[:k]
    out["top1%_precision"] = float(y[top].mean())
    out["top1%_recall"] = float(y[top].sum() / max(y.sum(), 1))
    neg = y == 0
    for code, name in DIFFICULTY.items():
        m = ((diff == code) & (y == 1)) | neg
        out[f"AP_{name}"] = _ap(y[m], s[m])
    out["AP_relay_cohorts"] = _ap(y[relay], s[relay]) if relay.any() else float("nan")
    return out


def write_scores(path, hashes, s):
    s = np.asarray(s, dtype=np.float64)
    assert len(s) == len(hashes) == len(set(hashes)) and np.isfinite(s).all() and (s >= 0).all() and (s <= 1).all()
    pd.DataFrame({"TRANSACTION_HASH": hashes, "score": s}).to_csv(path, index=False, float_format="%.8f")


def _fmt(m):
    return "  ".join(f"{k}={v:.3f}" if abs(v) < 1.5 else f"{k}={v:.1f}" for k, v in m.items())


# ---------------------------------------------------------------- protocols

def run_cv(bundle, out, cfg, nodes=None, test_nodes=None):
    """Out-of-fold scores on train with the bundle's outer folds; optional fold-model test average."""
    bundle, out = Path(bundle), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    log = Log(out / "gnn_log.txt")
    dev = _device(cfg)
    t0 = time.time()
    g = Graph(bundle / "train_graph.npz", cfg, dev, nodes)
    meta = load_meta(bundle, nodes)
    gt = Graph(bundle / "test_graph.npz", cfg, dev, test_nodes) if cfg.test_from_folds else None
    s1 = load_stage1(bundle, cfg)
    if s1 is not None:
        g.set_stage1(s1["oof"])                       # placeholder until the fold loop sets fold[k]
        if gt is not None:
            gt.set_stage1(s1["test"])
    y, fold, group = meta["y"], meta["fold"], meta["group"]
    log(f"[cv] device={dev} torch={torch.__version__} nodes={g.n} edges={g.n_edges} "
        f"per_type={g.type_counts} features={g.x.shape[1]} edge_dim={g.edge_dim} positives={int(y.sum())}")
    log(f"[cv] config {dataclasses.asdict(cfg)}")
    oof = np.zeros(g.n)
    test_sum = np.zeros(gt.n) if gt is not None else None
    best_eps, fold_rows = [], []
    for k in np.unique(fold):
        te, tr = np.where(fold == k)[0], np.where(fold != k)[0]
        if s1 is not None:
            g.set_stage1(s1["fold"][int(k)])          # no label of fold k behind any input
        s_te = np.zeros(len(te))
        for seed in cfg.seeds:
            t1 = time.time()
            a, b = inner_split(tr, y, group, seed, cfg.inner_splits)
            model, ep, ap_val = fit(g, y, a, b, cfg, seed=100 * seed + int(k))
            s = predict(model, g)
            s_te += s[te]
            if gt is not None:
                test_sum += predict(model, gt)
            best_eps.append(ep)
            log(f"[cv] fold {k} seed {seed}: best epoch {ep}, inner-val AP {100 * ap_val:.1f}, "
                f"fold AP (this seed) {_ap(y[te], s[te]):.1f}, {time.time() - t1:.0f}s")
            del model
        oof[te] = s_te / len(cfg.seeds)
        fold_rows.append({"fold": int(k), "AP": _ap(y[te], oof[te]), "positives": int(y[te].sum())})
        log(f"[cv] fold {k}: AP {fold_rows[-1]['AP']:.1f} ({fold_rows[-1]['positives']} positives)")
    m = metrics(oof, meta, g.hash)
    if s1 is not None:
        s1_oof = s1["oof"] if nodes is None else s1["oof"][nodes]
        m1 = metrics(s1_oof, meta, g.hash)
        log(f"[cv] stage-1 LightGBM OOF on the same nodes: AP {m1['AP']:.1f}")
    final_epochs = max(1, int(round(np.median(best_eps) * cfg.epoch_scale)))
    log(f"[cv] OOF {_fmt(m)}")
    log(f"[cv] best epochs {best_eps} -> test models train {final_epochs} epochs; {time.time() - t0:.0f}s total")
    write_scores(out / "gnn_oof_train.csv", g.hash, oof)
    if gt is not None:
        write_scores(out / "gnn_test_foldavg.csv", gt.hash, test_sum / len(best_eps))
    rep = {"config": dataclasses.asdict(cfg), "metrics": m, "folds": fold_rows, "best_epochs": best_eps,
           "final_epochs": final_epochs, "seconds": time.time() - t0, "device": str(dev),
           "torch": torch.__version__, "train_edges": g.n_edges, "train_edges_per_type": g.type_counts}
    (out / "gnn_report.json").write_text(json.dumps(rep, indent=1, default=float))
    return rep


def run_test(bundle, out, cfg, epochs, nodes=None, test_nodes=None):
    """Train on all of train for `epochs` steps per seed, average the test scores."""
    bundle, out = Path(bundle), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    log = Log(out / "gnn_log.txt")
    dev = _device(cfg)
    t0 = time.time()
    g = Graph(bundle / "train_graph.npz", cfg, dev, nodes)
    gt = Graph(bundle / "test_graph.npz", cfg, dev, test_nodes)
    y = load_meta(bundle, nodes)["y"]
    s1 = load_stage1(bundle, cfg)
    if s1 is not None:
        g.set_stage1(s1["train_full"])
        gt.set_stage1(s1["test"])
    s = np.zeros(gt.n)
    for seed in cfg.test_seeds:
        model, _, _ = fit(g, y, np.arange(g.n), None, cfg, seed=7000 + seed, epochs=epochs)
        s += predict(model, gt)
        log(f"[test] seed {seed}: {epochs} epochs, {time.time() - t0:.0f}s so far")
        del model
    s /= len(cfg.test_seeds)
    write_scores(out / "gnn_test.csv", gt.hash, s)
    log(f"[test] wrote {out / 'gnn_test.csv'} ({gt.n} rows, {gt.n_edges} test edges), {time.time() - t0:.0f}s")
    rep_path = out / "gnn_report.json"
    rep = json.loads(rep_path.read_text()) if rep_path.exists() else {}
    rep["test"] = {"epochs": epochs, "seeds": list(cfg.test_seeds), "seconds": time.time() - t0,
                   "test_edges": gt.n_edges, "config": dataclasses.asdict(cfg)}
    rep_path.write_text(json.dumps(rep, indent=1, default=float))
    return s


def bench(bundle, cfg, nodes=None, epochs=5):
    """Seconds per training epoch (forward + backward) and peak GPU memory on the train graph."""
    dev = _device(cfg)
    g = Graph(Path(bundle) / "train_graph.npz", cfg, dev, nodes)
    s1 = load_stage1(bundle, cfg)
    if s1 is not None:
        g.set_stage1(s1["train_full"])
    y = load_meta(bundle, nodes)["y"]
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    fit(g, y, np.arange(g.n), None, cfg, seed=0, epochs=1)              # warm-up
    t = time.time()
    fit(g, y, np.arange(g.n), None, cfg, seed=0, epochs=epochs)
    sec = (time.time() - t) / epochs
    mem = torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else float("nan")
    print(f"[bench] device={dev} nodes={g.n} edges={g.n_edges} {sec:.3f}s/epoch peak_gpu={mem:.2f}GB")
    return sec


def _read_scores(path, hashes):
    """Scores aligned with `hashes` (NaN where the CSV has no row)."""
    df = pd.read_csv(path)
    return df.set_index("TRANSACTION_HASH").score.reindex(hashes).to_numpy(float)


def blend_report(bundle, a_path, b_path="stage1", weights=(0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)):
    """OOF AP of rank blends (1 - w) * rank(b) + w * rank(a) on train. b = a CSV or 'stage1'.
    Only train rows scored by both are used (all of them for real runs; a subset for smoke runs)."""
    bundle = Path(bundle)
    hashes = np.load(bundle / "train_graph.npz")["hash"].astype(str)
    meta = load_meta(bundle)
    a = _read_scores(a_path, hashes)
    b = np.load(bundle / "stage1.npz")["oof"].astype(float) if b_path == "stage1" else _read_scores(b_path, hashes)
    rows = np.where(np.isfinite(a) & np.isfinite(b))[0]
    if len(rows) < len(hashes):
        print(f"[blend] WARNING: only {len(rows)} of {len(hashes)} train rows scored by both; AP on those")
    a, b, hashes = a[rows], b[rows], hashes[rows]
    meta = {k: v[rows] for k, v in meta.items()}
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    print(f"[blend] a={a_path} b={b_path}  rank corr {np.corrcoef(ra, rb)[0, 1]:.3f}")
    rows = []
    for w in weights:
        m = metrics((1 - w) * rb + w * ra, meta, hashes)
        rows.append({"w_a": w, **m})
        print(f"[blend] w_a={w:.1f}  {_fmt(m)}")
    return rows


def _parse_set(items):
    """["hidden=128", "seeds=[0,1]", "use_max=false"] -> typed Config overrides."""
    defaults, out = Config(), {}
    for it in items or []:
        k, v = it.split("=", 1)
        if not hasattr(defaults, k):
            raise SystemExit(f"unknown config field {k}")
        d = getattr(defaults, k)
        if isinstance(d, bool):
            out[k] = v.lower() in ("1", "true", "yes")
        elif isinstance(d, tuple):
            out[k] = tuple(json.loads(v))
        else:
            out[k] = type(d)(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out")
    ap.add_argument("--blend", nargs="+", metavar="CSV", help="A.csv [B.csv|stage1]: rank-blend AP table")
    ap.add_argument("--set", nargs="*", help="config overrides, e.g. hidden=128 seeds=[0,1]")
    ap.add_argument("--nodes", type=int, default=0, help="use only the first N nodes of each split")
    ap.add_argument("--smoke", action="store_true", help="CPU, 3,000 nodes, 2 epochs, 1 seed")
    ap.add_argument("--bench", action="store_true", help="time a few training epochs and exit")
    ap.add_argument("--skip-test", action="store_true")
    a = ap.parse_args()
    if a.blend:
        blend_report(a.bundle, *a.blend[:2])
        return
    if not a.out:
        raise SystemExit("--out is required")
    over = _parse_set(a.set)
    n_nodes = a.nodes
    if a.smoke:
        over = {"device": "cpu", "hidden": 32, "max_epochs": 2, "min_epochs": 0, "eval_every": 1,
                "seeds": (0,), "test_seeds": (0,), **over}
        n_nodes = n_nodes or 3000
    cfg = dataclasses.replace(Config(), **over)
    nodes = np.arange(n_nodes) if n_nodes else None
    if a.bench:
        bench(a.bundle, cfg, nodes)
        return
    rep = run_cv(a.bundle, a.out, cfg, nodes, nodes)
    if not a.skip_test:
        run_test(a.bundle, a.out, cfg, rep["final_epochs"], nodes, nodes)


if __name__ == "__main__":
    main()
