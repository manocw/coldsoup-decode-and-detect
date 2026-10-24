"""Build a local HTML viewer of planted scenarios (train) for manual inspection.

    ./venv/bin/python -m src.explore [--oof path.npy]   -> explore/scenarios.html, explore/train_annotated.csv

Per family: medium suspicious, a few easy suspicious and legitimate look-alikes side by side, step by
step; hard relay scenarios next to legitimate siblings of the same cohort. Local only (organiser data).
"""
import argparse
import html

import numpy as np
import pandas as pd

from src.load import ROOT, load_split

OUT = ROOT / "explore"


def annotate(p, oof=None):
    pair = p.sender + ">" + p.receiver
    p = p.assign(pair_n=pair.map(pair.value_counts()),
                 rev_n=(p.receiver + ">" + p.sender).map(pair.value_counts()).fillna(0).astype(int),
                 s_deg=p.sender.map(pd.concat([p.sender, p.receiver]).value_counts()),
                 r_deg=p.receiver.map(pd.concat([p.sender, p.receiver]).value_counts()),
                 hour_of_week=(p.ts - p.ts.min()) / 3600)
    ids = {w: f"W{i:03d}" for i, w in enumerate(pd.unique(pd.concat([p.sender, p.receiver])))}
    p["w_from"], p["w_to"] = p.sender.map(ids), p.receiver.map(ids)
    short = lambda a: "XRP" if a == "XRP" else a.split(":")[1] + "@" + a.split(":")[0][1:4]
    p["asset"] = [f"{short(s)}→{short(d)}" if s != d else short(d) for s, d in zip(p.src_asset, p.dst_asset)]
    if oof is not None:
        p["score"] = oof
        p["rank"] = p.score.rank(ascending=False).astype(int)
    return p


def table(d, title, cls):
    d = d.sort_values("ts")
    t0 = d.ts.iloc[0]
    rows = []
    prev = None
    for r in d.itertuples():
        gap = "" if prev is None else f"{(r.ts - prev) / 60:.1f}"
        prev = r.ts
        rank = f"{r.rank:,}" if hasattr(r, "rank") else ""
        hot = " hot" if hasattr(r, "rank") and r.rank <= 1250 else ""
        rows.append(f"<tr class='{hot}'><td>{(r.ts - t0) / 60:.1f}</td><td>{gap}</td><td>{r.w_from}</td>"
                    f"<td>{r.w_to}</td><td class=n>{r.dst_usd:,.2f}</td><td>{html.escape(r.asset)}</td>"
                    f"<td class=n>{r.pair_n}</td><td class=n>{r.rev_n}</td><td class=n>{r.s_deg}/{r.r_deg}</td>"
                    f"<td class=n>{rank}</td></tr>")
    span = (d.ts.max() - t0) / 60
    head = ("<tr><th>min</th><th>gap</th><th>from</th><th>to</th><th>USD</th><th>asset</th><th>pair×</th>"
            "<th>rev×</th><th>deg s/r</th><th>rank</th></tr>")
    return (f"<div class='card {cls}'><h4>{html.escape(title)} <span>n={len(d)} · span {span:.0f} min · "
            f"start h{d.hour_of_week.iloc[0]:.1f}</span></h4><table>{head}{''.join(rows)}</table></div>")


def build(oof_path=None):
    p = load_split("train")
    oof = np.load(oof_path) if oof_path else None
    p = annotate(p, oof)
    OUT.mkdir(exist_ok=True)
    cols = ["hash", "ts", "hour_of_week", "w_from", "w_to", "dst_usd", "src_usd", "asset", "n_legs", "pair_n", "rev_n",
            "s_deg", "r_deg", "is_suspicious", "family", "difficulty", "scenario_id", "cohort_id"]
    cols += [c for c in ("score", "rank") if c in p.columns]
    p[cols].to_csv(OUT / "train_annotated.csv", index=False)

    rng = np.random.default_rng(0)
    parts = []
    fams = ["fan_out", "fan_in", "fan_out_in", "bipartite", "chain", "cycle", "stack", "multi_stage", "random"]
    for fam in fams:
        f = p[p.family == fam]
        sc = f.groupby("scenario_id").agg(y=("is_suspicious", "first"), d=("difficulty", "first"))
        med = sc[(sc.y == 1) & (sc.d == "medium")].index
        easy = sc[(sc.y == 1) & (sc.d == "easy")].index[:2]
        look = sc[sc.y == 0].index
        look = rng.choice(look, size=min(4, len(look)), replace=False)
        cards = [table(f[f.scenario_id == s], f"MEDIUM suspicious", "med") for s in med]
        cards += [table(f[f.scenario_id == s], f"easy suspicious", "easy") for s in easy]
        cards += [table(f[f.scenario_id == s], f"legit look-alike", "legit") for s in look]
        parts.append(f"<section><h2 id='{fam}'>{fam} <span>{len(med)} medium · {len(easy)} easy shown · "
                     f"{len(look)} look-alikes shown</span></h2><div class=grid>{''.join(cards)}</div></section>")

    rel = p[p.cohort_id.notna()]
    cards = []
    for sid in rel[rel.is_suspicious == 1].scenario_id.unique():
        s = rel[rel.scenario_id == sid]
        cohort = s.cohort_id.iloc[0]
        cards.append(table(s, f"HARD suspicious {s.family.iloc[0]} (cohort {cohort[:6]})", "med"))
        sib = rel[(rel.cohort_id == cohort) & (rel.is_suspicious == 0)].scenario_id.unique()
        for t in rng.choice(sib, size=2, replace=False):
            cards.append(table(rel[rel.scenario_id == t], f"legit sibling (cohort {cohort[:6]})", "legit"))
    parts.append(f"<section><h2 id='relays'>hard relays <span>12 suspicious, each with 2 legit siblings "
                 f"from its cohort</span></h2><div class=grid>{''.join(cards)}</div></section>")

    nav = " · ".join(f"<a href='#{f}'>{f}</a>" for f in fams + ["relays"])
    doc = f"""<!doctype html><html><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>
<title>Scenario viewer</title><style>
:root{{--bg:#fafaf8;--fg:#1d1d1b;--mut:#6b6b66;--line:#e2e1dc;--med:#b4235a;--easy:#2563eb;--legit:#15803d;--hot:#fff4cc}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151514;--fg:#ecebe6;--mut:#9a9992;--line:#33332f;--hot:#3a3216}}}}
body{{margin:0;padding:16px;background:var(--bg);color:var(--fg);font:13px/1.4 system-ui,sans-serif}}
h1{{font-size:18px;margin:0 0 4px}} h2{{font-size:16px;margin:28px 0 8px;border-bottom:1px solid var(--line)}}
h2 span,h4 span{{color:var(--mut);font-weight:400;font-size:12px}} p{{color:var(--mut);max-width:900px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(520px,1fr));gap:12px}}
.card{{border:1px solid var(--line);border-left:4px solid;border-radius:6px;padding:8px;overflow-x:auto}}
.card.med{{border-left-color:var(--med)}} .card.easy{{border-left-color:var(--easy)}} .card.legit{{border-left-color:var(--legit)}}
h4{{margin:0 0 6px;font-size:13px}} table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}
th,td{{padding:2px 6px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}} th{{color:var(--mut);font-weight:500}}
td.n{{text-align:right}} tr.hot td{{background:var(--hot)}} a{{color:inherit}}
@media (max-width:600px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body>
<h1>Planted scenarios, train week</h1>
<p>Red = medium (or hard) suspicious, blue = easy suspicious, green = legitimate look-alike. Columns: minutes since the
scenario's first payment, gap from the previous step (min), wallets (W###, stable within the page), delivered USD,
asset (source→destination when converted; @xyz = issuer), how often this sender→receiver pair and its reverse occur in
the week, both wallets' weekly payment counts, and the payment's rank in our out-of-fold scores (yellow = inside the
top 1,250). Jump to: {nav}</p>
{''.join(parts)}</body></html>"""
    (OUT / "scenarios.html").write_text(doc)
    print(f"wrote {OUT / 'scenarios.html'} and {OUT / 'train_annotated.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", help="out-of-fold scores aligned with load_split('train') (.npy)")
    build(ap.parse_args().oof)
