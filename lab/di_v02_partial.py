"""Partial, like-for-like comparison on the Decision Index 0.2 panel using full v0.1-suite results.

0.2 keeps every v0.1 row except stratified cuts of ToolRet and BRIGHT, and adds seven benchmarks. Our v0.1 kit
scores cover 33 of the 40 panel benchmarks; this computes area means over the shared benchmarks only, for us and
for every leaderboard system, so the comparison is on identical benchmark sets. It is NOT a 0.2 index:
seven benchmarks are missing, and ToolRet/BRIGHT scores are on the full v0.1 sets rather than the 0.2 subsets.

  python lab/di_v02_partial.py runs/full-X-scores.json [more score files] [--lb /tmp/dilb2/data/index.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def area_means(get, cats, keep):
    out = {}
    for c in cats:
        vals = [get(b) for b in c["panel"] if b in keep]
        out[c["id"]] = sum(vals) / len(vals) if vals else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scores", nargs="+")
    ap.add_argument("--lb", default="/tmp/dilb2/data/index.json")
    ap.add_argument("--top", type=int, default=8)
    a = ap.parse_args()
    lb = json.loads(Path(a.lb).read_text())
    cats = [c for c in lb["categories"] if c.get("panel")]
    panel = [b for c in cats for b in c["panel"]]
    ours = {p: json.loads(Path(p).read_text())["benchmarks"] for p in a.scores}
    keep = {b for b in panel if all(str(b) in o and o[str(b)].get("score") is not None for o in ours.values())}
    missing = [lb["benchmarks"][str(b)]["dataset"] for b in panel if b not in keep]
    print(f"shared panel benchmarks: {len(keep)}/{len(panel)}; missing: {', '.join(missing)}")
    rows = []
    for p, o in ours.items():
        am = area_means(lambda b: o[str(b)]["score"], cats, keep)
        rows.append((Path(p).stem.replace("-scores", ""), 100 * sum(am.values()) / len(am), am, "ours"))
    systems = [("Jev 1.13.0", lb["jev"]["results"])] + [(m["name"], m["results"]) for m in lb["models"]]
    for name, res in systems:
        if not all((res.get(str(b)) or {}).get("score") is not None for b in keep):
            continue
        am = area_means(lambda b: res[str(b)]["score"], cats, keep)
        rows.append((name, 100 * sum(am.values()) / len(am), am, "lb"))
    rows.sort(key=lambda r: -r[1])
    print(f"{'system':42s} {'partial':>7s}  " + " ".join(f"{c['id'][:5]:>6s}" for c in cats))
    shown = 0
    for name, idx, am, kind in rows:
        if kind == "lb" and shown >= a.top and not name.startswith("Jev"):
            continue
        shown += kind == "lb"
        print(f"{name[:42]:42s} {idx:7.2f}  " + " ".join(f"{100 * am[c['id']]:6.1f}" for c in cats))


if __name__ == "__main__":
    main()
