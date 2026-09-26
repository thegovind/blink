"""Assemble a training mixture + DEV holdout from item files (group-disjoint split per source).

  python -m jevlab.mix --spec 'DIR/anli.jsonl:10000,DIR/esci.jsonl:10000,...' --out-train T.jsonl --out-dev D.jsonl --dev-per-src 300
Rows get "src" (from the row or file stem). Groups (meta.group) never straddle train/dev.
"""
import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


def gkey(r):
    return (r.get("meta") or {}).get("group") or r["id"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-dev", required=True)
    ap.add_argument("--dev-per-src", type=int, default=300)
    ap.add_argument("--seed", default="mix-v1")
    a = ap.parse_args()
    train, dev = [], []
    for part in a.spec.split(","):
        path, _, cap = part.rpartition(":")
        cap = int(cap)
        rows = [json.loads(l) for l in open(path) if l.strip()]
        stem = Path(path).stem
        for r in rows:
            r.setdefault("src", stem)
        groups = defaultdict(list)
        for r in rows:
            groups[gkey(r)].append(r)
        for g in [g for g, rs in groups.items() if len(rs) > 50]:  # generator-level groups are not latent worlds
            for r in groups.pop(g):
                groups[r["id"]].append(r)
        keys = sorted(groups, key=lambda g: hashlib.sha256(f"{a.seed}:{g}".encode()).hexdigest())
        d, t = [], []
        for g in keys:
            (d if len(d) < a.dev_per_src else t).extend(groups[g])
        random.Random(f"{a.seed}:{stem}").shuffle(t)
        train += t[:cap]
        dev += d
        print(f"{stem:14} total {len(rows):7} train {min(cap, len(t)):7} dev {len(d):4}")
    random.Random(a.seed).shuffle(train)
    with open(a.out_train, "w") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(a.out_dev, "w") as f:
        for r in dev:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("train", len(train), "dev", len(dev))


if __name__ == "__main__":
    main()
