"""Serial per-request latency of the published runtime (space/blink.py) on a uniform sample of the suite.

Comparable in kind to the Decision Index board's latency (per-request, synchronized, excludes model loading):
one request at a time, blink.decide end to end (rendering, tokenization, prefill, readout).
  python -m jevlab.latency_sample --model CKPT --space $BLINK_WORKDIR/space --out OUT.json [--n 1000 --seed 0]
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1) + 0.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--space", required=True)
    ap.add_argument("--rows", default="decision-index/suite/selected-rows.jsonl.gz")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.environ["BLINK_ENGINE"] = "torch"
    sys.path.insert(0, a.space)
    import blink
    import torch

    with gzip.open(a.rows, "rt") as fh:
        total = sum(1 for _ in fh)
    pick = set(random.Random(a.seed).sample(range(total), a.n))
    rows = []
    with gzip.open(a.rows, "rt") as fh:
        for i, line in enumerate(fh):
            if i in pick:
                r = json.loads(line)
                rows.append({"id": r["id"], "family": r.get("family"), "state": r["state"], "questions": r["questions"]})
    random.Random(a.seed + 1).shuffle(rows)

    eng = blink.TorchEngine(a.model)
    blink._ENGINE = eng
    for r in rows[: a.warmup]:
        blink.decide(r["state"], r["questions"])
    recs, fails = [], 0
    for r in rows:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            out = blink.decide(r["state"], r["questions"])
        except blink.BlinkError as exc:
            fails += 1
            recs.append({"id": r["id"], "family": r["family"], "error": str(exc)[:200]})
            continue
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        recs.append({"id": r["id"], "family": r["family"], "questions": len(r["questions"]),
                     "input_tokens": out["meta"]["input_tokens"], "ms": round(ms, 2)})
    ok = [x["ms"] for x in recs if "ms" in x]
    summ = {"model": a.model, "n": len(recs), "ok": len(ok), "unsupported": fails, "seed": a.seed,
            "median_ms": round(statistics.median(ok), 1), "p95_ms": round(pct(ok, 0.95), 1),
            "mean_ms": round(statistics.fmean(ok), 1),
            "mean_input_tokens": round(statistics.fmean(x["input_tokens"] for x in recs if "ms" in x), 1),
            "token_budget": eng.token_budget}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"summary": summ, "rows": recs}, indent=1))
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
