"""Numerical accuracy of the two bf16 paths against an FP32 reference (plain path, FP32 weights and activations).

  CUDA_VISIBLE_DEVICES=3 python precision_check.py --runtime $BLINK_WORKDIR/prefix --ckpt CKPT --name blink-4b \
      --set typesafe=... --set di-u1000=... --set heldout=... --out $BLINK_WORKDIR/runs/prefix/precision-blink-4b.json

For every multi-question request and question: p from (a) plain bf16, (b) shared-prefix bf16, (c) plain FP32.
Reports each bf16 path's max/p99 |p - p_fp32| and argmax agreement with FP32, and flips of (b) vs (a) with margins.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from pathlib import Path


def rows(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                if len(r["questions"]) >= 2:
                    yield r["id"], r["state"], r["questions"]


def softmax(xs):
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    s = sum(e)
    return [v / s for v in e]


def argmax(p):
    return max(range(len(p)), key=p.__getitem__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--set", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.runtime)
    import torch

    import blink

    bf = blink.TorchEngine(model_id=a.ckpt)
    f32 = blink.TorchEngine(model_id=a.ckpt)
    f32.model = f32.model.float()
    f32.prefix_cache = False
    report = {"name": a.name, "sets": {}}
    for spec in a.set:
        label, path = spec.split("=", 1)
        errs = {"plain": [], "prefix": []}
        agree = {"plain": 0, "prefix": 0}
        flips, n = [], 0
        for rid, state, qs in rows(path):
            st = blink.as_state(state)
            try:
                bf.prefix_cache = False
                plain, _ = bf.logits(st, qs)
                bf.prefix_cache = True
                pref, _ = bf.logits(st, qs)
                ref, _ = f32.logits(st, qs)
            except (blink.BlinkError, torch.cuda.OutOfMemoryError) as exc:
                print("skip", rid, type(exc).__name__, str(exc)[:80], flush=True)
                torch.cuda.empty_cache()
                continue
            for q in qs:
                p_ref, p_plain, p_pref = softmax(ref[q]), softmax(plain[q]), softmax(pref[q])
                n += 1
                for key, p in (("plain", p_plain), ("prefix", p_pref)):
                    errs[key].append(max(abs(x - y) for x, y in zip(p, p_ref)))
                    agree[key] += argmax(p) == argmax(p_ref)
                if argmax(p_plain) != argmax(p_pref):
                    top = sorted(p_ref, reverse=True)
                    flips.append({"id": rid, "q": q, "fp32_margin": top[0] - top[1],
                                  "plain_matches_fp32": argmax(p_plain) == argmax(p_ref),
                                  "prefix_matches_fp32": argmax(p_pref) == argmax(p_ref)})
        s = {"questions": n, "flips_prefix_vs_plain": flips}
        for key in ("plain", "prefix"):
            e = sorted(errs[key])
            s[key] = {"max_dp_vs_fp32": e[-1] if e else 0, "p99_dp_vs_fp32": e[int(0.99 * (len(e) - 1))] if e else 0,
                      "median_dp_vs_fp32": e[len(e) // 2] if e else 0, "argmax_agreement_with_fp32": agree[key] / max(n, 1)}
        report["sets"][label] = s
        print(a.name, label, n, "questions | plain vs fp32:", {k: round(v, 5) for k, v in s["plain"].items()},
              "| prefix vs fp32:", {k: round(v, 5) for k, v in s["prefix"].items()}, "| flips:", len(flips), flush=True)
    Path(a.out).write_text(json.dumps(report, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
