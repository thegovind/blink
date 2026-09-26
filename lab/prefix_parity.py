"""Shared-prefix prefill on GPU: parity and speed against the plain path, same engine, same requests.

  source $BLINK_WORKDIR/env.sh
  CUDA_VISIBLE_DEVICES=0 python prefix_parity.py --runtime $BLINK_WORKDIR/prefix --ckpt $BLINK_WORKDIR/ckpt/soup-t3-t4s300-t4f \
      --name blink-4b --set typesafe=$BLINK_WORKDIR/prefix/typesafe-requests.jsonl \
      --set di-u1000=$BLINK_WORKDIR/kit/suite-u1000/selected-rows.jsonl.gz --out $BLINK_WORKDIR/runs/prefix/blink-4b.json

For every request with 2+ questions: logits with the plain path and with the shared-prefix path (alternating which runs
first), softmax at the engine temperature, then argmax agreement, |dp| and each path's model time. Single-question
requests take the plain path by construction and are only counted.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import statistics
import sys
import time
from pathlib import Path


def rows(path: str):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                yield r["id"], r["state"], r["questions"]


def softmax(xs, t):
    m = max(x / t for x in xs)
    e = [math.exp(x / t - m) for x in xs]
    s = sum(e)
    return [v / s for v in e]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", required=True, help="directory holding the blink.py under test")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--set", action="append", required=True, help="label=path (jsonl or jsonl.gz rows with id/state/questions)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    sys.path.insert(0, a.runtime)
    import blink

    eng = blink.TorchEngine(model_id=a.ckpt)
    report = {"name": a.name, "ckpt": a.ckpt, "runtime": str(Path(a.runtime) / "blink.py"),
              "prefix_min_tokens": blink.PREFIX_MIN_TOKENS, "prefix_kv_tokens": blink.PREFIX_KV_TOKENS, "sets": {}}
    for spec in a.set:
        label, path = spec.split("=", 1)
        data = list(rows(path))
        if a.limit:
            data = data[: a.limit]
        multi = [(i, s, q) for i, s, q in data if len(q) >= 2]
        for rid, state, qs in multi[:2]:  # warm both paths (kernel compile, allocator)
            for mode in (False, True):
                eng.prefix_cache = mode
                eng.logits(blink.as_state(state), qs)
        per_req, dps, flips, n_q = [], [], [], 0
        t_start = time.time()
        for k, (rid, state, qs) in enumerate(multi):
            st = blink.as_state(state)
            got = {}
            for mode in ((False, True) if k % 2 == 0 else (True, False)):
                eng.prefix_cache = mode
                lg, tokens = eng.logits(st, qs)
                got[mode] = (lg, eng.last_model_ms, tokens)
            (off, ms_off, tok), (on, ms_on, _) = got[False], got[True]
            shared = blink._shared_prefix_len([w["ids"] for w in eng.render(st, qs)])
            req_max = 0.0
            for q in qs:
                p0, p1 = softmax(off[q], eng.temperature), softmax(on[q], eng.temperature)
                dp = max(abs(x - y) for x, y in zip(p0, p1))
                req_max = max(req_max, dp)
                dps.append(dp)
                n_q += 1
                a0, a1 = max(range(len(p0)), key=p0.__getitem__), max(range(len(p1)), key=p1.__getitem__)
                if a0 != a1:
                    top = sorted(p0, reverse=True)
                    flips.append({"id": rid, "question": q, "margin": top[0] - top[1], "dp": dp})
            per_req.append({"id": rid, "questions": len(qs), "tokens": tok, "shared_prefix": shared,
                            "ms_off": ms_off, "ms_on": ms_on, "max_dp": req_max})
            if (k + 1) % 20 == 0:
                print(f"{a.name} {label} {k + 1}/{len(multi)} flips={len(flips)} max_dp={max(dps):.4g}", flush=True)
        dps_sorted = sorted(dps)
        tot_off, tot_on = sum(r["ms_off"] for r in per_req), sum(r["ms_on"] for r in per_req)
        report["sets"][label] = {
            "requests": len(data), "single_question_requests": len(data) - len(multi), "multi_question_requests": len(multi),
            "questions_compared": n_q, "argmax_agreement": 1 - len(flips) / max(n_q, 1),
            "max_dp": max(dps) if dps else 0.0, "p99_dp": dps_sorted[int(0.99 * (len(dps) - 1))] if dps else 0.0,
            "median_dp": statistics.median(dps) if dps else 0.0, "flips": flips,
            "model_ms_off": tot_off, "model_ms_on": tot_on, "speedup": tot_off / tot_on if tot_on else None,
            "median_request_speedup": statistics.median([r["ms_off"] / r["ms_on"] for r in per_req if r["ms_on"]]) if per_req else None,
            "wall_s": round(time.time() - t_start, 1), "requests_detail": per_req,
        }
        s = report["sets"][label]
        print(f"{a.name} {label}: {len(multi)} multi-question requests, {n_q} questions, agreement {s['argmax_agreement']:.4f}, "
              f"max|dp| {s['max_dp']:.4g}, p99 {s['p99_dp']:.3g}, model time {tot_off / 1000:.1f}s -> {tot_on / 1000:.1f}s "
              f"(x{s['speedup']:.2f})", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
