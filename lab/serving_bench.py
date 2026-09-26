"""E1 serving arms on one GPU, one engine load: timing plus per-question outputs for the parity gate.

  source $BLINK_WORKDIR/env.sh
  CUDA_VISIBLE_DEVICES=0 python serving_bench.py --runtime DIR --ckpt CKPT --requests R.jsonl[.gz] --out-dir D \
      [--arms h0,h2,h1c4,h1c16,h0ctl] [--warmup 8] [--limit 0]

Arms (experiments/e1/PREREG.md, E1.3):
  h0     release path, one request at a time (per-request latency)
  h2     batched FP32 readout (BLINK_READOUT=batched), one request at a time
  h1cN   cross-request packing: consecutive groups of N requests in one logits_many call (a batching server under
         load at concurrency N); latency is per group
  h0ctl  h0 again at the end, to measure timing drift
Each arm writes D/<arm>.jsonl in run_requests.py shape; D/summary.json holds the timings. First-seen timing only:
each request is timed once per arm, after a shared warm-up on the first --warmup requests.
"""
from __future__ import annotations

import argparse
import gzip
import json
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


def pct(xs, q):
    xs = sorted(xs)
    return round(xs[int(q * (len(xs) - 1))], 2) if xs else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--requests", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--arms", default="h0,h2,h1c4,h1c16,h0ctl")
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    sys.path.insert(0, a.runtime)
    import torch

    import blink

    t0 = time.perf_counter()
    eng = blink.TorchEngine(model_id=a.ckpt)
    eng.prefix_cache = False
    load_s = round(time.perf_counter() - t0, 1)
    reqs, refused = [], []
    for rid, state, qs in rows(a.requests):
        try:
            blink.validate(qs)
            eng.render(blink.as_state(state), qs)
            reqs.append((rid, blink.as_state(state), qs))
        except blink.BlinkError as exc:
            refused.append({"id": rid, "error": str(exc)})
        if a.limit and len(reqs) >= a.limit:
            break
    for _, st, qs in reqs[: a.warmup]:
        eng.logits(st, qs)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    T = eng.temperature

    def answers(qs, raw):
        return {k: blink.answer_for(q, [key for key, _ in blink.question_options(q)], blink.softmax(raw[k], T))
                for k, q in qs.items()}

    summary = {"ckpt": a.ckpt, "requests": len(reqs), "questions": sum(len(q) for _, _, q in reqs),
               "refused": len(refused), "engine_load_s": load_s, "warmup_requests": min(a.warmup, len(reqs)),
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "arms": {}}
    for arm in a.arms.split(","):
        blink.READOUT = "batched" if arm == "h2" else "rows"
        lat, wall0 = [], time.perf_counter()
        with (out_dir / f"{arm}.jsonl").open("w") as fh:
            if arm.startswith("h1c"):
                n = int(arm[3:])
                for g in range(0, len(reqs), n):
                    group = reqs[g:g + n]
                    s = time.perf_counter()
                    res = eng.logits_many([(st, qs) for _, st, qs in group])
                    lat.append((time.perf_counter() - s) * 1000)
                    for (rid, _, qs), (raw, n_tok) in zip(group, res):
                        fh.write(json.dumps({"id": rid, "output": {"answers": answers(qs, raw),
                                                                   "meta": {"input_tokens": n_tok}}}) + "\n")
            else:
                for rid, st, qs in reqs:
                    s = time.perf_counter()
                    raw, n_tok = eng.logits(st, qs)
                    lat.append((time.perf_counter() - s) * 1000)
                    fh.write(json.dumps({"id": rid, "output": {"answers": answers(qs, raw),
                                                               "meta": {"input_tokens": n_tok}}}) + "\n")
        wall = time.perf_counter() - wall0
        summary["arms"][arm] = {"wall_s": round(wall, 2), "questions_per_s": round(summary["questions"] / wall, 2),
                                "requests_per_s": round(len(reqs) / wall, 2),
                                "latency_ms": {"p50": pct(lat, 0.5), "p95": pct(lat, 0.95), "p99": pct(lat, 0.99),
                                               "mean": round(statistics.mean(lat), 2) if lat else None},
                                "latency_unit": "group" if arm.startswith("h1c") else "request"}
        print(arm, json.dumps(summary["arms"][arm]), flush=True)
    blink.READOUT = "rows"
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    (out_dir / "refused.json").write_text(json.dumps(refused, indent=1) + "\n")
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
