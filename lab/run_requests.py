"""Run blink request rows through a local checkpoint with blink.py's own engine; one output row per request.

  source $BLINK_WORKDIR/env.sh
  CUDA_VISIBLE_DEVICES=0 python run_requests.py --runtime $BLINK_WORKDIR/prefix --ckpt $BLINK_WORKDIR/ckpt/soup-t3-t4s300-t4f \
      --requests heldout-requests.jsonl --out $BLINK_WORKDIR/runs/heldout/blink-4b.jsonl [--prefix-cache off]

Output rows: {"id", "output": {"answers", "meta"}} (the Space /v1_systemone shape; meta.model_ms = forward time).
Resumes: ids already in --out are skipped.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path


def rows(path: str):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                yield r["id"], r["state"], r["questions"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--requests", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix-cache", choices=("on", "off"), default="on")
    a = ap.parse_args()
    sys.path.insert(0, a.runtime)
    import blink

    eng = blink.TorchEngine(model_id=a.ckpt)
    eng.prefix_cache = a.prefix_cache == "on"
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["id"] for line in out.open()} if out.exists() else set()
    with out.open("a") as fh:
        for n, (rid, state, qs) in enumerate(rows(a.requests), 1):
            if rid in done:
                continue
            try:
                blink.validate(qs)
                raw, n_tokens = eng.logits(blink.as_state(state), qs)
            except blink.BlinkError as exc:  # over-limit or invalid: refused, recorded, scored as unanswered
                fh.write(json.dumps({"id": rid, "error": str(exc)}) + "\n")
                continue
            answers = {k: blink.answer_for(q, [key for key, _ in blink.question_options(q)],
                                           blink.softmax(raw[k], eng.temperature)) for k, q in qs.items()}
            meta = {"model": a.ckpt, "engine": "torch", "temperature": eng.temperature, "input_tokens": n_tokens,
                    "prefill_tokens": getattr(eng, "last_prefill_tokens", None), "generated_tokens": 0,
                    "model_ms": eng.last_model_ms, "prefix_cache": eng.prefix_cache}
            fh.write(json.dumps({"id": rid, "output": {"answers": answers, "meta": meta}}) + "\n")
            if n % 50 == 0:
                fh.flush()
                print(f"{n} requests", flush=True)
    print("wrote", out)


if __name__ == "__main__":
    main()
