"""Per-question reference probabilities for engine parity: blink's plain path in bf16 (the release) and in FP32.

  source $BLINK_WORKDIR/env.sh
  CUDA_VISIBLE_DEVICES=0 python fp32_reference.py --runtime $BLINK_WORKDIR/prefix --ckpt CKPT \
      --requests R.jsonl[.gz] --out refs.jsonl [--min-questions 1] [--limit 0]

One row per (request, question): {"id", "q", "keys", "bf16": [p...], "fp32": [p...], "input_tokens"}, probabilities
over the offered options in request order at temperature 1. Requests too large for the FP32 model are skipped and
listed in <out>.skips.json. `engine_parity.py` compares any other engine's outputs against these rows.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path


def rows(path: str, min_q: int):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                if len(r["questions"]) >= min_q:
                    yield r["id"], r["state"], r["questions"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--requests", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-questions", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    sys.path.insert(0, a.runtime)
    import torch

    import blink

    bf = blink.TorchEngine(model_id=a.ckpt)
    bf.prefix_cache = False
    f32 = blink.TorchEngine(model_id=a.ckpt)
    f32.model = f32.model.float()
    f32.prefix_cache = False
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    skips, n = [], 0
    with out.open("w") as fh:
        for idx, (rid, state, qs) in enumerate(rows(a.requests, a.min_questions)):
            if idx % a.nshards != a.shard:
                continue
            if a.limit and n >= a.limit:
                break
            st = blink.as_state(state)
            try:
                blink.validate(qs)
                p16, n_tok = bf.logits(st, qs)
                p32, _ = f32.logits(st, qs)
            except (blink.BlinkError, torch.cuda.OutOfMemoryError) as exc:
                skips.append({"id": rid, "why": f"{type(exc).__name__}: {str(exc)[:120]}"})
                torch.cuda.empty_cache()
                continue
            n += 1
            for q, spec in qs.items():
                keys = [k for k, _ in blink.question_options(spec)]
                fh.write(json.dumps({"id": rid, "q": q, "keys": keys, "input_tokens": n_tok,
                                     "bf16": blink.softmax(p16[q], 1.0), "fp32": blink.softmax(p32[q], 1.0)}) + "\n")
    Path(str(out) + ".skips.json").write_text(json.dumps({"requests": n, "skipped": skips}, indent=1) + "\n")
    print(f"wrote {out}: {n} requests, {len(skips)} skipped")


if __name__ == "__main__":
    main()
