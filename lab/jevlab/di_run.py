"""Sharded, batched Decision Index runner writing kit-format results rows.

Usage (one process per GPU):
  python -m jevlab.di_run --rows ROWS.jsonl.gz --out OUT --model M --shard i --nshards n [--adapter A]
Rows are the kit's frozen-suite rows. Only `state` and `questions` reach the model.
Latency in these rows is amortised batch time (throughput mode), not single-request latency.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch

from .decide import assemble, build
from .render import MAX_OPTIONS, RenderError, Renderer
from .scorer import Scorer, load_model


def stamp():
    return datetime.now(timezone.utc).isoformat()


def read_rows(path):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--template", default="v1")
    ap.add_argument("--backend", default="hf", choices=["hf", "vllm"])
    ap.add_argument("--max-model-len", type=int, default=65536)
    ap.add_argument("--order", default="id")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--budget", type=int, default=32768)
    ap.add_argument("--max-tokens", type=int, default=131072)
    ap.add_argument("--chunk-rows", type=int, default=256)
    ap.add_argument("--name", default=None)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)

    sys.path.insert(0, os.environ.get("DECISION_INDEX_KIT", "decision-index"))
    from decision_index.engines.base import validate

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res_path = out / f"results.shard{a.shard:02d}.jsonl"
    done = set()
    if res_path.exists():
        for line in res_path.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("status") != "error":
                done.add(r["run_id"])
    t0 = time.time()
    if a.backend == "vllm":
        from transformers import AutoTokenizer
        from .vllm_score import VLLMScorer
        tok = AutoTokenizer.from_pretrained(a.model)
        sc = VLLMScorer(a.model, max_model_len=min(a.max_model_len, a.max_tokens + 1), temperature=a.temperature)
        a.max_tokens = sc.max_tokens
    else:
        tok, model = load_model(a.model, revision=a.revision, adapter=a.adapter)
        sc = Scorer(model, tok, token_budget=a.budget, max_tokens=a.max_tokens, temperature=a.temperature)
    rd = Renderer(tok, template=a.template)
    name = a.name or f"jevlab:{a.model}:{a.template}:{a.order}" + (f":{a.adapter}" if a.adapter else "")
    env = {"engine": "jevlab", "name": name, "model": a.model, "revision": a.revision, "adapter": a.adapter,
           "template": a.template, "order": a.order, "temperature": a.temperature, "max_tokens": a.max_tokens,
           "budget": a.budget, "backend": a.backend, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
           "loaded_seconds": time.time() - t0,
           "latency": "amortised batched throughput time per request (not single-request latency)",
           "renderer_system": rd.system}
    (out / f"environment.shard{a.shard:02d}.json").write_text(json.dumps(env, indent=1))

    rows = [r for i, r in enumerate(read_rows(a.rows)) if i % a.nshards == a.shard]
    if a.limit:
        rows = rows[: a.limit]
    rows = [r for r in rows if r["_evaluation"]["run_id"] not in done]
    print(f"shard {a.shard}: {len(rows)} rows to do, {len(done)} done", flush=True)
    counts = {}
    with res_path.open("a", encoding="utf-8") as f:
        for c0 in range(0, len(rows), a.chunk_rows):
            chunk = rows[c0: c0 + a.chunk_rows]
            items, spans, pre = [], [], {}
            for r in chunk:
                rid = r["_evaluation"]["run_id"]
                try:
                    for q in r["questions"].values():
                        crit = q.get("criteria")
                        if q.get("type") == "choice" and isinstance(crit, dict) and len(crit) > MAX_OPTIONS:
                            raise RenderError("options per choice over 255")
                    w = build(rd, r["state"], r["questions"], a.order)
                    if max(len(x["prompt_ids"]) for x in w) > a.max_tokens:
                        pre[rid] = ("unsupported", f"prompt longer than the {a.max_tokens}-token context window")
                        spans.append((r, None))
                        continue
                    spans.append((r, (len(items), len(items) + len(w))))
                    items.extend(w)
                except RenderError as e:
                    pre[rid] = ("unsupported", str(e))
                    spans.append((r, None))
                except Exception as e:  # noqa: BLE001
                    pre[rid] = ("error", f"{type(e).__name__}: {e}")
                    spans.append((r, None))
            torch.cuda.synchronize()
            t = time.perf_counter()
            try:
                logps = sc.score([w["prompt_ids"] for w in items], [w["cand_ids"] for w in items])
                err = None
            except torch.cuda.OutOfMemoryError as e:
                logps, err = [None] * len(items), f"OutOfMemoryError: {e}"
                torch.cuda.empty_cache()
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - t) * 1000
            tot_tokens = max(1, sum(len(w["prompt_ids"]) for w in items))
            for r, span in spans:
                e = r["_evaluation"]
                rid = e["run_id"]
                row = {**e, "started_utc": stamp(), "engine": "jevlab"}
                if span is None:
                    st, msg = pre[rid]
                    row.update(status=st, error=msg)
                    ms = 0.0
                else:
                    w, lp = items[span[0]: span[1]], logps[span[0]: span[1]]
                    ntok = sum(len(x["prompt_ids"]) for x in w)
                    ms = elapsed * ntok / tot_tokens
                    try:
                        if err:
                            raise RuntimeError(err)
                        answers, n_in = assemble(r["questions"], w, lp)
                        if answers is None:
                            raise RenderError("over capacity")
                        resp = {"model": name, "answers": answers, "usage": {"input_tokens": n_in, "output_tokens": 0}}
                        validate(r["questions"], resp)
                        row.update(status="ok", response=resp)
                    except RenderError as ex:
                        row.update(status="unsupported", error=str(ex))
                    except Exception as ex:  # noqa: BLE001
                        row.update(status="error", error=str(ex), exception=type(ex).__name__,
                                   traceback=traceback.format_exc()[-2000:])
                row.update(completed_utc=stamp(), total_wall_ms=ms, model_request_wall_ms=ms)
                counts[row["status"]] = counts.get(row["status"], 0) + 1
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(f"shard {a.shard}: {c0 + len(chunk)}/{len(rows)} {counts} {elapsed / 1000:.1f}s "
                  f"{tot_tokens / max(elapsed / 1000, 1e-9):.0f} tok/s", flush=True)
    print(f"shard {a.shard}: complete {counts}", flush=True)


if __name__ == "__main__":
    main()
