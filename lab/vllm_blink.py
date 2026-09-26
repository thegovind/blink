"""blink's readout on vLLM: the same prompts and offered-letter restriction as blink.py, one prefill per question.

  source $BLINK_WORKDIR/env.sh            # vLLM 0.30, torch 2.13, fla 0.5.2
  CUDA_VISIBLE_DEVICES=0 python vllm_blink.py --runtime $BLINK_WORKDIR/prefix \
      --ckpt $BLINK_WORKDIR/ckpt/soup-t3-t4s300-t4f --requests R.jsonl --out O.jsonl [--fp32-logits] \
      [--no-prefix-caching] [--max-num-batched-tokens 8192] [--mode serial|throughput]

Prompts come from blink.TorchEngine.render (same chat template, same letters, same token ids), so the only
differences from the release path are vLLM's kernels and, without --fp32-logits, its bf16 logits. The readout is
the one validated in lab/jevlab/vllm_score.py: `allowed_token_ids` = the offered letter ids with
`logprobs_mode="processed_logprobs"` and `max_logprobs=256`, so vLLM returns the log-softmax over exactly the offered
letters (up to 255, one request per question), which is blink's softmax over the offered letters. --fp32-logits
runs the lm_head in FP32, as blink does (hf_overrides head_dtype=float32, vLLM's own switch).

Output rows match run_requests.py: {"id", "output": {"answers", "meta"}}; meta.model_ms is the engine wall time.
--mode serial: one request at a time (latency). --mode throughput: every request in one generate call (continuous
batching; Q/s). Resumes: ids already in --out are skipped (serial mode only).
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
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


def fp32_head_overrides() -> dict:
    """vLLM 0.30's own switch for an FP32 lm_head (blink's readout precision): bf16 inputs, FP32 accumulation and
    output via torch.mm(out_dtype=float32). Documented in vllm/config/model.py (ModelConfig.head_dtype)."""
    return {"head_dtype": "float32"}


class _Render:
    """blink.TorchEngine's tokenizer, label verification and prompt rendering, without loading HF weights."""

    def __init__(self, blink, ckpt: str):
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(ckpt)
        self._wrap = blink.TorchEngine._wrap.__get__(self)
        self.labels, self.label_ids = blink.TorchEngine._verify_labels(self)
        self.render = blink.TorchEngine.render.__get__(self)


class VllmEngine:
    name = "vllm"

    def __init__(self, blink, ckpt: str, a):
        from vllm import LLM

        self.blink = blink
        self.r = _Render(blink, ckpt)
        self.temperature = blink.TEMPERATURE
        kw = dict(model=ckpt, tokenizer=ckpt, dtype="bfloat16", seed=0, enable_prefix_caching=a.prefix_caching,
                  max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_mem, max_num_seqs=a.max_num_seqs,
                  max_logprobs=256, logprobs_mode="processed_logprobs")
        if a.fp32_logits:
            kw["hf_overrides"] = fp32_head_overrides()
        if a.max_num_batched_tokens:
            kw["max_num_batched_tokens"] = a.max_num_batched_tokens
        if a.no_mm:
            kw["limit_mm_per_prompt"] = {"image": 0, "video": 0}
        if a.extra:
            kw.update(json.loads(a.extra))
        t0 = time.perf_counter()
        self.llm = LLM(**kw)
        self.ready_s = round(time.perf_counter() - t0, 1)
        self.kw = {k: v for k, v in kw.items() if k not in ("model", "tokenizer")}

    def _jobs(self, work):
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        prompts, params, where = [], [], []
        for wi, w in enumerate(work):
            cand = list(w["cand"])
            prompts.append(TokensPrompt(prompt_token_ids=w["ids"]))
            params.append(SamplingParams(max_tokens=1, temperature=1.0, logprobs=len(cand), allowed_token_ids=cand,
                                         detokenize=False))
            where.append((wi, cand))
        return prompts, params, where

    @staticmethod
    def _collect(work, outs, where):
        got = [dict() for _ in work]
        for out, (wi, chunk) in zip(outs, where):
            lp = out.outputs[0].logprobs[0]
            for tid in chunk:
                got[wi][tid] = float(lp[tid].logprob) if tid in lp else float("-inf")
        return [[got[wi][tid] for tid in w["cand"]] for wi, w in enumerate(work)]

    def logits(self, state, questions: dict):
        work = self.r.render(state, questions)
        prompts, params, where = self._jobs(work)
        t0 = time.perf_counter()
        outs = self.llm.generate(prompts, params, use_tqdm=False)
        self.last_model_ms = round((time.perf_counter() - t0) * 1000, 1)
        rows_ = self._collect(work, outs, where)
        return {w["qkey"]: r for w, r in zip(work, rows_)}, sum(len(w["ids"]) for w in work)


def answers_for(blink, qs, raw, T):
    return {k: blink.answer_for(q, [key for key, _ in blink.question_options(q)], blink.softmax(raw[k], T))
            for k, q in qs.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", required=True, help="directory holding the blink.py whose prompts are used")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--requests", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=("serial", "throughput"), default="serial")
    ap.add_argument("--fp32-logits", action="store_true")
    ap.add_argument("--no-prefix-caching", dest="prefix_caching", action="store_false")
    ap.add_argument("--max-num-batched-tokens", type=int, default=0)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--no-mm", action="store_true", help="vision-language checkpoints: reserve nothing for images")
    ap.add_argument("--extra", default="", help="JSON of extra vLLM LLM(...) kwargs, one knob per arm")
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    sys.path.insert(0, a.runtime)
    import blink

    eng = VllmEngine(blink, a.ckpt, a)
    T = eng.temperature
    reqs = list(rows(a.requests))
    if a.limit:
        reqs = reqs[: a.limit]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta0 = {"model": a.ckpt, "engine": "vllm", "temperature": T, "generated_tokens": 0, "vllm": eng.kw,
             "fp32_logits": a.fp32_logits, "ready_s": eng.ready_s}
    ok = []
    for rid, state, qs in reqs:
        try:
            blink.validate(qs)
            ok.append((rid, blink.as_state(state), qs))
        except blink.BlinkError as exc:
            ok.append((rid, None, str(exc)))
    for rid, state, qs in [r for r in ok if r[1] is not None][: a.warmup]:
        eng.logits(state, qs)
    if a.mode == "serial":
        done = {json.loads(line)["id"] for line in out.open()} if out.exists() else set()
        with out.open("a") as fh:
            for n, (rid, state, qs) in enumerate(ok, 1):
                if rid in done:
                    continue
                if state is None:
                    fh.write(json.dumps({"id": rid, "error": qs}) + "\n")
                    continue
                try:
                    raw, n_tok = eng.logits(state, qs)
                except blink.BlinkError as exc:
                    fh.write(json.dumps({"id": rid, "error": str(exc)}) + "\n")
                    continue
                meta = dict(meta0, input_tokens=n_tok, model_ms=eng.last_model_ms)
                fh.write(json.dumps({"id": rid, "output": {"answers": answers_for(blink, qs, raw, T), "meta": meta}}) + "\n")
                if n % 50 == 0:
                    fh.flush()
                    print(f"{n} requests", flush=True)
    else:
        good = [r for r in ok if r[1] is not None]
        work_all, spans = [], []
        for rid, state, qs in good:
            w = eng.r.render(state, qs)
            spans.append((len(work_all), len(w)))
            work_all.extend(w)
        prompts, params, where = eng._jobs(work_all)
        t0 = time.perf_counter()
        outs = eng.llm.generate(prompts, params, use_tqdm=False)
        wall = time.perf_counter() - t0
        rows_ = eng._collect(work_all, outs, where)
        with out.open("w") as fh:
            for (rid, state, qs), (s0, n) in zip(good, spans):
                raw = {w["qkey"]: r for w, r in zip(work_all[s0:s0 + n], rows_[s0:s0 + n])}
                meta = dict(meta0, input_tokens=sum(len(w["ids"]) for w in work_all[s0:s0 + n]))
                fh.write(json.dumps({"id": rid, "output": {"answers": answers_for(blink, qs, raw, T), "meta": meta}}) + "\n")
        n_q = len(work_all)
        summary = {"mode": "throughput", "requests": len(good), "questions": n_q, "wall_s": round(wall, 2),
                   "questions_per_s": round(n_q / wall, 2), "requests_per_s": round(len(good) / wall, 2),
                   "input_tokens": sum(len(w["ids"]) for w in work_all), **meta0}
        Path(str(out) + ".summary.json").write_text(json.dumps(summary, indent=1) + "\n")
        print(json.dumps(summary))
    print("wrote", out)


if __name__ == "__main__":
    main()
