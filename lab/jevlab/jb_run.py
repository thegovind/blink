"""JevBench v1.2 public items (231) through jevlab, scored with JevBench's own code.

Estimates only: the official score needs the judge tier, the held-out items and the sealed set.
  python -m jevlab.jb_run --model M [--adapter A] --out OUT.json [--serial]
"""
from __future__ import annotations

import os
import argparse
import json
import sys
import time
from pathlib import Path

import torch

from .decide import assemble, build
from .render import Renderer
from .scorer import Scorer, load_model

JB = os.environ.get("JEVBENCH_DIR", "jevbench")
TIERS = {"easy": "easy", "original": "standard", "hard": "hard"}
TARIFF = {"Qwen/Qwen3.5-4B": 0.03, "Qwen/Qwen3.5-9B": 0.10, "XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B": 0.10,
          "Qwen/Qwen3.8-27B": 0.214}


def request_for(task):
    q = {"type": task.question["type"], "instructions": task.question["instructions"]}
    if task.question.get("criteria") is not None:
        q["criteria"] = task.question["criteria"]
    return task.state, {"decision": q}


def to_probs(ans, qtype):
    if qtype == "noul":
        p = ans["noul"]
        return {"yes": p, "no": 1.0 - p}
    return ans["probabilities"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--template", default="v1")
    ap.add_argument("--backend", default="hf", choices=["hf", "vllm"])
    ap.add_argument("--max-model-len", type=int, default=65536)
    ap.add_argument("--order", default="id")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--serial", action="store_true", help="also time one request at a time")
    ap.add_argument("--tariff", type=float, default=None)
    a = ap.parse_args(argv)
    sys.path.insert(0, JB)
    from jevbench import composite_v13 as C13
    from jevbench.metrics import ece_top_label
    from jevbench.scoring import score_task
    from jevbench.tasks import load_jsonl

    if a.backend == "vllm":
        from transformers import AutoTokenizer
        from .vllm_score import VLLMScorer
        tok = AutoTokenizer.from_pretrained(a.model)
        sc = VLLMScorer(a.model, max_model_len=a.max_model_len, temperature=a.temperature)
    else:
        tok, model = load_model(a.model, revision=a.revision, adapter=a.adapter)
        sc = Scorer(model, tok, temperature=a.temperature)
    rd = Renderer(tok, template=a.template)
    tasks = []
    for f, tier in TIERS.items():
        for t in load_jsonl(f"{JB}/datasets/public/{f}.jsonl"):
            tasks.append((tier, t))
    works = []
    for tier, t in tasks:
        state, qs = request_for(t)
        works.append(build(rd, state, qs, a.order))
    flat = [w for ws in works for w in ws]
    t0 = time.perf_counter()
    lps = sc.score([w["prompt_ids"] for w in flat], [w["cand_ids"] for w in flat])
    batch_s = time.perf_counter() - t0
    per, i = [], 0
    for (tier, t), ws in zip(tasks, works):
        lp = lps[i: i + len(ws)]
        i += len(ws)
        state, qs = request_for(t)
        answers, n_in = assemble(qs, ws, lp)
        probs = to_probs(answers["decision"], t.question["type"])
        s = score_task(probs, t)
        per.append({"id": t.id, "tier": tier, "family": t.family, "type": t.question["type"], "correct": s["correct"],
                    "valid": s["valid"], "probs": s.get("probs"), "predicted": s.get("predicted"),
                    "expected": t.expected, "input_tokens": n_in, "gold_probs": t.provenance.get("gold_probs")})
    lat = []
    if a.serial:
        for (tier, t) in tasks:
            state, qs = request_for(t)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            ws = build(rd, state, qs, a.order)
            lp = sc.score([w["prompt_ids"] for w in ws], [w["cand_ids"] for w in ws])
            assemble(qs, ws, lp)
            torch.cuda.synchronize()
            lat.append((tier, time.perf_counter() - t1))
    tiers = {}
    for tier in ("easy", "standard", "hard"):
        rs = [r for r in per if r["tier"] == tier and r["correct"] is not None]
        tiers[tier] = sum(bool(r["correct"]) for r in rs) / len(rs)
    hard = [r for r in per if r["tier"] == "hard" and r["valid"] and r["correct"] is not None]
    ece = ece_top_label([(max(r["probs"].values()), r["correct"]) for r in hard])["ece"]
    gp = [r for r in per if r["gold_probs"]]
    tvd = sum(0.5 * sum(abs(r["probs"][k] - r["gold_probs"][k]) for k in r["gold_probs"]) for r in gp) / len(gp)
    cal = C13.calibration(ece, tvd)
    intel = C13.intelligence(tiers)
    toks = sum(r["input_tokens"] for r in per) / len(per)
    tariff = a.tariff if a.tariff is not None else TARIFF.get(a.model)
    usd_1000 = toks * 1000 * tariff / 1e6 if tariff else None
    cost = C13.cost(usd_1000) if usd_1000 else None
    summ = {"model": a.model, "adapter": a.adapter, "template": a.template, "order": a.order,
            "temperature": a.temperature, "tiers": tiers, "all_public_acc": sum(bool(r["correct"]) for r in per) / len(per),
            "hard_ece": ece, "prob_tvd": tvd, "calibration": cal, "intelligence_public": intel,
            "input_tokens_per_decision": toks, "usd_per_1000": usd_1000, "cost": cost, "batch_seconds": batch_s,
            "hard_by_family": {}}
    for fam in sorted({r["family"] for r in per if r["tier"] == "hard"}):
        rs = [r for r in per if r["tier"] == "hard" and r["family"] == fam]
        summ["hard_by_family"][fam] = round(sum(bool(r["correct"]) for r in rs) / len(rs), 3)
    if lat:
        xs = sorted(x for _, x in lat)
        p50, p95 = xs[len(xs) // 2], xs[int(0.95 * (len(xs) - 1))]
        spd = C13.speed(p50, p95, "gpu")
        summ.update(p50_s=p50, p95_s=p95, speed=spd, gpu=torch.cuda.get_device_name(0))
        if cost is not None:
            from jevbench import composite_v14 as C14
            axes = {"intelligence": intel, "calibration": cal, "speed": spd, "cost": cost}
            summ["est_score_public_only"] = C14.harmonic(axes)
            summ["est_axes"] = axes
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"summary": summ, "per_task": per}, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
