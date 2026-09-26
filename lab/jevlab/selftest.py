"""Renderer/scorer checks against a real tokenizer + model (run on an evaluation environment).

  python -m jevlab.selftest --model Qwen/Qwen3.5-4B
"""
from __future__ import annotations

import argparse
import sys

import torch

from .decide import assemble, build
from .render import Renderer
from .scorer import Scorer, load_model


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--template", default="v1")
    a = ap.parse_args(argv)
    tok, model = load_model(a.model)
    rd = Renderer(tok, template=a.template)
    fails = []

    def check(cond, msg):
        print(("ok   " if cond else "FAIL ") + msg)
        if not cond:
            fails.append(msg)

    check(len(rd.labels) == 255 and len(set(rd.label_ids)) == 255, "255 unique single-token labels")
    for K in (2, 26, 27, 255):
        q = {"type": "choice", "instructions": "Pick the matching item.", "criteria": {f"k{i}": f"item number {i}" for i in range(K)}}
        prompt, keys, cand = rd.render({"note": "x"}, q)
        base = tok(prompt, add_special_tokens=False)["input_ids"]
        good = all(tok(prompt + lab, add_special_tokens=False)["input_ids"] == base + [cid]
                   for lab, cid in zip(rd.labels[:K], cand))
        check(len(cand) == K and good and keys == [f"k{i}" for i in range(K)], f"K={K} boundary-verified labels")
    qs = {"QID_SECRET_123": {"type": "choice", "instructions": "Which?", "criteria": {"a1": "one", "b2": "two"}}}
    w = build(rd, "state text", qs)
    check("QID_SECRET_123" not in tok.decode(w[0]["prompt_ids"]), "question id not rendered")
    nq = {"n": {"type": "noul", "instructions": "Is it raining?"},
          "s": {"type": "score", "instructions": "How severe?", "criteria": ["none", "mild", "severe"]}}
    w = build(rd, "It is pouring outside.", nq)
    sc = Scorer(model, tok)
    lp = sc.score([x["prompt_ids"] for x in w], [x["cand_ids"] for x in w])
    ans, n = assemble(nq, w, lp)
    check(0 <= ans["n"]["noul"] <= 1 and abs(sum(ans["s"]["probabilities"].values()) - 1) < 1e-6, "noul/score assemble")
    print("   noul p(yes) =", round(ans["n"]["noul"], 4), " score =", round(ans["s"]["score"], 3))
    # batched right-padded vs one-at-a-time
    prompts = []
    for L in (1, 5, 40, 300):
        q = {"type": "choice", "instructions": "Which city is the capital of France?", "criteria": {"a": "Berlin", "b": "Paris", "c": "Rome"}}
        prompts += build(rd, "filler sentence. " * L, {"q": q})
    single = [sc.score([p["prompt_ids"]], [p["cand_ids"]])[0] for p in prompts]
    sc.token_budget = 10 ** 9
    batched = sc.score([p["prompt_ids"] for p in prompts], [p["cand_ids"] for p in prompts])
    diff = max(float((s - b).abs().max()) for s, b in zip(single, batched))
    # bf16 shape-dependent noise is ~0.1-0.17 logits; fp32 padded/batched vs single agree to <0.004 (2026-09-23 diag)
    check(diff < 0.25, f"batched vs single max |dlogp| = {diff:.4f} (bf16 tolerance 0.25)")
    # order-reversal bookkeeping
    q = {"type": "choice", "instructions": "Capital of France?", "criteria": {"a": "Berlin", "b": "Paris", "c": "Rome"}}
    w2 = build(rd, "", {"q": q}, "rev")
    lp2 = sc.score([x["prompt_ids"] for x in w2], [x["cand_ids"] for x in w2])
    ans2, _ = assemble({"q": q}, w2, lp2)
    check(ans2["q"]["choice"] == "b", f"reverse-order averaging picks Paris ({ans2['q']['probabilities']})")
    print("FAILURES:", fails if fails else "none")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
