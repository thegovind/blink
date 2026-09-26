"""Typed decisions: request -> prompts -> scored options -> TypeSafe-shaped answers."""
from __future__ import annotations

import math

from .render import RenderError, question_options

MAX_QUESTIONS = 512


def orders_for(k, mode):
    ident = list(range(k))
    if mode in (None, "id") or k < 2:
        return [ident]
    if mode == "rev":
        return [ident, ident[::-1]]
    if mode == "rot":
        return [ident, ident[k // 2:] + ident[: k // 2]]
    raise ValueError(mode)


def build(renderer, state, questions, order_mode=None):
    """-> list of work items {qkey, keys, prompt_ids, cand_ids} (one per question x order)."""
    if not isinstance(questions, dict) or not questions:
        raise RenderError("questions must be a non-empty map")
    if len(questions) > MAX_QUESTIONS:
        raise RenderError("too many questions")
    work = []
    for qkey, q in questions.items():
        k = len(question_options(q))
        for order in orders_for(k, order_mode):
            prompt, keys, cand = renderer.render(state, q, order=order)
            ids = renderer.tok(prompt, add_special_tokens=False)["input_ids"]
            work.append({"qkey": qkey, "keys": keys, "prompt_ids": ids, "cand_ids": cand, "qtype": q["type"]})
    return work


def assemble(questions, work, logps):
    """Average probabilities over orders per question; build answers. Returns (answers, n_input_tokens) or raises."""
    acc = {}
    for w, lp in zip(work, logps):
        if lp is None:
            return None, None
        p = [math.exp(float(x)) for x in lp]
        s = sum(p)
        d = acc.setdefault(w["qkey"], {"n": 0, "p": {}})
        d["n"] += 1
        for key, v in zip(w["keys"], p):
            d["p"][key] = d["p"].get(key, 0.0) + v / s
    answers = {}
    for qkey, q in questions.items():
        d = acc[qkey]
        probs = {k: v / d["n"] for k, v in d["p"].items()}
        z = sum(probs.values())
        probs = {k: v / z for k, v in probs.items()}
        qtype = q["type"]
        if qtype == "choice":
            order = [k for k, _ in question_options(q)]
            best = max(order, key=lambda k: (probs[k], -order.index(k)))
            K = len(order)
            conf = (probs[best] - 1 / K) / (1 - 1 / K) if K > 1 else 1.0
            answers[qkey] = {"type": "choice", "choice": best, "probabilities": {k: probs[k] for k in order},
                             "confidence": conf}
        elif qtype == "noul":
            answers[qkey] = {"type": "noul", "noul": probs["yes"]}
        elif qtype == "score":
            levels = [str(i) for i in range(len(q["criteria"]))]
            ev = sum(int(k) * probs[k] for k in levels)
            best = max(levels, key=lambda k: (probs[k], -int(k)))
            answers[qkey] = {"type": "score", "score": ev, "probabilities": {k: probs[k] for k in levels},
                             "legend": {k: q["criteria"][int(k)] for k in levels}, "choice": best}
    n_tokens = sum(len(w["prompt_ids"]) for w in work)
    return answers, n_tokens
