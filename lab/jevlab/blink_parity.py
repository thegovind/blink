"""Numeric parity: the Space runtime (space/blink.py TorchEngine) vs the jevlab scorer.

Requests: JevBench public (231) + JevK5 hard-65. Both paths run one request per forward
(identical batch shapes), so identical math should give identical label logits.
  python -m jevlab.blink_parity --model $BLINK_WORKDIR/ckpt/X --space $BLINK_WORKDIR/space --out parity.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from .decide import assemble, build
from .jb_run import JB, TIERS, request_for
from .render import Renderer
from .scorer import Scorer, load_model


def load_requests(extra):
    sys.path.insert(0, JB)
    from jevbench.tasks import load_jsonl

    reqs = []
    for f in TIERS:
        for t in load_jsonl(f"{JB}/datasets/public/{f}.jsonl"):
            state, qs = request_for(t)
            reqs.append((t.id, state, qs))
    if extra and Path(extra).exists():
        for line in open(extra):
            r = json.loads(line)
            reqs.append((r["id"], r["state"], {"decision": r["question"]}))
    return reqs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--space", required=True)
    ap.add_argument("--extra", default="data/mix/jevk5_hard65.jsonl")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--share", action="store_true", help="reuse the scorer's weights (27B: two copies do not fit)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    sys.path.insert(0, a.space)
    import blink

    reqs = load_requests(a.extra)
    tok, model = load_model(a.model)
    sc = Scorer(model, tok, temperature=a.temperature)
    rd = Renderer(tok, template="semif")
    if a.share:
        eng = blink.TorchEngine.__new__(blink.TorchEngine)
        eng.model_id, eng.temperature, eng.tok = a.model, a.temperature, tok
        eng.model, eng.body, eng.lm_head = model, model.model, model.lm_head.weight
        eng.pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
        eng.labels, eng.label_ids = eng._verify_labels()
        eng.token_budget = blink.TOKEN_BUDGET
        eng.key = id(eng)
        blink._LIVE[eng.key] = eng
    else:
        eng = blink.TorchEngine(a.model, temperature=a.temperature)  # exercises the Space's own loading path

    rows, n_ids_diff, n_cand_diff, n_argmax_diff, max_dlogit, max_dp = [], 0, 0, 0, 0.0, 0.0
    for rid, state, qs in reqs:
        ws = build(rd, state, qs, "id")
        lps = sc.score([w["prompt_ids"] for w in ws], [w["cand_ids"] for w in ws])
        ours, _ = assemble(qs, ws, lps)
        bw = eng.render(state, qs)
        for w, b in zip(ws, bw):
            n_ids_diff += w["prompt_ids"] != b["ids"]
            n_cand_diff += list(w["cand_ids"]) != list(b["cand"])
        raw, _ = eng.logits(state, qs)
        lp_by_q = {w["qkey"]: lp for w, lp in zip(ws, lps)}
        for qkey, q in qs.items():
            keys = [k for k, _ in blink.question_options(q)]
            pb = dict(zip(keys, blink.softmax(raw[qkey], a.temperature)))
            po = ours[qkey]["probabilities"] if q["type"] != "noul" else {"yes": ours[qkey]["noul"], "no": 1 - ours[qkey]["noul"]}
            lo = [float(x) for x in lp_by_q[qkey]]
            lb = [x / a.temperature for x in raw[qkey]]
            zo = [x - max(lo) for x in lo]
            zb = [x - max(lb) for x in lb]
            lse_o = math.log(sum(math.exp(x) for x in zo))
            lse_b = math.log(sum(math.exp(x) for x in zb))
            dl = max(abs((x - lse_o) - (y - lse_b)) for x, y in zip(zo, zb))
            dp = max(abs(po[k] - pb[k]) for k in keys)
            am = max(keys, key=lambda k: po[k]) != max(keys, key=lambda k: pb[k])
            n_argmax_diff += am
            max_dlogit, max_dp = max(max_dlogit, dl), max(max_dp, dp)
            rows.append({"id": rid, "q": qkey, "dlogp": dl, "dp": dp, "argmax_diff": bool(am)})
    dps = sorted(r["dp"] for r in rows)
    summ = {"model": a.model, "requests": len(reqs), "questions": len(rows),
            "prompt_ids_mismatch": n_ids_diff, "cand_ids_mismatch": n_cand_diff,
            "argmax_mismatch": n_argmax_diff, "max_abs_dlogp": max_dlogit, "max_abs_dp": max_dp,
            "p50_dp": dps[len(dps) // 2], "p99_dp": dps[int(0.99 * (len(dps) - 1))]}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"summary": summ, "rows": rows}, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
