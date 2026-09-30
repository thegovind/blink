"""KL(base || student) anchor rows: score prompts with the frozen base model and store its
restricted-label distribution as the soft target (CE against it = KL + const).

  python -m jevlab.anchor --model Qwen/Qwen3.5-4B --inp ITEMS.jsonl --out ANCHORS.jsonl [--limit N]
Option order is the canonical one; train.py re-permutes and remaps targets by key.
"""
import argparse
import json
import math
import random
from pathlib import Path

from .render import Renderer
from .scorer import Scorer, load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--template", default="semif")
    ap.add_argument("--device", default="cuda", help="cuda, or cpu for small checks")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.inp)]
    random.Random(0).shuffle(rows)
    rows = rows[: a.limit]
    tok, model = load_model(a.model, device=a.device)
    sc = Scorer(model, tok, max_tokens=16384)
    rd = Renderer(tok, template=a.template)
    work = []
    for r in rows:
        prompt, keys, cand = rd.render(r.get("state"), r["question"])
        work.append((r, keys, tok(prompt, add_special_tokens=False)["input_ids"], cand))
    lps = sc.score([w[2] for w in work], [w[3] for w in work])
    n = 0
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for (r, keys, _, _), lp in zip(work, lps):
            if lp is None:
                continue
            p = [math.exp(float(x)) for x in lp]
            z = sum(p)
            tgt = {k: v / z for k, v in zip(keys, p)}
            out = {k: r[k] for k in ("id", "state", "question") if k in r}
            out.update(id="anchor:" + r["id"], src="anchor_base", gold=max(tgt, key=tgt.get), target=tgt,
                       meta={"group": (r.get("meta") or {}).get("group", r["id"]), "anchor_of": r.get("src")})
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1
    print("anchors", n)


if __name__ == "__main__":
    main()
