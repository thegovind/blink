"""Per-source DEV evaluation (accuracy, NLL, Brier, ECE) for a base model or base+adapter.

  python -m jevlab.dev_run --model M [--adapter A] --dev DEV.jsonl --out OUT.json [--temperature T]
Uses the canonical option order (no permutation) and the fixed semif rendering.
"""
import argparse
import collections
import json
import math
from pathlib import Path

from .render import Renderer
from .scorer import Scorer, load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--template", default="semif")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-len", type=int, default=32768)
    ap.add_argument("--device", default="cuda", help="cuda, or cpu for small checks")
    a = ap.parse_args()
    tok, model = load_model(a.model, adapter=a.adapter, device=a.device)
    sc = Scorer(model, tok, temperature=a.temperature, max_tokens=a.max_len)
    rd = Renderer(tok, template=a.template)
    rows = [json.loads(l) for l in open(a.dev)]
    work = []
    for r in rows:
        prompt, keys, cand = rd.render(r.get("state"), r["question"])
        ids = tok(prompt, add_special_tokens=False)["input_ids"]
        tgt = r.get("target") or {r["gold"]: 1.0}
        work.append((r, keys, ids, cand, tgt))
    lps = sc.score([w[2] for w in work], [w[3] for w in work])
    per = collections.defaultdict(lambda: {"n": 0, "acc": 0.0, "nll": 0.0, "brier": 0.0, "conf": []})
    recs = []
    for (r, keys, ids, cand, tgt), lp in zip(work, lps):
        if lp is None:
            continue
        p = [math.exp(float(x)) for x in lp]
        z = sum(p)
        p = [x / z for x in p]
        t = [float(tgt.get(k, 0.0)) for k in keys]
        pred = keys[max(range(len(p)), key=lambda i: p[i])]
        gold = max(tgt, key=tgt.get)
        d = per[r.get("src", "?")]
        d["n"] += 1
        d["acc"] += pred == gold
        d["nll"] += -sum(ti * math.log(max(pi, 1e-12)) for ti, pi in zip(t, p))
        d["brier"] += sum((pi - ti) ** 2 for pi, ti in zip(p, t))
        d["conf"].append((max(p), pred == gold))
        recs.append({"id": r["id"], "src": r.get("src"), "pred": pred, "gold": gold, "p": dict(zip(keys, p))})

    def ece(pairs):
        bins = [[0, 0.0, 0] for _ in range(10)]
        for c, ok in pairs:
            b = bins[min(int(c * 10), 9)]
            b[0] += 1; b[1] += c; b[2] += ok
        n = sum(b[0] for b in bins)
        return sum(b[0] / n * abs(b[2] / b[0] - b[1] / b[0]) for b in bins if b[0]) if n else None

    summ = {}
    allc = []
    for s, d in sorted(per.items()):
        n = d["n"]
        summ[s] = {"n": n, "acc": d["acc"] / n, "nll": d["nll"] / n, "brier": d["brier"] / n, "ece": ece(d["conf"])}
        allc += d["conf"]
    macro = sum(v["acc"] for v in summ.values()) / len(summ)
    out = {"model": a.model, "adapter": a.adapter, "temperature": a.temperature, "macro_acc": macro,
           "overall_ece": ece(allc), "by_src": summ}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": out, "records": recs}, open(a.out, "w"))
    print(f"macro_acc {macro:.4f} ece {out['overall_ece']:.4f}")
    for s, v in summ.items():
        print(f"  {s:16} n={v['n']:4} acc={v['acc']:.3f} nll={v['nll']:.3f} brier={v['brier']:.3f} ece={v['ece']:.3f}")


if __name__ == "__main__":
    main()
