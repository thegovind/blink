#!/usr/bin/env python3
"""Anchor rows on a Mac: label general decisions with a blink model's own probabilities, so a fine-tune keeps them.

  python examples/mlx/anchor_mlx.py --model thegovind/blink-4b --inp data/general.jsonl --out data/anchors.jsonl

The same rows as lab/jevlab/anchor.py writes with PyTorch: each input row's state and question, scored in its
given option order, with the model's distribution over the offered options as the soft "target". Training
against them is a KL pull toward the model you started from. Mix them into the training data (for example
--data data/train.jsonl,data/anchors.jsonl). Rows whose prompt is longer than --max-len are skipped.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from blink_mlx import REVISION, load  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", help="a blink model repo or a local model folder (default: thegovind/blink-4b, or "
                                    "the base model --adapter was trained on)")
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--adapter", help="an MLX LoRA adapter folder to anchor to instead of the plain model")
    ap.add_argument("--inp", required=True, help="JSONL rows with state and question (labels are ignored)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, help="use this many rows, drawn at random")
    ap.add_argument("--max-len", type=int, default=16384)
    a = ap.parse_args(argv)
    with open(a.inp, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    random.Random(0).shuffle(rows)
    rows = rows[: a.limit]
    engine = load(a.model, a.revision, temperature=1.0, adapter=a.adapter).engine
    n = 0
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        for r in rows:
            work = engine.render(r.get("state"), {"q": r["question"]})[0]
            if len(work["ids"]) > a.max_len:
                continue
            logits = engine.logits(r.get("state"), {"q": r["question"]})[0]["q"]
            keys = work["keys"]
            top = max(logits)
            z = sum(math.exp(x - top) for x in logits)
            target = {k: math.exp(x - top) / z for k, x in zip(keys, logits)}
            out = {k: r[k] for k in ("id", "state", "question") if k in r}
            out.update(id=f"anchor:{r.get('id', n)}", src="anchor_base", gold=max(target, key=target.get),
                       target=target, meta={"group": (r.get("meta") or {}).get("group", r.get("id", n)),
                                            "anchor_of": r.get("src")})
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1
    print(f"anchors {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
