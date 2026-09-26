"""Contamination filter: drop training items that overlap the frozen suite (all 37 benchmarks) or JevBench items.

Two tests:
 * exact: any normalised string (>= 30 chars) of the item (state, instructions, option texts, recursively)
   equals a normalised string anywhere in a suite request or a JevBench public item;
 * shingle (question-style sources only): the item's question stem shares a 13-word shingle with any suite
   question stem / GSM8K problem / CRUXEval code (catches edited copies, e.g. MMLU-Pro vs MMLU).
  python -m jevlab.dedup --suite ROWS.jsonl.gz --inp IN_DIR --out OUT_DIR [--shingle-srcs knowledge,gpqa_ext,...]
"""
from __future__ import annotations

import os
import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path

JB = Path(os.environ.get("JEVBENCH_PUBLIC_DIR", "jevbench/datasets/public"))
W = 13


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", str(s).lower())).strip()


def h(s):
    return hashlib.blake2b(s.encode(), digest_size=8).digest()


def strings(x, out):
    if isinstance(x, str):
        n = norm(x)
        if len(n) >= 30:
            out.append(n)
    elif isinstance(x, dict):
        for v in x.values():
            strings(v, out)
    elif isinstance(x, list):
        for v in x:
            strings(v, out)
    return out


def shingles(text):
    w = norm(text).split()
    return {h(" ".join(w[i: i + W])) for i in range(0, max(0, len(w) - W + 1))}


def stem_texts(state, questions):
    out = []
    for q in questions.values():
        ins = q.get("instructions")
        if isinstance(ins, str):
            out.append(ins)
    if isinstance(state, dict):
        for k in ("question", "code"):
            if isinstance(state.get(k), str):
                out.append(state[k])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shingle-files", default="knowledge,gpqa_ext,mmlu_pro,supergpqa,gsm,crux,cladder,math")
    a = ap.parse_args()
    import collections
    cnt, shc = collections.Counter(), collections.Counter()
    op = gzip.open if a.suite.endswith(".gz") else open
    n = 0
    with op(a.suite, "rt") as f:
        for line in f:
            r = json.loads(line)
            for s in set(strings(r["state"], []) + strings(r["questions"], [])):
                cnt[h(s)] += 1
            for x in set().union(*[shingles(t) for t in stem_texts(r["state"], r["questions"])] or [set()]):
                shc[x] += 1
            n += 1
    # strings shared by >= 20 suite rows are task templates (fixed instructions, option descriptions), not item content
    exact = {k for k, c in cnt.items() if c < 20}
    sh = {k for k, c in shc.items() if c < 20}  # shingles in >= 20 rows are instruction templates
    print(f"template strings ignored: {sum(1 for c in cnt.values() if c >= 20)}", flush=True)
    for p in JB.glob("*.jsonl"):
        for line in open(p):
            r = json.loads(line)
            for s in strings(r["state"], []) + strings(r["question"], []):
                exact.add(h(s))
            if isinstance(r["question"].get("instructions"), str):
                sh |= shingles(r["question"]["instructions"])
    print(f"suite rows {n}: {len(exact)} exact strings, {len(sh)} shingles", flush=True)
    shingle_files = set(a.shingle_files.split(","))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    report = {}
    for p in sorted(Path(a.inp).glob("*.jsonl")):
        use_sh = p.stem in shingle_files
        kept = dropped_exact = dropped_sh = 0
        with open(Path(a.out) / p.name, "w") as g:
            for line in open(p):
                r = json.loads(line)
                ss = strings(r.get("state"), []) + strings(r["question"], [])
                if any(h(s) in exact for s in ss):
                    dropped_exact += 1
                    continue
                if use_sh:
                    q = {"q": r["question"]}
                    if any(shingles(t) & sh for t in stem_texts(r.get("state"), q)):
                        dropped_sh += 1
                        continue
                kept += 1
                g.write(line)
        report[p.stem] = {"kept": kept, "dropped_exact": dropped_exact, "dropped_shingle": dropped_sh}
        print(p.stem, report[p.stem], flush=True)
    (Path(a.out) / "dedup-report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
