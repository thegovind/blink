"""Contamination audit of the FINAL training mixtures (every row, authored rows and base-model anchors included).

Uses jevlab.dedup's own normalisation, hashing and template rule, but applies both tests to every row of every
mixture (the pre-mix filter applied the shingle test to question-style sources only), and keeps suite hits and
JevBench-public hits apart. Writes aggregate counts only (no item text); row ids of hits go to a local-only file.

  python -m jevlab.dedup_audit --suite $BLINK_WORKDIR/kit/suite/selected-rows.jsonl.gz --out $BLINK_WORKDIR/runs/dedup-audit.json \
      --mix $BLINK_WORKDIR/data/mix/t3_train.jsonl ...
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
from pathlib import Path

from jevlab.dedup import JB, h, shingles, stem_texts, strings


def suite_index(path):
    cnt, shc = collections.Counter(), collections.Counter()
    n = 0
    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            for s in set(strings(r["state"], []) + strings(r["questions"], [])):
                cnt[h(s)] += 1
            for x in set().union(*[shingles(t) for t in stem_texts(r["state"], r["questions"])] or [set()]):
                shc[x] += 1
            n += 1
    # same template rule as jevlab.dedup: strings or shingles in >= 20 suite rows are task templates
    return n, {k for k, c in cnt.items() if c < 20}, {k for k, c in shc.items() if c < 20}


def jevbench_index():
    exact, sh, n = set(), set(), 0
    for p in JB.glob("*.jsonl"):
        for line in open(p):
            r = json.loads(line)
            n += 1
            for s in strings(r["state"], []) + strings(r["question"], []):
                exact.add(h(s))
            for t in stem_texts(r["state"], {"q": r["question"]}):
                sh |= shingles(t)
    return n, exact, sh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--mix", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    n_suite, s_exact, s_sh = suite_index(a.suite)
    n_jb, j_exact, j_sh = jevbench_index()
    report = {"suite_rows": n_suite, "jevbench_public_items": n_jb, "tests": {
        "exact": "a normalised string of >= 30 characters anywhere in the row (state, instructions, options) equals one in a suite request / public JevBench item (suite strings in >= 20 rows are templates and ignored)",
        "shingle": "the row's question stem (instructions, state.question, state.code) shares a 13-word shingle with a suite question stem / public JevBench instruction or state (suite shingles in >= 20 rows ignored)"},
        "mixtures": {}}
    hits_ids = {}
    for p in a.mix:
        name = Path(p).stem
        tot = collections.Counter()
        by_src = collections.defaultdict(collections.Counter)
        ids = collections.defaultdict(list)
        for line in open(p):
            r = json.loads(line)
            src = str(r.get("src"))
            q = {"q": r["question"]}
            ss = {h(s) for s in strings(r.get("state"), []) + strings(r["question"], [])}
            shs = set().union(*[shingles(t) for t in stem_texts(r.get("state"), q)] or [set()])
            flags = {"suite_exact": bool(ss & s_exact), "suite_shingle": bool(shs & s_sh),
                     "jevbench_exact": bool(ss & j_exact), "jevbench_shingle": bool(shs & j_sh)}
            tot["rows"] += 1
            by_src[src]["rows"] += 1
            for k, v in flags.items():
                if v:
                    tot[k] += 1
                    by_src[src][k] += 1
                    ids[k].append(r.get("id"))
        report["mixtures"][name] = {"totals": dict(tot),
                                    "sources_with_hits": {s: dict(c) for s, c in sorted(by_src.items()) if len(c) > 1}}
        hits_ids[name] = {k: v for k, v in ids.items()}
        print(name, dict(tot), flush=True)
    Path(a.out).write_text(json.dumps(report, indent=1))
    Path(a.out).with_suffix(".hit-ids.json").write_text(json.dumps(hits_ids))
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
