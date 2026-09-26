"""Teacher pilot stats: python -m jevlab.pilot_stats FILE.jsonl [...]"""
import collections
import json
import sys

rows = [json.loads(l) for f in sys.argv[1:] for l in open(f)]
by = collections.defaultdict(lambda: collections.Counter())
toks = []
for r in rows:
    src = r["id"].split(":")[0]
    d = by[src]
    s, g = r["samples"], r["gold"]
    d["n"] += 1
    d["s1"] += s[0] == g
    d["any"] += g in s
    d["both"] += all(x == g for x in s)
    d["unparsed"] += all(x is None for x in s)
    d["tok"] += sum(r["gen_tokens"]) / len(r["gen_tokens"])
    d["trunc"] += sum(f == "length" for f in r["finish"])
    toks += r["gen_tokens"]
for k, d in by.items():
    n = d["n"]
    print(f"{k:10} n={n} acc1={d['s1'] / n:.3f} any={d['any'] / n:.3f} both={d['both'] / n:.3f} "
          f"unparsed={d['unparsed'] / n:.3f} avg_tok={d['tok'] / n:.0f} truncated={d['trunc']}")
toks.sort()
print("tokens p50", toks[len(toks) // 2], "p90", toks[int(0.9 * len(toks))], "max", toks[-1])
