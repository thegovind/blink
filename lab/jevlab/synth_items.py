"""Authored documents -> item rows; verified-item filter after blind teacher solves.

  python -m jevlab.synth_items items DOCS.jsonl OUT_ITEMS.jsonl OUT_PROB_ITEMS.jsonl
  python -m jevlab.synth_items keep ITEMS.jsonl SOLVES.jsonl OUT.jsonl [--dist]
"""
import json
import sys

NOUL = {"true": "yes", "false": "no"}


def doc_items(doc):
    for j, q in enumerate(doc["questions"]):
        qq = dict(q["question"])
        gold = q["expected"]
        target = q.get("distribution")
        if qq["type"] == "noul":
            gold = NOUL[gold]
            if target:
                target = {NOUL[k]: v for k, v in target.items()}
        yield {"id": f"{doc['doc_id']}:q{j}", "src": f"synth_{doc['spec']['family']}", "state": doc["state"], "question": qq,
               "gold": gold, "target": target, "meta": {"group": doc["doc_id"], "family": doc["spec"]["family"],
                                                        "domain": doc["spec"]["domain"], "split": doc["split"], "length": doc["spec"]["length"]}}


def tvd(p, q):
    return 0.5 * sum(abs(p.get(k, 0) - q.get(k, 0)) for k in set(p) | set(q))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "items":
        n = m = 0
        with open(sys.argv[3], "w") as f, open(sys.argv[4], "w") as g:
            for line in open(sys.argv[2]):
                for it in doc_items(json.loads(line)):
                    (g if it["target"] else f).write(json.dumps(it, ensure_ascii=False) + "\n")
                    m += bool(it["target"]); n += 1
        print(f"items {n} (probability {m})")
    elif cmd == "keep":
        dist = "--dist" in sys.argv
        solves = {}
        for line in open(sys.argv[3]):
            r = json.loads(line)
            solves[r["id"]] = r
        kept = tot = 0
        with open(sys.argv[4], "w") as f:
            for line in open(sys.argv[2]):
                it = json.loads(line)
                s = solves.get(it["id"])
                if not s:
                    continue
                tot += 1
                if dist:
                    ok = s["dist"] is not None and tvd(s["dist"], it["target"]) <= 0.03
                else:
                    ok = s["samples"] and s["samples"][0] == it["gold"]
                if ok:
                    kept += 1
                    f.write(json.dumps(it, ensure_ascii=False) + "\n")
        print(f"kept {kept}/{tot}")
