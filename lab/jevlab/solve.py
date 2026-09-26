"""Blind teacher solves for authored items (server mode).

  python -m jevlab.solve emit ITEMS.jsonl REQ.jsonl [--dist]
  python -m jevlab.solve keep ITEMS.jsonl REQ.jsonl RESP.jsonl OUT.jsonl [--dist]
An answer item is kept when the blind solve (shuffled option order) returns the intended key; a
probability item when the solved distribution is within TVD 0.03 of the author's exact distribution.
"""
import json
import random
import sys

from .render import question_options
from .teacher import parse, parse_dist, teacher_messages


def tvd(p, q):
    return 0.5 * sum(abs(p.get(k, 0) - q.get(k, 0)) for k in set(p) | set(q))


def emit(items, out, dist):
    rng = random.Random("solve")
    with open(out, "w") as f:
        for line in open(items):
            it = json.loads(line)
            k = len(question_options(it["question"]))
            order = list(range(k))
            if it["question"]["type"] != "score":
                rng.shuffle(order)
            msgs, labels, keys = teacher_messages(it["state"], it["question"], order, "dist" if dist else "answer")
            f.write(json.dumps({"id": it["id"], "messages": msgs, "labels": labels, "keys": keys, "thinking": True,
                                "max_tokens": 12288, "temperature": 0.6}, ensure_ascii=False) + "\n")


def keep(items, reqf, respf, out, dist):
    reqs = {json.loads(l)["id"]: json.loads(l) for l in open(reqf)}
    resps = {json.loads(l)["id"]: json.loads(l) for l in open(respf)}
    kept = tot = 0
    with open(out, "w") as f:
        for line in open(items):
            it = json.loads(line)
            r, q = resps.get(it["id"]), reqs.get(it["id"])
            if not r or not q:
                continue
            tot += 1
            if dist:
                d = parse_dist(r["text"], q["labels"], q["keys"])
                ok = d is not None and tvd(d, it["target"]) <= 0.03
            else:
                ok = parse(r["text"], q["labels"], q["keys"]) == it["gold"]
            if ok:
                kept += 1
                f.write(line)
    print(f"kept {kept}/{tot}")


if __name__ == "__main__":
    dist = "--dist" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--dist"]
    if args[0] == "emit":
        emit(args[1], args[2], dist)
    else:
        keep(args[1], args[2], args[3], args[4], dist)
