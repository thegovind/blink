"""Token census of the frozen suite under a renderer: prompts per request, tokens, option counts.

  python -m jevlab.census --rows ROWS.jsonl.gz --model Qwen/Qwen3.5-4B --template semif --procs 32
"""
import argparse
import collections
import gzip
import json
import multiprocessing as mp

_rd = None


def init(model, template):
    global _rd
    from transformers import AutoTokenizer
    from .render import Renderer
    _rd = Renderer(AutoTokenizer.from_pretrained(model), template=template)


def work(line):
    from .decide import build
    r = json.loads(line)
    e = r["_evaluation"]
    try:
        w = build(_rd, r["state"], r["questions"])
        lens = [len(x["prompt_ids"]) for x in w]
        ks = [len(x["cand_ids"]) for x in w]
        return e["dataset"], lens, ks, None
    except Exception as ex:  # noqa: BLE001
        return e["dataset"], [], [], f"{type(ex).__name__}: {ex}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--template", default="semif")
    ap.add_argument("--procs", type=int, default=32)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    op = gzip.open if a.rows.endswith(".gz") else open
    lines = list(op(a.rows, "rt"))
    agg = collections.defaultdict(lambda: {"requests": 0, "prompts": 0, "tokens": 0, "max_len": 0, "over32k": 0, "over64k": 0, "k_gt26": 0, "max_k": 0, "errors": 0})
    allens = []
    with mp.Pool(a.procs, initializer=init, initargs=(a.model, a.template)) as pool:
        for ds, lens, ks, err in pool.imap_unordered(work, lines, chunksize=64):
            d = agg[ds]
            d["requests"] += 1
            if err:
                d["errors"] += 1
                continue
            d["prompts"] += len(lens)
            d["tokens"] += sum(lens)
            d["max_len"] = max(d["max_len"], max(lens))
            d["over32k"] += sum(x > 32768 for x in lens)
            d["over64k"] += sum(x > 65536 for x in lens)
            d["k_gt26"] += sum(k > 26 for k in ks)
            d["max_k"] = max(d["max_k"], max(ks))
            allens += lens
    tot = {k: sum(d[k] for d in agg.values()) for k in ("requests", "prompts", "tokens", "over32k", "over64k", "k_gt26", "errors")}
    allens.sort()
    tot["p50_len"] = allens[len(allens) // 2]
    tot["p99_len"] = allens[int(len(allens) * 0.99)]
    tot["max_len"] = allens[-1]
    for ds in sorted(agg, key=lambda x: -agg[x]["tokens"]):
        d = agg[ds]
        print(f"{ds:28} req {d['requests']:6} prompts {d['prompts']:7} tok {d['tokens'] / 1e6:8.2f}M max {d['max_len']:7} >32k {d['over32k']:5} >64k {d['over64k']:4} K>26 {d['k_gt26']:6} maxK {d['max_k']:3} err {d['errors']}")
    print("TOTAL", json.dumps(tot))
    if a.out:
        json.dump({"total": tot, "by_dataset": agg}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
