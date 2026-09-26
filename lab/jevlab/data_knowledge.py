"""Knowledge MCQ builders (non-suite sources) -> item rows {id, source, state, question, gold, meta}."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import re
import zipfile

SEED = "jevlab-20260924"


def norm(s):
    return re.sub(r"\W+", " ", str(s).lower()).strip()


def mc_item(source, sid, question, options, gold_idx, state="", shuffle=True, **meta):
    order = list(range(len(options)))
    if shuffle:
        random.Random(f"{SEED}:{source}:{sid}").shuffle(order)
    keys = [chr(65 + i) for i in range(len(options))] if len(options) <= 26 else [f"option_{i}" for i in range(len(options))]
    crit = {k: options[j] for k, j in zip(keys, order)}
    gold = keys[order.index(gold_idx)]
    return {"id": f"{source}:{sid}", "source": source, "state": state,
            "question": {"type": "choice", "instructions": question, "criteria": crit}, "gold": gold,
            "meta": {"stem_hash": hashlib.sha256(norm(question).encode()).hexdigest()[:16], **meta}}


def gpqa_extended_minus_diamond(zip_path):
    pw = b"deserted-untie-orchid"
    with zipfile.ZipFile(zip_path) as z:
        ext = list(csv.DictReader(io.StringIO(z.read("dataset/gpqa_extended.csv", pwd=pw).decode("utf-8-sig"))))
        dia = list(csv.DictReader(io.StringIO(z.read("dataset/gpqa_diamond.csv", pwd=pw).decode("utf-8-sig"))))
    dstems = {norm(r["Question"]) for r in dia}
    out = []
    for i, r in enumerate(ext):
        if norm(r["Question"]) in dstems:
            continue
        opts = [r["Correct Answer"]] + [r[f"Incorrect Answer {n}"] for n in (1, 2, 3)]
        out.append(mc_item("gpqa_ext", r.get("Record ID") or i, r["Question"], opts, 0,
                           domain=r.get("High-level domain"), subdomain=r.get("Subdomain")))
    return out, len(ext), len(dia)


def mmlu_pro(split="test"):
    from datasets import load_dataset
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split=split)
    out = []
    for r in ds:
        if str(r.get("src", "")).startswith("ori_mmlu"):
            continue
        opts = [o for o in r["options"] if str(o).strip() and str(o).strip() != "N/A"]
        if r["answer_index"] >= len(opts) or opts[r["answer_index"]] != r["options"][r["answer_index"]]:
            continue
        out.append(mc_item("mmlu_pro", r["question_id"], r["question"], opts, r["answer_index"],
                           category=r.get("category"), src=r.get("src")))
    return out


def supergpqa():
    from datasets import load_dataset
    ds = load_dataset("m-a-p/SuperGPQA", split="train")
    out = []
    for r in ds:
        opts = r["options"]
        gi = ord(r["answer_letter"]) - 65
        if not 0 <= gi < len(opts):
            continue
        out.append(mc_item("supergpqa", r["uuid"], r["question"], opts, gi, discipline=r.get("discipline"),
                           field=r.get("field"), difficulty=r.get("difficulty")))
    return out


if __name__ == "__main__":
    import sys
    outdir = sys.argv[1]
    gp, n_ext, n_dia = gpqa_extended_minus_diamond("data/sources/gpqa/dataset.zip")
    print("gpqa ext", n_ext, "diamond", n_dia, "kept", len(gp))
    mp = mmlu_pro()
    print("mmlu_pro non-ori", len(mp))
    sg = supergpqa()
    print("supergpqa", len(sg))
    for name, rows in (("gpqa_ext", gp), ("mmlu_pro", mp), ("supergpqa", sg)):
        with open(f"{outdir}/{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
