"""Gold-label training items from TRAIN/DEV splits of public datasets (never the suite's test splits).

Each builder yields items {id, src, state, question, gold, meta{group, stem}} in the same task format the
suite uses for that task type, so the model learns the task, not a new format. Suite test rows are
removed later by `dedup` against the frozen suite.
  python -m jevlab.data_gold OUTDIR
"""
from __future__ import annotations

import os
import csv
import io
import json
import random
import re
import sys
import zipfile
from pathlib import Path

RAW = Path(os.environ.get("DECISION_INDEX_RAW_DIR", "decision-index/work/artifacts/benchmark-suite/raw"))
SEED = "jevlab-gold-20260924"


def norm(s):
    return re.sub(r"\W+", " ", str(s).lower()).strip()


def item(src, sid, state, instructions, criteria, gold, group=None, **meta):
    return {"id": f"{src}:{sid}", "src": src, "state": state,
            "question": {"type": "choice", "instructions": instructions, "criteria": criteria}, "gold": gold,
            "meta": {"group": group or f"{src}:{sid}", **meta}}


def mc(src, sid, question, options, gold_idx, state="", group=None, **meta):
    order = list(range(len(options)))
    random.Random(f"{SEED}:{src}:{sid}").shuffle(order)
    keys = [chr(65 + i) for i in range(len(options))] if len(options) <= 26 else [f"option_{i}" for i in range(len(options))]
    return item(src, sid, state, question, {k: options[j] for k, j in zip(keys, order)}, keys[order.index(gold_idx)], group, **meta)


# ---------------- language / retrieval / arts (panel task types, train splits) ----------------

def esci_train(n=40000):
    import pyarrow.parquet as pq
    repo = RAW / "repos/esci/shopping_queries_dataset"
    ex = pq.read_table(repo / "shopping_queries_dataset_examples.parquet", filters=[("split", "=", "train")]).to_pylist()
    prods = {(r["product_locale"], r["product_id"]): r for r in pq.read_table(repo / "shopping_queries_dataset_products.parquet").to_pylist()}
    rng = random.Random(SEED + "esci")
    rng.shuffle(ex)
    crit = {"E": "Exact: the product satisfies the search query.", "S": "Substitute: a product that could substitute for the requested product.",
            "C": "Complement: a product that complements the requested product.", "I": "Irrelevant: the product does not address the requested product need."}
    out = []
    for r in ex[:n]:
        p = prods.get((r["product_locale"], r["product_id"]))
        if not p:
            continue
        product = {k.removeprefix("product_"): p[k] for k in ["product_title", "product_description", "product_bullet_point", "product_brand", "product_color"] if p.get(k) is not None}
        out.append(item("esci", r["example_id"], {"search_query": r["query"], "product": product},
                        "Classify the relevance of this product to the search query using the ESCI categories.", dict(crit), r["esci_label"],
                        group=f"esci:q{r['query_id']}", locale=r["product_locale"], query_id=r["query_id"]))
    return out


def isarcasm_train():
    base = RAW / "repos/isarcasm/train"
    out = []
    for path in sorted(base.glob("*.csv")):
        lang = "En" if "En" in path.name else ("Ar" if "Ar" in path.name else None)
        with path.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        for i, r in enumerate(rows):
            text = r.get("tweet") or r.get("text")
            if not text or r.get("sarcastic") not in {"0", "1"}:
                continue
            out.append(item(f"isarcasm_{lang}", i, text, "Is this text intended to be sarcastic?", {"no": "No", "yes": "Yes"},
                            "yes" if r["sarcastic"] == "1" else "no"))
            if r.get("rephrase") and r["sarcastic"] == "1" and r["rephrase"].strip():
                pair = [text, r["rephrase"]]
                j = random.Random(f"{SEED}:isc:{lang}:{i}").randint(0, 1)
                crit = {"text_0": pair[j], "text_1": pair[1 - j]}
                out.append(item(f"isarcasmC_{lang}", i, "", "Which of these two texts is the sarcastic one?", crit, "text_0" if j == 0 else "text_1",
                                group=f"isarcasm_{lang}:{i}"))
            labs = ["sarcasm", "irony", "satire", "understatement", "overstatement", "rhetorical_question"]
            if r["sarcastic"] == "1" and all(r.get(l) in {"0", "1"} for l in labs):
                for l in labs:
                    out.append(item(f"isarcasmB_{lang}", f"{i}:{l}", text, f'Does this text exhibit {l.replace("_", " ")}? Evaluate this category independently.',
                                    {"no": "No", "yes": "Yes"}, "yes" if r[l] == "1" else "no", group=f"isarcasm_{lang}:{i}"))
    return out


def vast_train():
    out = []
    for split in ("train", "dev"):
        p = RAW / f"vast/data/VAST/vast_{split}.csv"
        with p.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                lab = int(r["label"])
                prompt = f"Topic: {r['topic_str']}\nPost: {r['post']}\nDetermine the stance of the post toward the topic."
                out.append(mc("vast", f"{split}:{r['new_id']}", prompt, ["against", "favor", "neutral"], lab, group=f"vast:{norm(r['post'])[:80]}"))
    return out


def contractnli_train():
    zp = RAW / "repos/contractnli/resources/contract-nli.zip"
    out = []
    labels = {"Entailment": "entailment", "Contradiction": "contradiction", "NotMentioned": "not_mentioned"}
    with zipfile.ZipFile(zp) as z:
        names = z.namelist()
        for split in ("train", "dev"):
            member = next(n for n in names if n.endswith(f"{split}.json"))
            d = json.loads(z.read(member))
            hyps = d["labels"]
            for doc in d["documents"]:
                for hk, ann in doc["annotation_sets"][0]["annotations"].items():
                    hyp = hyps[hk]["hypothesis"]
                    choice = ann["choice"]
                    crit = {"entailment": "Entailment: the contract supports the hypothesis.",
                            "contradiction": "Contradiction: the contract contradicts the hypothesis.",
                            "not_mentioned": "Not mentioned: the contract does not address the hypothesis."}
                    out.append(item("contractnli", f"{split}:{doc['id']}:{hk}", doc["text"],
                                    f"Hypothesis: {hyp}\nDoes the contract entail, contradict, or not mention this hypothesis?",
                                    crit, labels[choice], group=f"contractnli:{doc['id']}"))
    return out


def humicroedit_train():
    zp = RAW / "downloads/humicroedit-full.zip"
    out = []
    with zipfile.ZipFile(zp) as z:
        for split in ("train", "dev"):
            member = f"semeval-2020-task-7-dataset/subtask-2/{split}.csv"
            for r in csv.DictReader(io.StringIO(z.read(member).decode("utf-8-sig"))):
                if r["label"] not in {"1", "2"}:
                    continue
                crit = {}
                for n in (1, 2):
                    rendered, c = re.subn(r"<[^<>]+/>", lambda _: r[f"edit{n}"], r[f"original{n}"])
                    if c != 1:
                        break
                    crit[f"headline_{n}"] = rendered
                if len(crit) == 2:
                    out.append(item("humicroedit", f"{split}:{r['id']}", "", "Which edited news headline is funnier?", crit,
                                    "headline_" + r["label"], group=f"hum:{norm(r['original1'])[:80]}"))
    return out


# ---------------- NLI / QA / intent (general decision skills) ----------------

def hf(name, config=None, split="train"):
    from datasets import load_dataset
    return load_dataset(name, config, split=split) if config else load_dataset(name, split=split)


def anli_train(n=30000):
    out = []
    lab = ["entailment", "neutral", "contradiction"]
    for rnd in ("train_r1", "train_r2", "train_r3"):
        for r in hf("facebook/anli", split=rnd):
            state = f"Premise: {r['premise']}\nHypothesis: {r['hypothesis']}"
            out.append(mc("anli", r["uid"], "What is the relationship between the premise and the hypothesis?", lab, r["label"], state=state))
    random.Random(SEED).shuffle(out)
    return out[:n]


def wanli_train(n=30000):
    lab = {"entailment": 0, "neutral": 1, "contradiction": 2}
    out = [mc("wanli", r["id"], "What is the relationship between the premise and the hypothesis?", ["entailment", "neutral", "contradiction"],
              lab[r["gold"]], state=f"Premise: {r['premise']}\nHypothesis: {r['hypothesis']}") for r in hf("alisawuffles/WANLI")]
    random.Random(SEED).shuffle(out)
    return out[:n]


def boolq_train():
    return [item("boolq", i, r["passage"], r["question"].rstrip("?") + "?", {"no": "No", "yes": "Yes"}, "yes" if r["answer"] else "no")
            for i, r in enumerate(hf("google/boolq"))]


def banking77_train():
    base = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/"
    import urllib.request
    cats = json.loads(urllib.request.urlopen(base + "categories.json").read())
    rows = list(csv.DictReader(io.StringIO(urllib.request.urlopen(base + "train.csv").read().decode())))
    return [item("banking77", i, {}, "Classify the banking intent of this user request:\n" + r["text"],
                 {f"option_{j}" if len(cats) > 26 else chr(65 + j): c for j, c in enumerate(cats)},
                 (f"option_{cats.index(r['category'])}" if len(cats) > 26 else chr(65 + cats.index(r["category"])))) for i, r in enumerate(rows)]


# ---------------- knowledge MCQ (non-test splits) ----------------

def knowledge_all():
    out = []
    for r in hf("cais/mmlu", "all", split="auxiliary_train"):
        out.append(mc("mmlu_aux", len(out), r["question"], r["choices"], r["answer"]))
    for cfg in ("ARC-Easy", "ARC-Challenge"):
        for split in ("train", "validation"):
            for r in hf("allenai/ai2_arc", cfg, split=split):
                if r["answerKey"] in r["choices"]["label"]:
                    out.append(mc("arc", f"{cfg}:{r['id']}", r["question"], r["choices"]["text"], r["choices"]["label"].index(r["answerKey"])))
    for r in hf("allenai/openbookqa", "main", split="train"):
        out.append(mc("obqa", r["id"], r["question_stem"], r["choices"]["text"], r["choices"]["label"].index(r["answerKey"])))
    for i, r in enumerate(hf("allenai/sciq")):
        out.append(mc("sciq", i, r["question"], [r["correct_answer"], r["distractor1"], r["distractor2"], r["distractor3"]], 0))
    for r in hf("tau/commonsense_qa"):
        if r["answerKey"]:
            out.append(mc("csqa", r["id"], r["question"], r["choices"]["text"], r["choices"]["label"].index(r["answerKey"])))
    med = list(hf("openlifescienceai/medmcqa"))
    random.Random(SEED).shuffle(med)
    for r in med[:40000]:
        if r["choice_type"] == "single":
            out.append(mc("medmcqa", r["id"], r["question"], [r["opa"], r["opb"], r["opc"], r["opd"]], r["cop"]))
    for i, r in enumerate(hf("deepmind/aqua_rat", "raw")):
        opts = [re.sub(r"^[A-E]\)\s*", "", o) for o in r["options"]]
        out.append(mc("aqua", i, r["question"], opts, "ABCDE".index(r["correct"])))
    return out


BUILDERS = {"esci": esci_train, "isarcasm": isarcasm_train, "vast": vast_train, "contractnli": contractnli_train,
            "humicroedit": humicroedit_train, "anli": anli_train, "wanli": wanli_train, "boolq": boolq_train,
            "banking77": banking77_train, "knowledge": knowledge_all}

if __name__ == "__main__":
    outdir = Path(sys.argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    names = sys.argv[2:] or list(BUILDERS)
    for name in names:
        try:
            rows = BUILDERS[name]()
        except Exception as e:  # noqa: BLE001
            print(f"{name}: FAILED {type(e).__name__}: {e}", flush=True)
            continue
        with open(outdir / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)}", flush=True)
