"""LocalLLaMA/typed-decisions: build blink request rows from the test split, and score blink outputs against it.

  python lab/typed_decisions.py build --out typesafe/typed-decisions/            # requests.jsonl, gold.jsonl, meta.json
  python lab/typed_decisions.py score --gold .../gold.jsonl --outputs runs.jsonl [--name blink-4b]
  python lab/typed_decisions.py references --out .../                           # Uniform and Prior rows, to check metrics

The card publishes no scorer, so every metric is defined here and checked against the card's reference rows, which
need no model. KL, TV and Brier reproduce the card's Uniform row exactly (0.444 / 0.381 / 0.238). Accuracy is
unambiguous for untied predictions; the card's tie handling and its ECE and macro-F1 definitions are not published,
so those three are ours and not verified. Full distributions are kept so the maintainers can rescore them their way.
Generalist mode only: blink never trains on this dataset, train split included.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

REPO = "LocalLLaMA/typed-decisions"


def _load(split: str, revision: str | None):
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq

    sha = HfApi().dataset_info(REPO, revision=revision).sha
    path = hf_hub_download(REPO, f"all/{split}-00000-of-00001.parquet", repo_type="dataset", revision=sha)
    return sha, pq.read_table(path).to_pylist()


def build(a) -> None:
    sha, rows = _load("test", a.revision)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    n_q = 0
    with (out / "requests.jsonl").open("w") as rq, (out / "gold.jsonl").open("w") as gd:
        for r in rows:
            questions = json.loads(r["questions"])
            n_q += len(questions)
            rq.write(json.dumps({"id": r["id"], "state": json.loads(r["state"]), "questions": questions}) + "\n")
            gd.write(json.dumps({"id": r["id"], "workflow": r["workflow"], "gold": json.loads(r["gold"])}) + "\n")
    meta = {"dataset": REPO, "revision": sha, "split": "test", "cases": len(rows), "decisions": n_q}
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta))


# --- metrics ------------------------------------------------------------------------


def _argmax(p: dict) -> str:
    keys = list(p)
    return max(keys, key=lambda k: (p[k], -keys.index(k)))


def _pred_dist(qtype: str, answer: dict, gold_keys: list[str]) -> dict:
    """blink's answer in the gold's label space and key order."""
    if qtype == "noul":
        pt = float(answer["noul"])
        p = {"true": pt, "false": 1.0 - pt}
    else:
        p = {str(k): float(v) for k, v in answer["probabilities"].items()}
    missing = [k for k in gold_keys if k not in p]
    if missing:
        raise ValueError(f"prediction lacks labels {missing}")
    return {k: p[k] for k in gold_keys}


def _ece(conf: list[float], hit: list[bool], bins: int = 10) -> float:
    tot, n = 0.0, len(conf)
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(conf) if (lo < c <= hi) or (b == 0 and c == 0.0)]
        if idx:
            acc = sum(hit[i] for i in idx) / len(idx)
            cf = sum(conf[i] for i in idx) / len(idx)
            tot += len(idx) / n * abs(acc - cf)
    return tot


def _macro_f1(pairs: list[tuple[str, str]]) -> float:
    labels = sorted({g for g, _ in pairs} | {p for _, p in pairs})
    f1s = []
    for lab in labels:
        tp = sum(1 for g, p in pairs if g == lab and p == lab)
        fp = sum(1 for g, p in pairs if g != lab and p == lab)
        fn = sum(1 for g, p in pairs if g == lab and p != lab)
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return sum(f1s) / len(f1s)


def metrics(decisions: list[dict]) -> dict:
    """decisions: {workflow, qid, qtype, gold: {label: p}, label, pred: {label: p}} in gold key order."""
    acc, soft, kl, tv, brier, brier_label, conf, hit = [], [], [], [], [], [], [], []
    mae, within1 = [], []
    by_schema = collections.defaultdict(list)
    by_type = collections.defaultdict(list)
    for d in decisions:
        g, p, lab = d["gold"], d["pred"], d["label"]
        top = _argmax(p)
        ok = top == lab
        acc.append(ok)
        by_type[d["qtype"]].append(ok)
        soft.append(g[top])
        kl.append(sum(gk * math.log(gk / max(p[k], 1e-12)) for k, gk in g.items() if gk > 0))
        tv.append(0.5 * sum(abs(g[k] - p[k]) for k in g))
        brier.append(sum((p[k] - g[k]) ** 2 for k in g))
        brier_label.append(sum((p[k] - (1.0 if k == lab else 0.0)) ** 2 for k in g))
        conf.append(p[top])
        hit.append(ok)
        by_schema[(d["workflow"], d["qid"])].append((lab, top))
        if d["qtype"] == "score":
            ev_p = sum(int(k) * v for k, v in p.items())
            ev_g = sum(int(k) * v for k, v in g.items())
            mae.append(abs(ev_p - ev_g))
            within1.append(abs(int(top) - int(lab)) <= 1)
    mean = lambda xs: sum(xs) / len(xs) if xs else None  # noqa: E731
    return {
        "decisions": len(decisions),
        "acc": mean(acc),
        "soft_acc": mean(soft),
        "macro_f1": mean([_macro_f1(v) for v in by_schema.values()]),
        "kl": mean(kl),
        "tv": mean(tv),
        "brier": mean(brier),
        "brier_vs_label": mean(brier_label),
        "ece": _ece(conf, hit),
        "score_mae": mean(mae),
        "within_1_level": mean(within1),
        "acc_by_type": {t: mean(v) for t, v in sorted(by_type.items())},
        "definitions": {
            "acc": "argmax of the prediction equals the gold label (ties: first key)",
            "soft_acc": "gold probability of the predicted label",
            "macro_f1": "per question schema (workflow, question id), macro F1 over labels, then the mean",
            "kl": "KL(gold || prediction), natural log, prediction floored at 1e-12",
            "tv": "total variation, 0.5 * sum |gold - prediction|",
            "brier": "sum over labels of (prediction - gold probability)^2",
            "brier_vs_label": "sum over labels of (prediction - one-hot gold label)^2",
            "ece": "top-label confidence vs argmax correctness, 10 equal-width bins",
            "score_mae": "score questions: |expected level of prediction - expected level of gold|",
            "within_1_level": "score questions: |argmax level - gold label level| <= 1",
        },
    }


def _gold_rows(path: str):
    with open(path) as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _decisions_from(gold_path: str, pred_for) -> tuple[list[dict], list[str]]:
    out, missing = [], []
    for r in _gold_rows(gold_path):
        for qid, g in r["gold"].items():
            gold = {str(k): float(v) for k, v in g["probabilities"].items()}
            pred = pred_for(r, qid, g, list(gold))
            if pred is None:
                missing.append(f"{r['id']}/{qid}")
                continue
            out.append({"workflow": r["workflow"], "qid": qid, "qtype": g["type"], "gold": gold,
                        "label": str(g["label"]), "pred": pred})
    return out, missing


def score(a) -> None:
    outputs = {}
    with open(a.outputs) as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                outputs[row["id"]] = row.get("output")

    def pred_for(r, qid, g, keys):
        out = outputs.get(r["id"])
        if not out or qid not in out.get("answers", {}):
            return None
        return _pred_dist(g["type"], out["answers"][qid], keys)

    decisions, missing = _decisions_from(a.gold, pred_for)
    m = metrics(decisions)
    m["name"] = a.name
    m["unanswered"] = len(missing)
    if missing:
        m["unanswered_examples"] = missing[:10]
    print(json.dumps(m, indent=1))
    if a.save:
        Path(a.save).write_text(json.dumps(m, indent=1) + "\n")


def references(a) -> None:
    """The card's Uniform and Prior rows, recomputed with these metric definitions."""
    _, train = _load("train", a.revision)
    freq = collections.defaultdict(collections.Counter)
    for r in train:
        for qid, g in json.loads(r["gold"]).items():
            freq[(r["workflow"], qid)][str(g["label"])] += 1
    out = Path(a.out)
    rows = {}
    for name in ("uniform", "prior"):
        def pred_for(r, qid, g, keys, name=name):
            if name == "uniform":
                return {k: 1.0 / len(keys) for k in keys}
            c = freq[(r["workflow"], qid)]
            n = sum(c.values())
            return {k: c.get(k, 0) / n for k in keys}

        decisions, _ = _decisions_from(str(out / "gold.jsonl"), pred_for)
        rows[name] = metrics(decisions)
    card = {"uniform": {"acc": 0.308, "soft_acc": 0.311, "macro_f1": 0.152, "kl": 0.444, "tv": 0.381,
                        "brier": 0.238, "ece": 0.169},
            "prior": {"acc": 0.470, "soft_acc": 0.430, "macro_f1": 0.207, "kl": 0.347, "tv": 0.317,
                      "brier": 0.189, "ece": 0.088}}
    for name, want in card.items():
        got = rows[name]
        print(name, " ".join(f"{k}={got[k]:.3f}(card {v:.3f})" for k, v in want.items()),
              f"brier_vs_label={got['brier_vs_label']:.3f}")
    (out / "references.json").write_text(json.dumps(rows, indent=1) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--revision", default=None)
    s = sub.add_parser("score")
    s.add_argument("--gold", required=True)
    s.add_argument("--outputs", required=True)
    s.add_argument("--name", default="model")
    s.add_argument("--save", default=None)
    r = sub.add_parser("references")
    r.add_argument("--out", required=True)
    r.add_argument("--revision", default=None)
    a = ap.parse_args()
    {"build": build, "score": score, "references": references}[a.cmd](a)


if __name__ == "__main__":
    main()
