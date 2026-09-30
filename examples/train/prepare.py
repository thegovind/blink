#!/usr/bin/env python3
"""Check and split your own decision data for blink's trainer.

  python examples/train/prepare.py my_rows.jsonl --out-dir data --dev-fraction 0.15
  python examples/train/prepare.py train_rows.jsonl --dev dev_rows.jsonl --out-dir data
  python examples/train/prepare.py my_rows.jsonl --out-dir data --dev-fraction 0.15 \\
      --model thegovind/blink-4b --max-len 4096      # also count prompt tokens

Each input line is one question about one state, in the trainer's format (README.md), with three
shortcuts:

  "votes":   {"refund": 3, "replace": 1}   annotator counts; become a soft "target"
  "outcome": "refund"                      what actually happened later; becomes "gold"
  "gold":    true / false, or an integer   yes/no and score levels, written the trainer's way

Rows are checked with the trainer's own rules. The split keeps each "group" (or, without one,
each identical state) on one side, and a dev file that repeats a training question is refused.
Writes OUT_DIR/train.jsonl and OUT_DIR/dev.jsonl and prints a short report.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lab"))
from jevlab.render import RenderError, question_options  # noqa: E402


def _where(row: dict, where: str) -> str:
    return f"{where} (id {row['id']})" if isinstance(row, dict) and "id" in row else where


def _digest(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]


def normalize(row: dict, where: str = "row") -> dict:
    """One row as the trainer wants it; raises ValueError naming the row when it cannot be."""
    if not isinstance(row, dict):
        raise ValueError(f"{where}: each line must be a JSON object")
    loc = _where(row, where)
    q = row.get("question")
    if not isinstance(q, dict):
        raise ValueError(f"{loc}: needs one 'question' object (not a 'questions' map)")
    try:
        keys = [k for k, _ in question_options(q)]
    except RenderError as exc:
        raise ValueError(f"{loc}: {exc}") from None
    out = dict(row)
    meta = dict(out.get("meta") or {})

    if "votes" in out:
        if "target" in out or "gold" in out:
            raise ValueError(f"{loc}: give votes, or target/gold, not both")
        votes = out.pop("votes")
        if not isinstance(votes, dict) or not votes:
            raise ValueError(f"{loc}: votes must map offered option keys to counts")
        out["target"] = {str(k): v for k, v in votes.items()}
        meta["label_from"] = "votes"
    if "outcome" in out:
        if "target" in out or "gold" in out:
            raise ValueError(f"{loc}: give outcome, or target/gold, not both")
        out["gold"] = out.pop("outcome")
        meta["label_from"] = "outcome"

    gold = out.get("gold")
    if q["type"] == "noul" and isinstance(gold, bool):
        out["gold"] = "yes" if gold else "no"
    elif q["type"] == "noul" and isinstance(gold, str) and gold.strip().lower() in ("yes", "no"):
        out["gold"] = gold.strip().lower()
    elif q["type"] == "score" and isinstance(gold, int) and not isinstance(gold, bool):
        out["gold"] = str(gold)

    if out.get("target") is not None:
        target = out["target"]
        if not isinstance(target, dict) or set(map(str, target)) - set(keys):
            raise ValueError(f"{loc}: target must use offered option keys {keys}")
        try:
            values = {str(k): float(v) for k, v in target.items()}
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"{loc}: target values must be numbers") from None
        total = sum(values.values())
        if any(not math.isfinite(v) or v < 0 for v in values.values()) or not math.isfinite(total) or total <= 0:
            raise ValueError(f"{loc}: target must be finite, nonnegative and not all zero")
        out["target"] = {k: values.get(k, 0.0) / total for k in keys if values.get(k, 0.0) > 0}
        out.pop("gold", None)
    elif out.get("gold") not in keys:
        raise ValueError(f"{loc}: gold {out.get('gold')!r} is not an offered option key {keys}")

    try:
        weight = float(out.get("weight", 1.0))
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{loc}: weight must be a number") from None
    if not math.isfinite(weight) or weight < 0:
        raise ValueError(f"{loc}: weight must be finite and nonnegative")
    if "state" not in out:
        raise ValueError(f"{loc}: needs a 'state' (text, object or array)")
    if meta:
        out["meta"] = meta
    if "id" not in out:
        out["id"] = "row-" + _digest([out["state"], q, out.get("gold"), out.get("target")])
    return out


def question_key(row: dict) -> str:
    """Same state and same question, whatever the label."""
    return _digest([row.get("state"), row["question"]])


def group_of(row: dict) -> str:
    group = row.get("group", (row.get("meta") or {}).get("group"))
    return str(group) if group is not None else "state-" + _digest(row.get("state"))


def split(rows: list[dict], fraction: float, seed: int = 0) -> tuple[list[dict], list[dict]]:
    """Whole groups go to dev until it holds about `fraction` of the rows."""
    if not 0 < fraction < 1:
        raise ValueError("--dev-fraction must be between 0 and 1")
    groups = collections.defaultdict(list)
    for row in rows:
        groups[group_of(row)].append(row)
    if len(groups) < 2:
        raise ValueError("need at least two groups (or distinct states) to split")
    names = sorted(groups)
    random.Random(seed).shuffle(names)
    want = fraction * len(rows)
    dev_names, n = set(), 0
    for name in names:
        if n >= want or len(dev_names) == len(names) - 1:
            break
        dev_names.add(name)
        n += len(groups[name])
    train = [r for name in names if name not in dev_names for r in groups[name]]
    dev = [r for name in names if name in dev_names for r in groups[name]]
    return train, dev


def load(path: str, where_prefix: str = "") -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            if not line.strip():
                continue
            where = f"{where_prefix}{path}:{i}"
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{where}: not JSON ({exc.msg})") from None
            rows.append(normalize(raw, where))
    return rows


def unique_ids(rows: list[dict]) -> int:
    seen, renamed = collections.Counter(), 0
    for row in rows:
        seen[row["id"]] += 1
        if seen[row["id"]] > 1:
            row["id"] = f"{row['id']}#{seen[row['id']]}"
            renamed += 1
    return renamed


def report(name: str, rows: list[dict]) -> list[str]:
    types = collections.Counter(r["question"]["type"] for r in rows)
    srcs = collections.Counter(r.get("src", "-") for r in rows)
    soft = sum(1 for r in rows if r.get("target") is not None)
    froms = collections.Counter((r.get("meta") or {}).get("label_from", "gold/target") for r in rows)
    lines = [f"{name}: {len(rows)} rows, {len({group_of(r) for r in rows})} groups",
             "  types: " + ", ".join(f"{k} {v}" for k, v in sorted(types.items())),
             "  sources: " + ", ".join(f"{k} {v}" for k, v in srcs.most_common(8)),
             f"  soft targets: {soft}; labels from: " + ", ".join(f"{k} {v}" for k, v in sorted(froms.items()))]
    golds = collections.Counter(r["gold"] for r in rows if r.get("target") is None)
    if golds:
        lines.append("  most common gold labels: " + ", ".join(f"{k} {v}" for k, v in golds.most_common(6)))
    return lines


def conflicts(rows: list[dict]) -> tuple[int, int]:
    """(repeated questions, repeated questions whose labels disagree)."""
    labels = collections.defaultdict(set)
    for row in rows:
        labels[question_key(row)].add(json.dumps([row.get("gold"), row.get("target")], sort_keys=True))
    counts = collections.Counter(question_key(r) for r in rows)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    disagree = sum(1 for k, v in labels.items() if len(v) > 1)
    return repeated, disagree


def token_lengths(rows: list[dict], model: str, max_len: int) -> list[str]:
    from transformers import AutoTokenizer

    from jevlab.render import Renderer

    tok = AutoTokenizer.from_pretrained(model)
    rd = Renderer(tok, template="semif")
    lengths = sorted(len(tok(rd.render(r.get("state"), r["question"])[0], add_special_tokens=False)["input_ids"])
                     for r in rows)
    over = sum(1 for n in lengths if n > max_len)
    p95 = lengths[min(len(lengths) - 1, int(0.95 * len(lengths)))]
    return [f"prompt tokens: median {lengths[len(lengths) // 2]}, p95 {p95}, max {lengths[-1]}; "
            f"{over} over --max-len {max_len} (the trainer skips those)"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("inputs", nargs="+", help="JSONL files of training rows")
    ap.add_argument("--out-dir", required=True)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--dev", help="a JSONL file of development rows you already hold out")
    group.add_argument("--dev-fraction", type=float, help="hold out about this share of rows, by group")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", help="tokenizer to count prompt tokens with, e.g. thegovind/blink-4b")
    ap.add_argument("--max-len", type=int, default=4096)
    a = ap.parse_args(argv)
    try:
        rows = [r for path in a.inputs for r in load(path)]
        if a.dev:
            train, dev = rows, load(a.dev)
            leaked = {question_key(r) for r in train} & {question_key(r) for r in dev}
            shared = {group_of(r) for r in train} & {group_of(r) for r in dev}
            if leaked or shared:
                bad = [r["id"] for r in dev if question_key(r) in leaked or group_of(r) in shared]
                raise ValueError(f"{len(bad)} dev rows repeat a training question or group, e.g. {bad[:5]}")
        else:
            train, dev = split(rows, a.dev_fraction, a.seed)
        if not train or not dev:
            raise ValueError("both the training and the development set need rows")
    except (OSError, ValueError) as exc:
        print(f"prepare: {exc}", file=sys.stderr)
        return 1
    renamed = unique_ids(train + dev)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train), ("dev", dev)):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as fh:
            for row in part:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    lines = report("train", train) + report("dev", dev)
    repeated, disagree = conflicts(train)
    lines.append(f"train repeats {repeated} questions; {disagree} of them with different labels")
    if renamed:
        lines.append(f"{renamed} repeated ids were made unique with a #n suffix")
    if a.model:
        lines += token_lengths(train + dev, a.model, a.max_len)
    lines.append(f"wrote {out / 'train.jsonl'} and {out / 'dev.jsonl'}")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
