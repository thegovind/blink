"""Fit one temperature (optionally one per question type) on dev decisions and report calibration before/after.

  python fit_temperature.py --decisions dev.jsonl [--per-type] [--apply other.jsonl ...] [--out fit.json]

A decisions row is {"id", "q", "type": choice|noul|score, "keys": [...], "p": [...], "label": key}: blink's
probabilities at T=1 over the offered options in request order, plus the gold option. Adapters that turn each
source (Decision Index kit results, JevBench items, typed-decisions) into rows live beside their runners.

Temperature scaling p_T ∝ p^(1/T) is exactly softmax(logits / T), so it is applied to saved probabilities with no
GPU. T is fitted by minimising the mean negative log-likelihood of the gold option (golden-section search on log T
in [0.25, 8]). The argmax of every question is unchanged; a score's expected level and every confidence change.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path


def load(path: str) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def scaled(p: list[float], T: float) -> list[float]:
    if T == 1.0:
        return p
    logs = [math.log(max(x, 1e-300)) / T for x in p]
    m = max(logs)
    e = [math.exp(v - m) for v in logs]
    s = sum(e)
    return [v / s for v in e]


def stats(rows: list[dict], T_of) -> dict:
    nll, brier, conf, hit = [], [], [], []
    for r in rows:
        p = scaled(r["p"], T_of(r))
        y = r["keys"].index(r["label"])
        top = max(range(len(p)), key=lambda i: (p[i], -i))
        nll.append(-math.log(max(p[y], 1e-300)))
        brier.append(sum((pi - (1.0 if i == y else 0.0)) ** 2 for i, pi in enumerate(p)))
        conf.append(p[top])
        hit.append(top == y)
    ece, n = 0.0, len(rows)
    for b in range(10):
        idx = [i for i, c in enumerate(conf) if b / 10 < c <= (b + 1) / 10 or (b == 0 and c == 0)]
        if idx:
            ece += len(idx) / n * abs(sum(hit[i] for i in idx) / len(idx) - sum(conf[i] for i in idx) / len(idx))
    mean = lambda xs: sum(xs) / len(xs) if xs else None  # noqa: E731
    return {"n": n, "acc": mean(hit), "mean_conf": mean(conf), "ece": ece, "nll": mean(nll), "brier": mean(brier)}


def fit(rows: list[dict]) -> float:
    f = lambda lt: stats(rows, lambda r: math.exp(lt))["nll"]  # noqa: E731
    lo, hi = math.log(0.25), math.log(8.0)
    g = (math.sqrt(5) - 1) / 2
    c, d = hi - g * (hi - lo), lo + g * (hi - lo)
    fc, fd = f(c), f(d)
    for _ in range(60):
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - g * (hi - lo)
            fc = f(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + g * (hi - lo)
            fd = f(d)
    return round(math.exp((lo + hi) / 2), 4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", required=True)
    ap.add_argument("--per-type", action="store_true")
    ap.add_argument("--apply", nargs="*", default=[])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = load(a.decisions)
    if a.per_type:
        by = collections.defaultdict(list)
        for r in dev:
            by[r["type"]].append(r)
        temps = {t: fit(rs) for t, rs in by.items()}
        T_of = lambda r: temps.get(r["type"], 1.0)  # noqa: E731
    else:
        temps = {"all": fit(dev)}
        T_of = lambda r: temps["all"]  # noqa: E731
    report = {"fit_on": a.decisions, "temperatures": temps,
              "dev": {"T=1": stats(dev, lambda r: 1.0), "fitted": stats(dev, T_of)}, "applied": {}}
    for path in a.apply:
        rows = load(path)
        report["applied"][path] = {"T=1": stats(rows, lambda r: 1.0), "fitted": stats(rows, T_of)}
    print(json.dumps(report, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
