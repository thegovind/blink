#!/usr/bin/env python3
"""Why the reward matters for a decision model: a small simulation in pure Python (under ten seconds).

  python examples/rl/reward_demo.py

A policy picks one of K options in each of N situations: a softmax over one score per option, the same shape
as blink's answer over the offered letters. Each situation has a true answer distribution q, known here and
unknown in real life. Every visit draws one realised outcome y from q, the way real feedback arrives. The same
policy is trained on the same kind of outcomes with three rewards:

  accuracy (GRPO)   sample a group of answers, reward 1 when an answer equals y, normalise within the group
  log score         reward log p(y), the probability the model gave to what happened
  Brier score       reward -1/2 * sum_k (p_k - [k = y])^2

It then reports, against the true q: how often the top option is the most likely outcome (accuracy), the mean
top-option probability (confidence), how often the top option comes true (hit rate), the binned gap between
confidence and hit rate (calibration error), the entropy, and the expected log loss.
"""
from __future__ import annotations

import argparse
import math
import random


def softmax(z: list[float]) -> list[float]:
    top = max(z)
    e = [math.exp(x - top) for x in z]
    s = sum(e)
    return [x / s for x in e]


def draw(p: list[float], rng: random.Random) -> int:
    u, acc = rng.random(), 0.0
    for i, x in enumerate(p):
        acc += x
        if u < acc:
            return i
    return len(p) - 1


def true_distributions(n: int, k: int, rng: random.Random) -> list[list[float]]:
    """Some situations are clear, some are close calls, as in real decisions."""
    out = []
    for _ in range(n):
        sharp = rng.choice([0.3, 1.0, 3.0])
        g = [rng.gammavariate(sharp, 1.0) for _ in range(k)]
        s = sum(g)
        out.append([x / s for x in g])
    return out


def grad_accuracy_grpo(p, y, rng, group=8, eps=0.1):
    """GRPO on one decision: sampled answers, reward 1 if right, advantage (r - mean) / (std + eps)."""
    acts = [draw(p, rng) for _ in range(group)]
    r = [1.0 if a == y else 0.0 for a in acts]
    mean = sum(r) / group
    std = math.sqrt(sum((x - mean) ** 2 for x in r) / group)
    g = [0.0] * len(p)
    for a, ri in zip(acts, r):
        adv = (ri - mean) / (std + eps)
        for j in range(len(p)):
            g[j] += adv * ((1.0 if j == a else 0.0) - p[j]) / group
    return g


def grad_log_score(p, y, rng):
    """d/dz log p_y = onehot(y) - p: exact, no sampling. The same as training on the outcome as a label."""
    return [(1.0 if j == y else 0.0) - pj for j, pj in enumerate(p)]


def grad_brier(p, y, rng):
    """d/dz of -1/2 * sum_k (p_k - [k = y])^2 through the softmax."""
    d = [pk - (1.0 if k == y else 0.0) for k, pk in enumerate(p)]
    dot = sum(dk * pk for dk, pk in zip(d, p))
    return [-pj * (dj - dot) for pj, dj in zip(p, d)]


def train(rule, qs, steps, lr, rng, power=0.7):
    """One score per option per situation; each situation's step size decays as lr / visits**power."""
    z = [[0.0] * len(q) for q in qs]
    visits = [0] * len(qs)
    for _ in range(steps):
        n = rng.randrange(len(qs))
        visits[n] += 1
        p = softmax(z[n])
        y = draw(qs[n], rng)
        step = lr / visits[n] ** power
        for j, gj in enumerate(rule(p, y, rng)):
            z[n][j] += step * gj
    return [softmax(row) for row in z]


def report(ps, qs, bins=10) -> dict:
    tops = [max(range(len(p)), key=p.__getitem__) for p in ps]
    conf = [p[t] for p, t in zip(ps, tops)]
    hit = [q[t] for q, t in zip(qs, tops)]
    cells = [[0, 0.0, 0.0] for _ in range(bins)]
    for c, h in zip(conf, hit):
        cell = cells[min(int(c * bins), bins - 1)]
        cell[0] += 1
        cell[1] += c
        cell[2] += h
    n = len(ps)
    return {
        "accuracy": sum(t == max(range(len(q)), key=q.__getitem__) for t, q in zip(tops, qs)) / n,
        "confidence": sum(conf) / n,
        "hit_rate": sum(hit) / n,
        "calibration_error": sum(abs(c[1] - c[2]) for c in cells if c[0]) / n,
        "entropy": sum(-sum(x * math.log(x) for x in p if x > 0) for p in ps) / n,
        "expected_log_loss": sum(-sum(qk * math.log(max(pk, 1e-12)) for qk, pk in zip(q, p))
                                 for p, q in zip(ps, qs)) / n,
    }


RULES = (("accuracy reward (GRPO)", grad_accuracy_grpo), ("log score", grad_log_score), ("Brier score", grad_brier))


def run(situations=200, options=4, steps=40000, lr=2.0, seed=0) -> dict:
    rng = random.Random(seed)
    qs = true_distributions(situations, options, rng)
    results = {"start (uniform)": report([[1.0 / options] * options for _ in qs], qs)}
    for name, rule in RULES:
        results[name] = report(train(rule, qs, steps, lr, random.Random(seed + 1)), qs)
    results["true distribution"] = report(qs, qs)
    return results


def longer(situations=200, options=4, steps=40000, lr=2.0, seed=0) -> dict:
    """The first two rewards after a quarter, all and four times the outcomes."""
    qs = true_distributions(situations, options, random.Random(seed))
    return {(name, n): report(train(rule, qs, n, lr, random.Random(seed + 1)), qs)
            for name, rule in RULES[:2] for n in (steps // 4, steps, steps * 4)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--situations", type=int, default=200)
    ap.add_argument("--options", type=int, default=4)
    ap.add_argument("--steps", type=int, default=40000, help="outcomes seen during training")
    ap.add_argument("--lr", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    cols = ("accuracy", "confidence", "hit_rate", "calibration_error", "entropy", "expected_log_loss")
    print(f"Same outcomes, three rewards ({a.situations} situations, {a.options} options, {a.steps:,} outcomes):")
    print(f"{'':24}" + "".join(f"{c.replace('_', ' '):>19}" for c in cols))
    for name, r in run(a.situations, a.options, a.steps, a.lr, a.seed).items():
        print(f"{name:24}" + "".join(f"{r[c]:19.3f}" for c in cols))
    print("\nTrain longer:")
    print(f"{'':24}{'outcomes':>10}" + "".join(f"{c.replace('_', ' '):>19}" for c in cols[:4]))
    for (name, n), r in longer(a.situations, a.options, a.steps, a.lr, a.seed).items():
        print(f"{name:24}{n:>10,}" + "".join(f"{r[c]:19.3f}" for c in cols[:4]))


if __name__ == "__main__":
    main()
