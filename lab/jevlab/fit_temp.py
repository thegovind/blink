"""Fit one positive temperature on dev_run records (probabilities at T=1): p_T ∝ p^(1/T); minimise mean NLL.
  python -m jevlab.fit_temp DEV_RUN.json [--srcs prefix1,prefix2]
"""
import json
import math
import sys


def nll(recs, T):
    tot = 0.0
    for r in recs:
        ps = r["p"]
        z = sum(max(v, 1e-12) ** (1 / T) for v in ps.values())
        tot += -math.log(max(ps[r["gold"]], 1e-12) ** (1 / T) / z)
    return tot / len(recs)


def ece(recs, T, bins=10):
    b = [[0, 0.0, 0] for _ in range(bins)]
    for r in recs:
        ps = r["p"]
        z = sum(max(v, 1e-12) ** (1 / T) for v in ps.values())
        q = {k: max(v, 1e-12) ** (1 / T) / z for k, v in ps.items()}
        pred = max(q, key=q.get)
        c = q[pred]
        x = b[min(int(c * bins), bins - 1)]
        x[0] += 1; x[1] += c; x[2] += pred == r["gold"]
    n = sum(x[0] for x in b)
    return sum(x[0] / n * abs(x[2] / x[0] - x[1] / x[0]) for x in b if x[0])


recs = json.load(open(sys.argv[1]))["records"]
if "--srcs" in sys.argv:
    pre = tuple(sys.argv[sys.argv.index("--srcs") + 1].split(","))
    recs = [r for r in recs if str(r.get("src", "")).startswith(pre)]
grid = [0.5 + 0.02 * i for i in range(126)]
best = min(grid, key=lambda T: nll(recs, T))
print(json.dumps({"n": len(recs), "T": round(best, 3), "nll_T1": round(nll(recs, 1.0), 4), "nll_T": round(nll(recs, best), 4),
                  "ece_T1": round(ece(recs, 1.0), 4), "ece_T": round(ece(recs, best), 4)}))
