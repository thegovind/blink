"""Paired, stratified case-group bootstrap of Decision Index differences (kit metrics, area weights).

  python -m jevlab.di_boot --suite-dir $BLINK_WORKDIR/kit/suite-s3000 --a RUN_A/results.jsonl --b RUN_B/results.jsonl [--reps 300]
Within each panel benchmark, case groups are resampled with replacement (the same draw for A and B);
every metric is recomputed by the kit's own static_score/index code on the resampled rows.
"""
import os
import argparse
import collections
import copy
import json
import random
import sys

sys.path.insert(0, os.environ.get("DECISION_INDEX_KIT", "decision-index"))
from decision_index import constants as C  # noqa: E402
from decision_index.scoring import index as I  # noqa: E402
from decision_index.scoring.report import load_results  # noqa: E402
from decision_index.suite.io import Suite  # noqa: E402


def res_map(results, ids):
    return {rid: {"status": r["status"], "answers": (r.get("response") or {}).get("answers", {})}
            for rid, r in results.items() if r.get("catalog_id") in ids}


def di_of(rows_by_n, res, baselines):
    scored = {n: I.static_score(n, rs, res, baselines) for n, rs in rows_by_n.items()}
    for n in C.INTERACTIVE:
        scored[n] = I.interactive_placeholder(n)
    e = I.index_entry(scored)
    return e["index"], {a["id"]: a["raw"] for a in e["areas"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite-dir", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    ids = I.panel_ids()
    rows_by_n = collections.defaultdict(list)
    for r in Suite(a.suite_dir).rows(apply_exclusions=True):
        n = r["_evaluation"]["catalog_id"]
        if n in ids:
            rows_by_n[n].append(r)
    baselines = I.chance_baselines()
    ra, rb = res_map(load_results(a.a), ids), res_map(load_results(a.b), ids)
    base_a, areas_a = di_of(rows_by_n, ra, baselines)
    base_b, areas_b = di_of(rows_by_n, rb, baselines)
    groups = {n: collections.defaultdict(list) for n in rows_by_n}
    for n, rs in rows_by_n.items():
        for r in rs:
            groups[n][r["_evaluation"]["group_id"]].append(r)
    rng = random.Random(a.seed)
    diffs, das, dbs = [], [], []
    for rep in range(a.reps):
        rows_rep = {}
        ra2, rb2 = {}, {}
        for n, g in groups.items():
            keys = list(g)
            draw = [rng.choice(keys) for _ in keys]
            out = []
            for k, gid in enumerate(draw):
                for r in g[gid]:
                    rr = copy.copy(r)
                    ev = dict(r["_evaluation"])
                    rid = ev["run_id"]
                    ev["run_id"] = f"{rid}#{k}"
                    ev["group_id"] = f"{ev['group_id']}#{k}"
                    rr["_evaluation"] = ev
                    out.append(rr)
                    if rid in ra:
                        ra2[ev["run_id"]] = ra[rid]
                    if rid in rb:
                        rb2[ev["run_id"]] = rb[rid]
            rows_rep[n] = out
        da, _ = di_of(rows_rep, ra2, baselines)
        db, _ = di_of(rows_rep, rb2, baselines)
        diffs.append(db - da)
        das.append(da)
        dbs.append(db)
    diffs.sort()
    lo, hi = diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs)) - 1]
    print(json.dumps({"A": round(base_a, 2), "B": round(base_b, 2), "B_minus_A": round(base_b - base_a, 2),
                      "ci95": [round(lo, 2), round(hi, 2)], "p_B_gt_A": round(sum(d > 0 for d in diffs) / len(diffs), 3),
                      "areas_A": {k: round(v * 100, 1) for k, v in areas_a.items()},
                      "areas_B": {k: round(v * 100, 1) for k, v in areas_b.items()}, "reps": a.reps}, indent=1))


if __name__ == "__main__":
    main()
