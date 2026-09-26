"""Card data for the local Decision Index 0.2 run (CPU; kit aggregate reports only, no per-request records).

  PYTHONPATH=<decision-index checkout> python lab/di02_card_data.py --results <dir> --out runs/di02-local.json

<dir> holds the kit's reports di02-<model>.scores.json and the training-row screening audit audit/added-<tag>.json.
Every number is recomputed with the kit's own 0.2 transformations (decision_index.scoring.index02): each benchmark's
index value from its native metric (the kit's rounded values for track-scored benchmarks), then the five-area
aggregate. The recomputed headline, raw and breadth must match the kit's reported values. Also written:
  - the MMLU-Pro-excluded sensitivity: MMLU-Pro dropped, the other nine Knowledge benchmarks averaged, five equal
    areas (a descriptive statistic, not a contamination-free score);
  - unrounded area values (rounded only for display);
  - the seven benchmarks 0.2 added, and the screening counts: training rows with an exact normalised match of
    30+ characters anywhere in the added requests, by training stage and source (not unique exposed questions).
No leaderboard-style exposure penalty is applied: the kit's scorer doesn't read training-overlap audits.
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
from pathlib import Path

from decision_index.scoring import index02 as I

MODELS = {  # card name: (audit tag, {stage label: audit mixture key})
    "blink-27b": ("27b", {"T2": "t2_27b_train", "T4": "t4_27b_train"}),
    "blink-mimo-9b": ("mimo", {"MiMo": "mimo_train"}),
    "blink-4b": ("4b", {"T3": "t3_train", "T4": "t4_4b_train"}),
}
MMLU_PRO = 57
SOURCES = {"mmlu_pro": "MMLU-Pro", "supergpqa": "SuperGPQA", "boolq": "BoolQ", "medmcqa": "MedMCQA"}


def values_for(report: dict, spec: dict) -> dict:
    out = {}
    for area in spec["areas"]:
        for n in area["benchmarks"]:
            key, native = str(n), report["benchmarks"][str(n)]
            if n in spec["track_scored"]:
                v = report["index_benchmarks"][key]  # the kit rounds track values itself
                out[n] = {"raw": v["raw"], "skill": v["skill"], "coverage": v["coverage"]}
            else:
                out[n] = I.benchmark_value(n, spec, native=native)
    return out


def without(spec: dict, n: int) -> dict:
    s = copy.deepcopy(spec)
    for area in s["areas"]:
        area["benchmarks"] = [b for b in area["benchmarks"] if b != n]
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--evaluated", default="2026-09-25")
    a = ap.parse_args()
    res = Path(a.results)
    spec = I.spec()
    kit_dir = Path(I.__file__).resolve().parents[2]
    kit = subprocess.run(["git", "-C", str(kit_dir), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    models, suite = {}, None
    for name, (tag, stages) in MODELS.items():
        report = json.loads((res / f"di02-{name}.scores.json").read_text())
        assert report["engine"] == name and report["panel_id"] == spec["panel_id"], name
        assert report["complete"] and report["counts"]["ok"] == report["completed"], name
        suite = suite or report["suite"]
        assert report["suite"] == suite, "the three reports must share one suite"
        vals = values_for(report, spec)
        full, areas = I.aggregate(vals, spec)
        for k in ("balanced_skill", "balanced_raw", "breadth_skill"):
            assert f"{full[k]:.2f}" == f"{report['scores'][k]:.2f}", (name, k, full[k], report["scores"][k])
        no_mmlu, _ = I.aggregate(vals, without(spec, MMLU_PRO))
        audit = json.loads((res / "audit" / f"added-{tag}.json").read_text())
        assert audit["suite_rows"] == suite["added_requests"]
        screening = []
        for label, key in stages.items():
            mix = audit["mixtures"][key]
            hits = mix.get("sources_with_hits", {})
            assert set(hits) <= set(SOURCES), (name, set(hits) - set(SOURCES))
            screening.append({"stage": label, "rows": mix["totals"]["rows"], "matched": mix["totals"]["suite_exact"],
                              "by_source": {SOURCES[s]: hits.get(s, {}).get("suite_exact", 0) for s in SOURCES}})
        models[name] = {
            "balanced_skill": round(full["balanced_skill"], 2),
            "balanced_raw": round(full["balanced_raw"], 2),
            "breadth_skill": round(full["breadth_skill"], 2),
            "without_mmlu_pro": round(no_mmlu["balanced_skill"], 2),
            "completed": report["completed"],
            "areas": [{"id": x["id"], "label": x["label"], "benchmarks": x["n"],
                       "skill": round(100 * x["skill"], 1), "raw": round(100 * x["raw"], 1)} for x in areas],
            "added": [{"id": int(k), "dataset": report["benchmarks"][k]["dataset"],
                       "metric": report["benchmarks"][k]["metric"], "requests": report["benchmarks"][k]["requests"],
                       "answered": report["benchmarks"][k]["answered"],
                       "raw": round(100 * vals[int(k)]["raw"], 1), "skill": round(100 * vals[int(k)]["skill"], 1)}
                      for k in sorted(spec["added"], key=int)],
            "screening": screening,
        }
    scored = {len(json.loads((res / f"di02-{n}.scores.json").read_text())["benchmarks"]) for n in MODELS}
    assert len(scored) == 1
    out = {
        "kit_commit": kit, "evaluated": a.evaluated, "panel_id": spec["panel_id"],
        "requests": suite["scoreable"] + suite["added_requests"], "added_requests": suite["added_requests"],
        "benchmarks_scored": scored.pop(), "benchmarks_in_index": sum(len(x["benchmarks"]) for x in spec["areas"]),
        "rows_sha256": suite["rows_sha256"], "added_sha256": suite["added_sha256"],
        "exposure_penalty_applied": False, "models": models,
    }
    for m in models.values():
        assert m["completed"] == out["requests"]
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    for name, m in models.items():
        print(f"{name}: skill {m['balanced_skill']} raw {m['balanced_raw']} breadth {m['breadth_skill']} "
              f"without MMLU-Pro {m['without_mmlu_pro']} | areas "
              + " / ".join(f"{x['skill']}" for x in m["areas"]))


if __name__ == "__main__":
    main()
