"""Build space/results.json from measured files only.

  python lab/build_results.py [--ours4 runs/full-<4b>-scores.json] [--replay space/replay.json]

Sources:
  Decision Index leaderboard snapshot  runs/leaderboard-di-index-20260922.json (multimodalart/jev-decision-index data/index.json)
  our full-suite kit scores            runs/full-*-scores.json (kit `score` output)
  JevBench official values             benchmarkheaven.com/jev-models (Jev 1.13.0, JevK5 v0.2.0), transcribed below
  our JevBench public-item estimate    jevlab.jb_run summaries (soup-t3-t4s300-t4f.json, ref-jevk5-semif.json)
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AREAS = [
    ("knowledge", "Knowledge", "#5976bd"),
    ("language", "Language", "#407e9e"),
    ("retrieval", "Retrieval", "#3e8e5e"),
    ("tools", "Tools", "#b77a38"),
    ("arts", "Arts", "#b6698a"),
]
MIN_COMPLETED = 131980  # the leaderboard's own scoreable count; lower means partial coverage

# benchmarkheaven.com/jev-models, fetched 2026-09-24 (official sealed runs)
JB_OFFICIAL = {
    "jev-1.13.0": {"name": "Jev 1.13.0", "kind": "reference", "score": 63.3,
                   "axes": {"intelligence": 53.1, "calibration": 76.3, "speed": 83.3, "cost": 52.0},
                   "tiers": {"easy": 1.00, "standard": 0.99, "judge": 0.95, "hard": 0.74, "sealed": 0.37}},
    "jevk5-0.2.0": {"name": "JevK5 v0.2.0", "kind": "open", "score": 62.0,
                    "axes": {"intelligence": 48.9, "calibration": 74.5, "speed": 91.1, "cost": 59.5},
                    "tiers": {"easy": 1.00, "standard": 0.96, "judge": 0.95, "hard": 0.70, "sealed": 0.33}},
}


def harmonic(axes):
    v = list(axes.values())
    return len(v) / sum(1 / x for x in v)


def kit_system(path, sid, name, params, served):
    s = json.loads(Path(path).read_text())
    areas = {a["id"]: round(100 * a["raw"], 1) for a in s["areas"]}
    sc = s["scores"]
    return {"id": sid, "name": name, "kind": "ours", "params": params, "served_params": served,
            "index": sc["balanced_raw"], "skill": sc["balanced_skill"], "breadth": sc["breadth_skill"],
            "areas": areas, "completed": s["completed"], "latency_ms": s.get("latency_ms"),
            "source": "local full-suite run, kit scorer"}


def _lat(x):
    """Numbers only (median / p95 / mean ms); the board's free-text notes stay on the board."""
    if not isinstance(x, dict):
        return None
    return {k: x[k] for k in ("median", "p95", "mean") if isinstance(x.get(k), (int, float))} or None


def lb_areas(cats, res):
    out = {}
    for c in cats:
        vals = [(res.get(str(b)) or {}).get("score") for b in c["panel"]]
        if any(v is None for v in vals):
            return None
        out[c["id"]] = round(100 * sum(vals) / len(vals), 1)
    return out


def fmt_params(n):
    if n is None:
        return "closed"
    b = n / 1e9
    return f"{b:.0f}B" if b >= 9.5 else f"{b:.1f}B" if b >= 1 else f"{n / 1e6:.0f}M"


SIZE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?[BbMm](?:-A\d+(?:\.\d+)?B)?)(?![a-z])")


def nominal(base_model, served):
    """Nominal size from the base model's name (Qwen3.8-27B -> 27B), else the served count."""
    m = SIZE.findall(str(base_model or "").split("/")[-1].replace("_", "-"))
    return m[-1].upper() if m else fmt_params(served)


def frontier(points):
    """Ids nothing else beats: no other point with fewer-or-equal params and a higher score."""
    pts = sorted(points, key=lambda p: (p["params_b"], -p["score"]))
    best, out = -1.0, []
    for p in pts:
        if p["score"] > best:
            out.append(p["id"])
            best = p["score"]
    return out



HELDOUT_TASKS = [("massive_intent", "Intent routing"), ("enron_spam", "Spam"), ("toxic_chat", "Toxicity"),
                 ("ledgar_clause", "Contract clause"), ("climate_fever", "Claim vs evidence"),
                 ("svamp_numeric", "Word problems"), ("yelp_review_score", "Review stars"), ("ecthr_a_long", "Long case files")]


def heldout_section():
    """The frozen held-out pack (heldout/): equal-task mean accuracy with its bootstrap CI, per-task accuracy and ECE."""
    manifest = json.loads((ROOT / "heldout/MANIFEST.json").read_text())
    systems = []
    for sid in ("blink-27b", "blink-mimo-9b", "blink-4b"):
        sc = json.loads((ROOT / "heldout/outputs" / sid / "score.json").read_text())
        assert set(sc["tasks"]) == {t for t, _ in HELDOUT_TASKS}, sid
        assert all(v["missing_outputs"] == 0 and v["items_scored"] == 250 for v in sc["tasks"].values()), sid
        m = sc["equal_task_mean"]
        systems.append({"id": sid, "name": sid, "accuracy": round(m["accuracy"]["point"], 4),
                        "ci95": [round(x, 4) for x in m["accuracy"]["ci95"]], "ece": round(m["ece"]["point"], 4),
                        "tasks": {t: {"accuracy": round(v["accuracy"], 4), "ece": round(v["ece"], 4)} for t, v in sc["tasks"].items()}})
    return {"label": "Held-out tasks",
            "caption": "Eight public tasks, 250 items each, never used for training or for choosing a model.",
            "frozen_utc": manifest["freeze_time_utc"], "items_per_task": manifest["items_per_task"],
            "tasks": [{"id": t, "label": lab} for t, lab in HELDOUT_TASKS], "systems": systems}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lb", default=str(ROOT / "runs/leaderboard-di-index-20260922.json"))
    ap.add_argument("--ours27", default=str(ROOT / "runs/full-t4-27b-scores.json"))
    ap.add_argument("--ours4", default=str(ROOT / "runs/full-soup3-4b-scores.json"))
    ap.add_argument("--ours-mimo", default=str(ROOT / "runs/full-mimo-9b-scores.json"))
    ap.add_argument("--jb-ours", default=str(ROOT / "runs/jb-soup3-serial.json"))
    ap.add_argument("--jb-jevk5", default=str(ROOT / "runs/jb-ref-jevk5-semif.json"))
    ap.add_argument("--replay", default=str(ROOT / "space/replay.json"))
    ap.add_argument("--lb02", default=str(ROOT / "runs/leaderboard-di-index-v02-20260924.json"))
    ap.add_argument("--out", default=str(ROOT / "space/results.json"))
    a = ap.parse_args()

    lb = json.loads(Path(a.lb).read_text())
    cats = [c for c in lb["categories"] if c.get("panel")]
    # served parameters = what the released checkpoints hold (text model only; no vision or MTP tensors):
    # model.safetensors.index.json total_parameters of thegovind/blink-27b and thegovind/blink-4b
    systems = [kit_system(a.ours27, "blink-27b", "blink-27b", "27B", 26895998464)]
    if a.ours4:
        systems.append(kit_system(a.ours4, "blink-4b", "blink-4b", "4B", 4205751296))
    ours_mimo = a.ours_mimo if a.ours_mimo and Path(a.ours_mimo).exists() else None
    if ours_mimo:  # served text model: 8,953,803,264 parameters (the vision tower is not used for text decisions)
        systems.append(kit_system(ours_mimo, "blink-mimo-9b", "blink-mimo-9b", "9B", 8953803264))
    j = lb["jev"]
    systems.append({"id": "jev-1.13.0", "name": "Jev 1.13.0", "kind": "reference", "params": "closed",
                    "served_params": None, "index": j["scores"]["balanced_raw"],
                    "skill": j["scores"]["balanced_skill"], "breadth": j["scores"]["breadth_skill"],
                    "areas": lb_areas(cats, j["results"]), "latency_ms": _lat(j.get("latency")),
                    "source": "leaderboard"})
    for m in lb["models"]:
        if (m.get("completed") or 0) < MIN_COMPLETED:
            continue
        meta = m.get("meta") or {}
        systems.append({"id": m["engine"], "name": m["name"], "kind": "open",
                        "params": nominal(meta.get("base_model"), meta.get("served_params")),
                        "served_params": meta.get("served_params"), "base_model": meta.get("base_model"),
                        "index": m["scores"]["balanced_raw"], "skill": m["scores"]["balanced_skill"],
                        "breadth": m["scores"]["breadth_skill"], "areas": lb_areas(cats, m["results"]),
                        "completed": m.get("completed"), "latency_ms": _lat(m.get("latency")),
                        "source": "leaderboard"})
    systems.sort(key=lambda s: -s["index"])

    jev_raw = {c["id"]: 100 * sum(j["results"][str(b)]["score"] for b in c["panel"]) / len(c["panel"]) for c in cats}
    ours_files = {"blink-27b": a.ours27, "blink-4b": a.ours4, **({"blink-mimo-9b": ours_mimo} if ours_mimo else {})}
    for x in systems:
        if x["kind"] == "ours":
            raw = {ar["id"]: 100 * ar["raw"] for ar in json.loads(Path(ours_files[x["id"]]).read_text())["areas"]}
            raw["language"] = jev_raw["language"]
            x["index_language_equalized"] = round(sum(raw.values()) / len(raw), 2)

    v02 = None
    if Path(a.lb02).exists():
        lb2 = json.loads(Path(a.lb02).read_text())
        cats2 = [c for c in lb2["categories"] if c.get("panel")]
        panel2 = [b for c in cats2 for b in c["panel"]]
        ours_files = {"blink-27b": a.ours27, **({"blink-4b": a.ours4} if a.ours4 else {}),
                      **({"blink-mimo-9b": ours_mimo} if ours_mimo else {})}
        ours_b = {k: json.loads(Path(v).read_text())["benchmarks"] for k, v in ours_files.items()}
        keep = [b for b in panel2 if all((o.get(str(b)) or {}).get("score") is not None for o in ours_b.values())]

        def part(get):
            am = {}
            for c in cats2:
                vals = [get(b) for b in c["panel"] if b in keep]
                am[c["id"]] = 100 * sum(vals) / len(vals)
            return round(sum(am.values()) / len(am), 2), {k: round(v, 1) for k, v in am.items()}

        rows = []
        for k, o in ours_b.items():
            idx, am = part(lambda b: o[str(b)]["score"])
            rows.append({"id": k, "name": k, "kind": "ours", "partial": idx, "areas": am})
        for name, sid, kind, res, full in [("Jev 1.13.0", "jev-1.13.0", "reference", lb2["jev"]["results"], lb2["jev"]["scores"]["balanced_raw"])] + \
                [(m["name"], m["engine"], "open", m["results"], m["scores"]["balanced_raw"]) for m in lb2["models"]]:
            if all((res.get(str(b)) or {}).get("score") is not None for b in keep):
                idx, am = part(lambda b: res[str(b)]["score"])
                rows.append({"id": sid, "name": name, "kind": kind, "partial": idx, "areas": am, "official_02": full})
        rows.sort(key=lambda r: -r["partial"])
        v02 = {
            "label": "Decision Index 0.2 · shared benchmarks",
            "caption": f"Area means over the {len(keep)} of {len(panel2)} panel benchmarks that 0.2 keeps from 0.1, for every system alike. Not a 0.2 score: seven new benchmarks are missing, and ToolRet/BRIGHT are scored on the full 0.1 sets.",
            "snapshot": lb2["generated_utc"], "shared": len(keep), "panel": len(panel2),
            "missing": [lb2["benchmarks"][str(b)]["dataset"] for b in panel2 if b not in keep],
            "jev_official_02": lb2["jev"]["scores"]["balanced_raw"],
            "systems": rows,
        }

    # speed vs quality: for ours, the Decision Index kit's own per-request timer (http engine against release/serve.py,
    # one request at a time; lab/kit_latency.sh) over 1,000 uniform suite requests, else the in-process serial sample
    # (lab/jevlab/latency_sample.py, same requests); for everyone else, the latency their leaderboard run reported
    lat_files = {m: [ROOT / f"runs/kit-latency-{m}.json", ROOT / f"runs/latency-{m}.json"]
                 for m in ("blink-4b", "blink-27b", "blink-mimo-9b")}
    speed_pts = []
    for x in systems:
        if x["kind"] == "ours":
            f = next((f for f in lat_files.get(x["id"], []) if f.exists()), None)
            if f is None:
                continue
            L = json.loads(f.read_text())["summary"]
            med, p95 = L["median_ms"], L["p95_ms"]
        else:
            L = x.get("latency_ms") or {}
            med, p95 = L.get("median"), L.get("p95")
        if med:
            speed_pts.append({"id": x["id"], "name": x["name"], "kind": x["kind"], "latency_ms": round(float(med), 1),
                              "p95_ms": round(float(p95), 1) if p95 else None, "score": x["index"]})
    best, speed_front = -1.0, []
    for q in sorted(speed_pts, key=lambda q: (q["latency_ms"], -q["score"])):
        if q["score"] > best:
            speed_front.append(q["id"])
            best = q["score"]
    speed = {
        "label": "Speed against quality",
        "caption": "Decision Index 0.1 vs median time per request.",
        "x_key": "latency_ms", "x_label": "Median latency per request", "y_label": "Decision Index",
        "points": speed_pts, "frontier": speed_front,
        "note": "Each row uses that system's own kit-timer measurement, not one shared machine. Jev is a hosted API over the network; blink uses its own HTTP server, one request at a time over 1,000 random suite requests.",
    }

    # computer use, a transfer probe (cua/eval_cua.py): pick the element to act on next, 5 options, one pass
    cua = None
    cua_path = ROOT / "cua/results.json"
    if cua_path.exists():
        S = json.loads(cua_path.read_text())["summaries"]
        mimo_cua = ROOT / "cua/results-mimo.json"
        if mimo_cua.exists():
            S.update(json.loads(mimo_cua.read_text())["summaries"])
        rows_def = [("blink-4b-text::text", "blink-4b", "text", "ours"),
                    ("blink-4b-vision-graft::vision", "blink-4b + vision tower", "screenshot", "ours"),
                    ("blink-mimo-9b::text", "blink-mimo-9b", "text", "ours"),
                    ("blink-mimo-9b::vision", "blink-mimo-9b", "screenshot", "ours"),
                    ("qwen3.5-4b-base::text", "Qwen3.5-4B", "text", "base"),
                    ("qwen3.5-4b-base::vision", "Qwen3.5-4B", "screenshot", "base"),
                    ("qwen3.5-9b-base::text", "Qwen3.5-9B", "text", "base"),
                    ("qwen3.5-9b-base::vision", "Qwen3.5-9B", "screenshot", "base"),
                    ("mimo-v2.6-distill-qwen-9b::text", "MiMo-V2.6-Distill-Qwen-9B", "text", "base"),
                    ("mimo-v2.6-distill-qwen-9b::vision", "MiMo-V2.6-Distill-Qwen-9B", "screenshot", "base")]
        cua_rows = []
        for key, name, inp, kind in rows_def:
            r = S.get(key)
            if r and r.get("status") == "complete":
                cua_rows.append({"name": name, "input": inp, "kind": kind, "n": r["n"],
                                 "accuracy": round(100 * r["accuracy"], 1),
                                 "ci95": [round(100 * v, 1) for v in r["accuracy_ci95"]],
                                 "ece": round(r["top_label_ece_10bin"], 3)})
        cua = {
            "label": "Picking the next click",
            "caption": "500 Multimodal-Mind2Web test steps, five page elements per step. Text sees element attributes; screenshot sees a crop with five boxed elements.",
            "chance": 20.0, "rows": cua_rows,
            "note": "A custom harness and sampling, not the official Mind2Web evaluation. Five choices are easier than ranking a whole page; blink never trained on web actions. With page text, blink-4b beats its base by about 10 points; a screenshot adds nothing after training.",
        }

    pts = [{"id": s["id"], "name": s["name"], "kind": s["kind"], "params_b": round(s["served_params"] / 1e9, 3),
            "score": s["index"]} for s in systems if s["served_params"]]

    jb_ours = json.loads(Path(a.jb_ours).read_text())["summary"]
    jb_k5 = json.loads(Path(a.jb_jevk5).read_text())["summary"]

    def tariff_cost(s):  # JevBench v1.4 cost axis: 100 - 30 log10(USD per 1,000 decisions / 0.001), 4B tariff $0.03/M input
        usd = s["input_tokens_per_decision"] * 1000 * 0.03 / 1e6  # USD per 1,000 decisions
        return max(0.0, min(100.0, 100 - 30 * math.log10(usd / 0.001))) if usd > 0 else 100.0

    def est(s):
        axes = {"intelligence": s["intelligence_public"], "calibration": s["calibration"], "speed": s["speed"],
                "cost": tariff_cost(s)}
        return {k: round(v, 1) for k, v in axes.items()}, round(harmonic(axes), 1)

    ours_axes, ours_est = est(jb_ours)
    k5_axes, k5_est = est(jb_k5)
    jb_systems = [
        {"id": "blink-4b", "name": "blink-4b", "kind": "ours", "official": None,
         "public_estimate": ours_est, "public_axes": ours_axes,
         "public": {"standard": jb_ours["tiers"]["standard"], "hard": jb_ours["tiers"]["hard"],
                    "hard_ece": round(jb_ours["hard_ece"], 3), "tvd": round(jb_ours["prob_tvd"], 3)}},
        {"id": "jevk5-0.2.0", "name": "JevK5 v0.2.0", "kind": "open", "official": JB_OFFICIAL["jevk5-0.2.0"],
         "public_estimate": k5_est, "public_axes": k5_axes,
         "public": {"standard": jb_k5["tiers"]["standard"], "hard": jb_k5["tiers"]["hard"],
                    "hard_ece": round(jb_k5["hard_ece"], 3), "tvd": round(jb_k5["prob_tvd"], 3)}},
        {"id": "jev-1.13.0", "name": "Jev 1.13.0", "kind": "reference", "official": JB_OFFICIAL["jev-1.13.0"],
         "public_estimate": None, "public_axes": None, "public": None},
    ]

    lat = None
    rp = Path(a.replay)
    if rp.exists():
        rep = json.loads(rp.read_text())
        xs = sorted(v["latency_ms"] for v in rep["requests"].values())
        lat = {"requests": len(xs), "p50_ms": xs[len(xs) // 2], "max_ms": xs[-1], "generated_tokens": 0,
               "caption": "End to end on bundled examples."}

    out = {
        "placeholder": False,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": {"small": "blink-4b", "large": "blink-27b", "base_small": "Qwen/Qwen3.5-4B",
                  "base_large": "Qwen/Qwen3.8-27B", "adapter": "LoRA, merged",
                  "bases": {"thegovind/blink-4b": "Qwen/Qwen3.5-4B", "thegovind/blink-27b": "Qwen/Qwen3.8-27B",
                            "thegovind/blink-mimo-9b": "XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B"}},
        "decision_index": {
            "label": "Decision Index 0.1",
            "edition": "0.1 (archived 19-benchmark edition; leaderboard snapshot 2026-09-22)",
            "caption": "Complete archived 0.1 suite: 132,422 requests across 37 benchmarks. The headline index averages 19 panel benchmarks. Local official-kit scorer, not leaderboard submissions; unanswered requests count as wrong.",
            "suite": "132,422 requests · 37 benchmarks",
            "leaderboard_snapshot": lb["generated_utc"],
            "areas": [{"id": i, "label": l, "color": c} for i, l, c in AREAS],
            "reference": "jev-1.13.0",
            "systems": systems,
        },
        "jevbench": {
            "label": "JevBench",
            "caption": "Public items only. Development proxies, not official scores or predictions.",
            "axes": [{"id": "intelligence", "label": "Intelligence"}, {"id": "calibration", "label": "Calibration"},
                     {"id": "speed", "label": "Speed"}, {"id": "cost", "label": "Cost"}],
            "reference": "jev-1.13.0",
            "systems": jb_systems,
            "notes": {
                "hard_correct": {"blink-4b": round(jb_ours["tiers"]["hard"] * 111), "jevk5-0.2.0": round(jb_k5["tiers"]["hard"] * 111),
                                 "jevk5_native_published": 82, "n": 111},
                "speed_measured": {"blink-4b": round(jb_ours["speed"], 1), "jevk5-0.2.0": round(jb_k5["speed"], 1)},
                "speed_note": (f"Speed comes from each row's own serial run over the public items (blink-4b raw p50 {1000 * jb_ours['p50_s']:.0f} ms, p95 {1000 * jb_ours['p95_s']:.0f} ms). "
                               f"The proxy Speed axes, {jb_ours['speed']:.1f} for blink-4b and {jb_k5['speed']:.1f} for JevK5, already "
                               "apply JevBench's self-hosted adjustment: twice raw latency plus 0.15 seconds. Official Speed would depend "
                               "on the evaluator's own measurements."),
                "cost_note": "Cost uses JevBench's 4B tariff ($0.03 per million input tokens) times measured tokens per decision; not a production bill.",
                "selection_note": "The 231 public items and JevK5's 65 hand-written hard items were reused for candidate selection, though never for training. These are development-set results.",
            },
        },
        "speed_pareto": speed,
        "cua_probe": cua,
        "decision_index_v02_partial": None,  # withheld: needs matched request subsets and 0.2 metric transformations
        "decision_index_v02_note": "No Decision Index 0.2 result is reported for these models. Comparable shared-benchmark results require matched request subsets and the 0.2 metric transformations.",
        "pareto": {
            "label": "Size against quality",
            "caption": "Decision Index 0.1 vs served parameters, not a cost or latency result. The line connects open models that score higher than every smaller model.",
            "x_label": "Served parameters",
            "x_key": "params_b",
            "y_label": "Decision Index",
            "points": pts,
            "frontier": frontier(pts),
            "reference": {"name": "Jev 1.13.0 (closed)", "score": j["scores"]["balanced_raw"]},
        },
        "latency": lat,
        "heldout": heldout_section(),
    }
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"wrote {a.out}: {len(systems)} DI systems, frontier {out['pareto']['frontier']}")


if __name__ == "__main__":
    main()
