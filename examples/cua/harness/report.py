"""Aggregate independently verified episodes and their visited-step decisions."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

_COUNTS = ("element_correct", "element_total", "done_tp", "done_fp", "done_fn",
           "done_tn", "risk_tp", "risk_fp", "risk_fn", "risk_tn", "gates", "oracle_issues")


def wilson(successes: int, total: int) -> tuple[float, float] | None:
    if not total:
        return None
    z = 1.959963984540054
    p = successes / total
    center = (p + z * z / (2 * total)) / (1 + z * z / total)
    radius = (z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
              / (1 + z * z / total))
    return max(0.0, center - radius), min(1.0, center + radius)


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _percentile(values: list[float], at: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * at
    low = int(position)
    return round(ordered[low] + (ordered[min(low + 1, len(ordered) - 1)]
                                  - ordered[low]) * (position - low), 2)


def aggregate(rows: list[dict]) -> dict:
    counts = {key: sum(int(row.get(key, 0)) for row in rows) for key in _COUNTS}
    success = sum(row["success"] for row in rows)
    latency = [ms for row in rows for ms in row.get("wall_ms", [])]
    modes = Counter(row["stop_reason"] for row in rows if not row["success"])
    details = Counter(str(row.get("detail")) for row in rows if not row["success"])
    waves = [row["waves_survived"] for row in rows if isinstance(row.get("waves_survived"), (int, float))]
    hits = [row["hits"] for row in rows if isinstance(row.get("hits"), (int, float))]
    return {
        "episodes": len(rows), "successes": success,
        "success_rate": _ratio(success, len(rows)),
        "success_wilson_95": wilson(success, len(rows)),
        "mean_steps": round(statistics.mean(row["steps"] for row in rows), 2) if rows else None,
        "mean_optimal_steps": round(statistics.mean(row["optimal_steps"] for row in rows), 2) if rows else None,
        "element_accuracy": _ratio(counts["element_correct"], counts["element_total"]),
        "done_precision": _ratio(counts["done_tp"], counts["done_tp"] + counts["done_fp"]),
        "done_recall": _ratio(counts["done_tp"], counts["done_tp"] + counts["done_fn"]),
        "risk_gate_recall": _ratio(counts["risk_tp"], counts["risk_tp"] + counts["risk_fn"]),
        "false_gate_rate": _ratio(counts["risk_fp"], counts["risk_fp"] + counts["risk_tn"]),
        "p50_decision_ms": _percentile(latency, 0.5),
        "p95_decision_ms": _percentile(latency, 0.95),
        "decision_count": len(latency),
        "errors": sum(bool(row.get("error")) for row in rows),
        "video_errors": sum(bool(row.get("video_error")) for row in rows),
        "failure_stop_reasons": dict(modes.most_common()),
        "failure_details": dict(details.most_common(5)),
        "mean_waves_survived": round(statistics.mean(waves), 2) if waves else None,
        "mean_hits": round(statistics.mean(hits), 2) if hits else None,
        **counts,
    }


def load_results(run_dir: Path) -> list[dict]:
    path = run_dir / "results.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    seen = set()
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            row = json.loads(line)
            key = (row["scenario"], row["task_index"], row["seed"], row["model"],
                   row["interrupts"], row["mode"])
            if key in seen:
                raise ValueError(f"Duplicate episode {key} in {path}:{line_no}")
            seen.add(key)
            rows.append(row)
    return rows


def _cell(value: float | None, *, percentage: bool = True) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1%}" if percentage else f"{value:.0f}"


def markdown(report: dict) -> str:
    lines = [f"# CUA run {report['run_id']}", "",
             f"{report['total']['episodes']} episodes; success is `blinkScenario.check().success` "
             "(including forbidden side effects). Confidence intervals are Wilson 95%; "
             "decision latency is client-observed loopback wall time, excluding the video display beat.", "",
             "| Model | Scenario | Interrupts | Mode | N | Success (95% CI) | "
             "Steps / optimal | Element | Done P / R | Risk recall / false gate | p50 / p95 ms | "
             "Game waves / hits |",
             "|---|---|---:|---|---:|---|---|---:|---|---|---:|---:|"]
    if len(report.get("sources", [])) > 1:
        origins = ", ".join(f"{s['run_id']} ({', '.join(s['harness_shas'])})" for s in report["sources"])
        lines[4:4] = [f"Source runs, with harness commits: {origins}. Distinct commits are "
                      "reported, not treated as a matched version.", ""]
    for entry in report["groups"]:
        a = entry["metrics"]
        interval = a["success_wilson_95"]
        ci = f"{_cell(a['success_rate'])} [{_cell(interval[0])}, {_cell(interval[1])}]" if interval else "n/a"
        latency = (f"{_cell(a['p50_decision_ms'], percentage=False)} / "
                   f"{_cell(a['p95_decision_ms'], percentage=False)}")
        game = (f"{a['mean_waves_survived']:.1f} / {a['mean_hits']:.1f}"
                if a["mean_waves_survived"] is not None and a["mean_hits"] is not None else "n/a")
        lines.append(
            f"| {entry['model']} | {entry['scenario']} | {entry['interrupts']} | {entry['mode']} | "
            f"{a['episodes']} | {ci} | {a['mean_steps']} / {a['mean_optimal_steps']} | "
            f"{_cell(a['element_accuracy'])} | "
            f"{_cell(a['done_precision'])} / {_cell(a['done_recall'])} | "
            f"{_cell(a['risk_gate_recall'])} / {_cell(a['false_gate_rate'])} | {latency} | {game} |"
        )
    lines.extend(["", "Element misses include any off-path step whose oracle target is not on-screen; "
                  "these are counted separately as `oracle_issues` in report.json. "
                  "Done and risk denominators are *visited* non-game steps; risk excludes "
                  "already-done steps. Steps / optimal uses the registry's `optimal_steps`; "
                  "tasks marked `optimal_steps_exact: false` use an upper bound, not a per-seed optimum. "
                  "An unavailable denominator is `n/a`, not zero. "
                  "Game rows report mean waves survived / hits; full counts are in report.json.", ""])
    return "\n".join(lines)


def _build(run_id: str, rows: list[dict], sources: list[dict]) -> dict:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (row["model"], row["interrupts"], row["mode"])
        grouped[(*key, "ALL")].append(row)
        grouped[(*key, row["scenario"])].append(row)
    groups = [{"model": model, "interrupts": interrupts, "mode": mode, "scenario": scenario,
               "metrics": aggregate(members)}
              for (model, interrupts, mode, scenario), members in sorted(grouped.items())]
    return {"run_id": run_id, "total": aggregate(rows), "groups": groups, "sources": sources}


def _write(report: dict, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output / "report.md").write_text(markdown(report), encoding="utf-8")
    return report


def generate(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    rows = load_results(run_dir)
    sources = [{"run_id": run_dir.name,
                "harness_shas": sorted({row["harness_sha"] for row in rows})}]
    return _write(_build(run_dir.name, rows, sources), run_dir)


def combine(run_dirs: list[Path], output: Path) -> dict:
    if not run_dirs:
        raise ValueError("At least one source run is required")
    rows = []
    sources = []
    seen = set()
    for folder in run_dirs:
        folder = Path(folder)
        current = load_results(folder)
        for row in current:
            key = (row["scenario"], row["task_index"], row["seed"], row["model"],
                   row["interrupts"], row["mode"])
            if key in seen:
                raise ValueError(f"Episode {key} occurs in multiple source runs")
            seen.add(key)
        rows.extend(current)
        sources.append({"run_id": folder.name,
                        "harness_shas": sorted({row["harness_sha"] for row in current})})
    return _write(_build(Path(output).name, rows, sources), Path(output))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, help="output folder, required when combining runs")
    args = parser.parse_args(argv)
    if len(args.run_dirs) > 1 and args.out is None:
        parser.error("--out is required when combining multiple runs")
    result = (combine(args.run_dirs, args.out) if len(args.run_dirs) > 1 else
              generate(args.run_dirs[0]) if args.out is None else
              combine(args.run_dirs, args.out))
    print(markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
