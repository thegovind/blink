"""Run every requested scenario/task/seed/model against loopback blink servers.

    python -m examples.cua.harness.sweep --run-id mimo-a --seeds 1-10 --workers 8
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import hashlib
import http.server
import json
import os
import re
import subprocess
import sys
import threading
from contextlib import AsyncExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from playwright.async_api import async_playwright

from .agent import run_episode
from .client import BlinkClient, FanoutClient, MockClient
from .mark import screens
from .report import generate

ROOT = Path(__file__).resolve().parents[3]
REGISTRY = ROOT / "examples/cua/apps/registry.json"
RUNS = ROOT / "examples/cua/runs"


@dataclass(frozen=True)
class Job:
    scenario: dict
    task: dict
    seed: int
    model: str

    @property
    def path(self) -> Path:
        return Path(self.scenario["id"]) / f"{self.task['index']}-{self.seed}-{self.model.split('/')[-1]}"


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_args) -> None:
        pass


def _serve(apps_root: Path):
    if not apps_root.is_dir():
        raise FileNotFoundError(apps_root)
    handler = functools.partial(_Quiet, directory=str(apps_root.resolve()))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def _seeds(spec: str) -> list[int]:
    found = []
    for piece in spec.split(","):
        bounds = re.fullmatch(r"(\d+)(?:-|\.\.)(\d+)", piece.strip())
        if bounds:
            lo, hi = map(int, bounds.groups())
            if hi < lo:
                raise ValueError(f"Descending seed range: {piece}")
            found.extend(range(lo, hi + 1))
        else:
            found.append(int(piece))
    if not found or len(found) != len(set(found)):
        raise ValueError("At least one distinct seed is required")
    return found


def _sha() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                            text=True, check=False)
    if result.returncode:
        raise ValueError("Source has no .git; pass --harness-sha with the worktree's commit SHA")
    return result.stdout.strip()


def _load_registry(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    scenarios = data.get("scenarios", [data] if "id" in data else [])
    if not scenarios:
        raise ValueError(f"Registry has no scenarios: {path}")
    return scenarios


def _jobs(args, scenarios: list[dict], models: list[str]) -> list[Job]:
    if args.jobs_file:
        if args.scenario or args.task:
            raise ValueError("--jobs-file cannot be combined with --scenario or --task")
        manifest = json.loads(args.jobs_file.read_text(encoding="utf-8"))
        source_registry = manifest.get("source_registry_sha256")
        if source_registry and source_registry != hashlib.sha256(args.registry.read_bytes()).hexdigest():
            raise ValueError("Showcase jobs were selected from a different registry")
        by_id = {s["id"]: s for s in scenarios}
        jobs = []
        seen = set()
        for entry in manifest["jobs"]:
            scenario = by_id.get(entry["scenario"])
            if scenario is None:
                raise ValueError(f"Unknown showcase scenario: {entry['scenario']!r}")
            task = next((t for t in scenario["tasks"] if t["index"] == entry["task"]), None)
            if task is None or entry["model"] not in models:
                raise ValueError(f"Invalid showcase task/model: {entry}")
            seed = entry["seed"]
            if not isinstance(seed, int) or isinstance(seed, bool) or seed < 1:
                raise ValueError(f"Invalid showcase seed: {seed!r}")
            job = Job(scenario, task, seed, entry["model"])
            if str(job.path) in seen:
                raise ValueError(f"Duplicate showcase job: {job.path}")
            seen.add(str(job.path))
            jobs.append(job)
        if not jobs:
            raise ValueError("Showcase manifest has no jobs")
        return jobs
    chosen = set(args.scenario or (s["id"] for s in scenarios))
    missing = chosen - {s["id"] for s in scenarios}
    if missing:
        raise ValueError(f"Unknown scenarios: {sorted(missing)}")
    selected_tasks = set(args.task) if args.task else None
    jobs = []
    for scenario in scenarios:
        if scenario["id"] not in chosen:
            continue
        for task in scenario["tasks"]:
            if selected_tasks is not None and task["index"] not in selected_tasks:
                continue
            for model in models:
                for seed in _seeds(args.seeds):
                    jobs.append(Job(scenario, task, seed, model))
    if not jobs:
        raise ValueError("The scenario/task filters produced no jobs")
    return jobs


def _urls(args, models: list[str]) -> dict[str, list[str]]:
    urls = {model: [] for model in models}
    entries = args.server or [item.strip() for item in os.environ.get("BLINK_CUA_SERVERS", "").split(",")
                              if item.strip()]
    if not entries:
        raise ValueError("Supply --server model=URL or BLINK_CUA_SERVERS for every requested model")
    for entry in entries:
        name, separator, url = entry.partition("=")
        if not separator or name not in urls or not url:
            raise ValueError(f"Use --server model=URL for a requested model, got {entry!r}")
        urls[name].append(url)
    if any(not servers for servers in urls.values()):
        raise ValueError(f"Supply servers for every requested model: {models}")
    if args.fanout and any(len(set(entries)) < 3 for entries in urls.values()):
        raise ValueError("--fanout needs three different servers for each requested model")
    return urls


def _append(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(value, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _existing(run_dir: Path) -> set[str]:
    results = run_dir / "results.jsonl"
    if not results.exists():
        return set()
    seen = set()
    with results.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            path = row["episode"]
            if path in seen:
                raise ValueError(f"Duplicate episode in {results}: {path}")
            if not (run_dir / path / "summary.json").is_file():
                raise ValueError(f"Results row has no summary: {path}")
            seen.add(path)
    return seen


async def run(args) -> dict:
    if args.workers < 1 or args.beat < 0:
        raise ValueError("--workers must be >= 1 and --beat must be >= 0")
    if args.per_server_limit < 0 or (args.per_server_limit and args.fanout):
        raise ValueError("--per-server-limit must be >= 0 and cannot be combined with --fanout")
    if args.game_realtime and (args.beat or args.fanout):
        raise ValueError("--game-realtime requires --beat 0 and cannot use --fanout")
    models = list(dict.fromkeys(args.model or ["blink-mimo-9b"]))
    jobs = _jobs(args, _load_registry(args.registry), models)
    if args.game_realtime and any(job.scenario["kind"] != "game" for job in jobs):
        raise ValueError("--game-realtime requires only game scenarios")
    urls = {} if args.mock else _urls(args, models)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id) or run_id in {".", ".."}:
        raise ValueError(f"Unsafe run id: {run_id!r}")
    run_dir = args.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    harness_sha = args.harness_sha or _sha()
    selection = json.loads(args.jobs_file.read_text(encoding="utf-8")) if args.jobs_file else None
    config = {
        "harness_sha": harness_sha,
        "registry_sha256": hashlib.sha256(args.registry.read_bytes()).hexdigest(),
        "seeds": sorted({job.seed for job in jobs}) if args.jobs_file else _seeds(args.seeds),
        "models": models,
        "scenarios": sorted({job.scenario["id"] for job in jobs}),
        "thresholds": {"done_at": screens.DONE_AT, "risky_at": screens.RISKY_AT},
        "jobs": [str(job.path) for job in jobs],
        "interrupts": args.interrupts, "beat_ms": args.beat,
        "workers": args.workers,
        "per_server_limit": args.per_server_limit,
        "selection": ({"source_run": selection["source_run"],
                       "source_harness_sha": selection["source_harness_sha"],
                       "rule": selection["selection_rule"],
                       "manifest_sha256": hashlib.sha256(args.jobs_file.read_bytes()).hexdigest()}
                      if selection else None),
        "mode": "game-realtime" if args.game_realtime else "fanout" if args.fanout else "batched",
        "game_realtime": args.game_realtime,
        "mock": args.mock, "mock_noise": args.mock_noise,
        "servers": urls, "record_video": not args.no_video, "convert_video": not args.no_convert,
    }
    manifest = run_dir / "config.json"
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8")) != config:
            raise ValueError(f"{run_dir} has a different configuration; choose a new --run-id")
    else:
        manifest.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    seen = _existing(run_dir)
    results_file = run_dir / "results.jsonl"
    clients: dict[str, list[BlinkClient]] = {}
    if not args.mock:
        for model, entries in urls.items():
            clients[model] = [BlinkClient(url) for url in entries]
        for model, pool in clients.items():
            for client in pool:
                health = await client.http.get(f"{client.server}/healthz")
                health.raise_for_status()
                if not health.json().get("accepts_images"):
                    raise ValueError(f"{model} server {client.server} does not accept images")
    server_limits = ({client.server: asyncio.Semaphore(args.per_server_limit)
                      for pool in clients.values() for client in pool}
                     if args.per_server_limit else {})
    server, base_url = _serve(args.apps_root)
    lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(args.workers)
    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True, args=["--no-sandbox"])
            try:
                async def worker(index: int, job: Job) -> None:
                    episode = str(job.path)
                    if episode in seen:
                        return
                    async with AsyncExitStack() as stack:
                        if server_limits:
                            pool = clients[job.model]
                            await stack.enter_async_context(server_limits[pool[index % len(pool)].server])
                        await stack.enter_async_context(semaphore)
                        folder = run_dir / job.path
                        summary_path = folder / "summary.json"
                        if not summary_path.exists() and folder.exists():
                            archived = run_dir / "incomplete" / job.path.parent / job.path.name
                            archived.parent.mkdir(parents=True, exist_ok=True)
                            if archived.exists():
                                archived = archived.with_name(
                                    f"{archived.name}-{datetime.now(timezone.utc).strftime('%H%M%S%f')}")
                            os.replace(folder, archived)
                            _append(run_dir / "incomplete.jsonl",
                                    {"episode": episode, "archived_at": str(archived.relative_to(run_dir)),
                                     "reason": "interrupted before summary was written"})
                        if summary_path.exists():
                            summary = json.loads(summary_path.read_text(encoding="utf-8"))
                            if (summary["harness_sha"] != harness_sha or summary["seed"] != job.seed
                                    or summary["interrupts"] != args.interrupts or summary["model"] != job.model):
                                raise ValueError(f"Existing summary does not match this job: {summary_path}")
                        else:
                            if args.mock:
                                client = MockClient(seed=job.seed * 1009 + job.task["index"] * 17 +
                                                    sum(map(ord, job.scenario["id"])), noise=args.mock_noise)
                            elif args.fanout:
                                pool = clients[job.model]
                                client = FanoutClient([pool[(index + i) % len(pool)] for i in range(3)])
                            else:
                                pool = clients[job.model]
                                client = pool[index % len(pool)]
                            summary = await run_episode(
                                browser, base_url=base_url, scenario=job.scenario, task=job.task,
                                seed=job.seed, model=job.model, client=client, output=folder,
                                harness_sha=harness_sha, interrupts=args.interrupts, beat_ms=args.beat,
                                record_video=not args.no_video, convert_video=not args.no_convert,
                                mode="game-realtime" if args.game_realtime else
                                "fanout" if args.fanout else "batched",
                                game_realtime=args.game_realtime,
                            )
                        async with lock:
                            _append(results_file, {"episode": episode, **summary})
                            oracle_note = (f", oracle_issues={summary['oracle_issues']}"
                                           if summary["oracle_issues"] else "")
                            print(f"{episode}: {'PASS' if summary['success'] else 'FAIL'} "
                                  f"({summary['steps']} actions, {summary['stop_reason']}{oracle_note})",
                                  file=sys.stderr, flush=True)

                await asyncio.gather(*(worker(i, job) for i, job in enumerate(jobs)))
            finally:
                await browser.close()
    finally:
        server.shutdown()
        server.server_close()
        await asyncio.gather(*(client.close() for pool in clients.values() for client in pool))
    return generate(run_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path,
                        default=Path(os.environ.get("BLINK_CUA_REGISTRY", str(REGISTRY))))
    parser.add_argument("--apps-root", type=Path)
    parser.add_argument("--runs-dir", type=Path,
                        default=Path(os.environ.get("BLINK_CUA_RUNS_DIR", str(RUNS))))
    parser.add_argument("--run-id")
    parser.add_argument("--harness-sha", help="required when the source copy has no .git")
    parser.add_argument("--scenario", action="append", help="scenario id; repeat to filter")
    parser.add_argument("--task", type=int, action="append", help="task index; repeat to filter")
    parser.add_argument("--seeds", default="1", help="e.g. 1-10, 1..5 or 1,3,5")
    parser.add_argument("--jobs-file", type=Path, help="exact selected scenario/task/seed/model jobs")
    parser.add_argument("--model", action="append", help="model name; repeat to compare")
    parser.add_argument("--server", action="append", help="model=URL; repeat per server; or set BLINK_CUA_SERVERS")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--per-server-limit", type=int, default=0,
                        help="maximum concurrent episodes per assigned server; 0 is unlimited")
    parser.add_argument("--interrupts", type=int, choices=(0, 1), default=0)
    parser.add_argument("--beat", type=int, default=600, help="video display beat in ms; 0 = real time")
    parser.add_argument("--fanout", action="store_true", help="three single-question requests in parallel")
    parser.add_argument("--game-realtime", action="store_true",
                        help="run only game scenarios with an unpaused clock; requires --beat 0")
    parser.add_argument("--mock", action="store_true", help="CPU oracle engine, never counted as model results")
    parser.add_argument("--mock-noise", type=float, default=0.0)
    parser.add_argument("--no-video", action="store_true", help="tests only: do not record")
    parser.add_argument("--no-convert", action="store_true", help="tests only: keep WebM without previews")
    args = parser.parse_args(argv)
    if args.apps_root is None:
        args.apps_root = Path(os.environ.get("BLINK_CUA_APPS_ROOT", str(args.registry.parent)))
    report = asyncio.run(run(args))
    print(f"Report: {args.runs_dir / report['run_id'] / 'report.md'} "
          f"({report['total']['successes']}/{report['total']['episodes']} succeeded)")
    return 1 if report["total"]["errors"] or report["total"]["video_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
