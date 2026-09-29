"""Resumption, aggregation and interval accounting on the local contract fixture."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from playwright.async_api import async_playwright

from examples.cua.harness import report, sweep
from examples.cua.harness.agent import run_episode
from examples.cua.harness.client import MockClient

FIXTURES = Path(__file__).with_name("fixtures")
REGISTRY = FIXTURES / "stub/scenario.json"


def test_wilson_and_denominators():
    lo, hi = report.wilson(1, 2)
    assert lo == pytest.approx(0.0945, abs=0.001)
    assert hi == pytest.approx(0.9055, abs=0.001)
    assert report.wilson(0, 0) is None
    row = {"success": True, "steps": 6, "optimal_steps": 6, "stop_reason": "model_done",
           "element_correct": 5, "element_total": 6, "done_tp": 1, "done_fp": 0,
           "done_fn": 0, "done_tn": 6, "risk_tp": 1, "risk_fp": 1, "risk_fn": 1,
           "risk_tn": 3, "gates": 2, "wall_ms": [100, 200, 300],
           "waves_survived": None, "hits": None}
    second = {**row, "success": False, "steps": 8, "stop_reason": "step_limit",
              "done_tp": 0, "done_fp": 1, "done_fn": 1, "risk_tp": 0,
              "risk_fp": 0, "risk_fn": 1, "wall_ms": [400]}
    a = report.aggregate([row, second])
    assert a["success_rate"] == 0.5
    assert a["mean_steps"] == 7 and a["mean_optimal_steps"] == 6
    assert a["element_accuracy"] == pytest.approx(10 / 12)
    assert a["done_precision"] == 0.5 and a["done_recall"] == 0.5
    assert a["risk_gate_recall"] == pytest.approx(1 / 3)
    assert a["false_gate_rate"] == pytest.approx(1 / 7)
    assert a["p50_decision_ms"] == 250 and a["p95_decision_ms"] == 385
    assert a["mean_waves_survived"] is None and a["mean_hits"] is None


def test_mock_sweep_resumes_without_cherry_picking(tmp_path):
    args = ["--registry", str(REGISTRY), "--apps-root", str(FIXTURES),
            "--runs-dir", str(tmp_path), "--run-id", "fixture-sweep", "--seeds", "1-2",
            "--mock", "--no-video", "--workers", "2", "--harness-sha", "test-sha"]
    assert sweep.main(args) == 0
    path = tmp_path / "fixture-sweep"
    results = (path / "results.jsonl").read_text().splitlines()
    assert len(results) == 2
    assert all(json.loads(line)["success"] for line in results)
    config = json.loads((path / "config.json").read_text())
    assert config["harness_sha"] == "test-sha"
    assert config["seeds"] == [1, 2] and config["models"] == ["blink-mimo-9b"]
    assert config["thresholds"] == {"done_at": 0.60, "risky_at": 0.50}
    assert config["workers"] == 2 and config["game_realtime"] is False
    assert sweep.main(args) == 0
    assert (path / "results.jsonl").read_text().splitlines() == results
    metrics = json.loads((path / "report.json").read_text())
    assert metrics["total"]["episodes"] == 2 and metrics["total"]["successes"] == 2
    assert "Wilson 95%" in (path / "report.md").read_text()
    with pytest.raises(ValueError, match="different configuration"):
        sweep.main([*args, "--beat", "0"])


def test_seed_ranges_and_invalid_fanout():
    assert sweep._seeds("1-3,5..6") == [1, 2, 3, 5, 6]
    with pytest.raises(ValueError, match="Descending"):
        sweep._seeds("4-2")
    with pytest.raises(ValueError, match="distinct"):
        sweep._seeds("1,1")
    with pytest.raises(ValueError, match="requires --beat 0"):
        sweep.main(["--registry", str(REGISTRY), "--apps-root", str(FIXTURES),
                    "--mock", "--game-realtime"])
    with pytest.raises(ValueError, match="requires only game scenarios"):
        sweep.main(["--registry", str(REGISTRY), "--apps-root", str(FIXTURES),
                    "--mock", "--game-realtime", "--beat", "0"])


def test_realtime_game_sweep_keeps_its_own_mode_and_config(tmp_path):
    game = {"id": "game-stub", "path": "stub/game.html",
            "viewport": {"width": 960, "height": 576}, "scale": 1.5,
            "kind": "game", "cadence_ms": 20,
            "tasks": [{"index": 0, "optimal_steps": 5}], "states": []}
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"version": 1, "scenarios": [game]}))
    args = ["--registry", str(registry), "--apps-root", str(FIXTURES),
            "--runs-dir", str(tmp_path), "--run-id", "game-realtime",
            "--seeds", "1", "--mock", "--no-video", "--workers", "1",
            "--beat", "0", "--game-realtime", "--harness-sha", "test-sha"]
    assert sweep.main(args) == 0
    run_dir = tmp_path / "game-realtime"
    config = json.loads((run_dir / "config.json").read_text())
    assert config["mode"] == "game-realtime" and config["game_realtime"]
    assert config["seeds"] == [1] and config["models"] == ["blink-mimo-9b"]
    result, = report.load_results(run_dir)
    assert result["success"] and result["game_realtime"]
    assert result["mode"] == "game-realtime" and result["steps"] == 5


def test_combine_disjoint_runs_without_duplicate_episodes(tmp_path):
    folders = [tmp_path / "partial-a", tmp_path / "partial-b"]
    base = {"scenario": "stub", "task_index": 0, "model": "blink-mimo-9b",
            "interrupts": 0, "mode": "batched", "harness_sha": "test-sha",
            "success": True, "steps": 3, "optimal_steps": 4,
            "stop_reason": "model_done", "wall_ms": [200],
            "waves_survived": None, "hits": None}
    for seed, folder in enumerate(folders, 1):
        folder.mkdir()
        (folder / "results.jsonl").write_text(
            json.dumps({**base, "seed": seed}) + "\n", encoding="utf-8"
        )
    combined = report.combine(folders, tmp_path / "all-a")
    assert combined["total"]["episodes"] == 2
    assert len(combined["sources"]) == 2
    assert "partial-a (test-sha)" in (tmp_path / "all-a/report.md").read_text()
    assert "optimal_steps_exact" in (tmp_path / "all-a/report.md").read_text()
    (folders[1] / "results.jsonl").write_text(
        json.dumps({**base, "seed": 1}) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="multiple source runs"):
        report.combine(folders, tmp_path / "bad")


@pytest.mark.parametrize("interrupts", [0, 1])
def test_every_merged_app_task_runs_under_cpu_oracle(tmp_path, interrupts):
    registry = sweep.REGISTRY
    if not registry.is_file():
        pytest.skip("No app builders have merged yet")
    scenarios = [s for s in sweep._load_registry(registry) if s["kind"] != "game"]
    tasks = sum(len(s["tasks"]) for s in scenarios)
    assert tasks > 0
    args = ["--registry", str(registry), "--apps-root", str(registry.parent),
            "--runs-dir", str(tmp_path), "--run-id", "merged-apps", "--seeds", "1",
            "--mock", "--no-video", "--workers", "2", "--harness-sha", "test-sha",
            "--interrupts", str(interrupts)]
    for scenario in scenarios:
        args.extend(["--scenario", scenario["id"]])
    assert sweep.main(args) == 0
    rows = report.load_results(tmp_path / "merged-apps")
    assert len(rows) == tasks
    failures = [(r["scenario"], r["task_index"], r["error"], r["detail"])
                for r in rows if not r["success"]]
    assert not failures, failures


def test_merged_game_uses_lockstep_cadence_and_scores_waves(tmp_path):
    if not sweep.REGISTRY.is_file():
        pytest.skip("No app builders have merged yet")
    game = next((s for s in sweep._load_registry(sweep.REGISTRY) if s["kind"] == "game"), None)
    if game is None:
        pytest.skip("Game app has not merged yet")
    short_task = {**game["tasks"][0], "optimal_steps": 1}

    async def run():
        server, base = sweep._serve(sweep.REGISTRY.parent)
        try:
            async with async_playwright() as pw:
                browser = await pw.chromium.launch()
                try:
                    return await run_episode(
                        browser, base_url=base, scenario=game, task=short_task,
                        seed=1, model="mock", client=MockClient(1), output=tmp_path,
                        harness_sha="test-sha", beat_ms=0, record_video=False,
                    )
                finally:
                    await browser.close()
        finally:
            server.shutdown()
            server.server_close()

    summary = asyncio.run(run())
    assert summary["error"] is None, summary
    assert summary["game_clock"] and summary["game_cadence_ms"] == game["cadence"]["slow_ms"]
    assert summary["step_limit"] == 12
    assert summary["steps"] == 12 and summary["element_correct"] == 12
    assert summary["waves_survived"] >= 1 and summary["hits"] == 0
    metrics = report.aggregate([summary])
    assert metrics["mean_waves_survived"] == summary["waves_survived"]
    assert "Game waves / hits" in report.markdown(
        {"run_id": "game-smoke", "total": metrics,
         "groups": [{"model": "mock", "scenario": "game", "interrupts": 0,
                     "mode": "batched", "metrics": metrics}]}
    )
    rows = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert len(rows) == 12
    assert all(row["game_advance_ms"] == 400 and set(row["request_questions"]) == {"lane"}
               for row in rows)


def test_merged_game_realtime_keeps_world_unpaused(tmp_path):
    if not sweep.REGISTRY.is_file():
        pytest.skip("No app builders have merged yet")
    game = next((s for s in sweep._load_registry(sweep.REGISTRY) if s["kind"] == "game"), None)
    if game is None:
        pytest.skip("Game app has not merged yet")
    short_task = {**game["tasks"][0], "optimal_steps": 1}

    async def run():
        server, base = sweep._serve(sweep.REGISTRY.parent)
        try:
            async with async_playwright() as pw:
                browser = await pw.chromium.launch()
                try:
                    return await run_episode(
                        browser, base_url=base, scenario=game, task=short_task,
                        seed=1, model="mock", client=MockClient(1), output=tmp_path,
                        harness_sha="test-sha", beat_ms=0, record_video=False,
                        game_realtime=True,
                    )
                finally:
                    await browser.close()
        finally:
            server.shutdown()
            server.server_close()

    summary = asyncio.run(run())
    assert summary["error"] is None, summary
    assert summary["game_realtime"] and summary["mode"] == "game-realtime"
    assert summary["game_clock"].startswith("real-time:")
    assert summary["game_cadence_ms"] == game["cadence"]["slow_ms"]
    assert summary["hits"] is not None and summary["waves_survived"] is not None
    rows = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert all("paused=1" not in row["url"] and row["game_advance_ms"] == 0
               for row in rows)
