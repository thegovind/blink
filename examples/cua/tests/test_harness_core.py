"""CPU contract checks; no model, hosted service or external page is used."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path

import httpx
import pytest
from PIL import Image
from playwright.async_api import async_playwright

from examples.cua.harness.agent import _oracle_box, _plain_reason, run_episode
from examples.cua.harness.client import BlinkClient, BlinkClientError, Decision, FanoutClient, MockClient
from examples.cua.harness.dom import candidates, matches_expected
from examples.cua.harness.mark import PAD_CSS, mark_screenshot, screens
from examples.cua.harness.sweep import _serve
from examples.cua.harness.video import poster_frame

FIXTURES = Path(__file__).with_name("fixtures")
STUB = json.loads((FIXTURES / "stub/scenario.json").read_text(encoding="utf-8"))


def test_dom_order_outermost_visibility_occlusion_and_marks():
    async def run():
        server, base = _serve(FIXTURES)
        try:
            async with async_playwright() as pw:
                browser = await pw.chromium.launch()
                try:
                    page = await browser.new_page(viewport=STUB["viewport"], device_scale_factor=1.5)
                    await page.goto(f"{base}/stub/index.html?seed=2&task=0&interrupts=1")
                    found = await candidates(page)
                    assert [item.selector for item in found] == ["#search", "#search-button"]
                    assert [item.name for item in found] == ["Search catalog", "Search"]
                    assert await matches_expected(page, found[1], {"selector": "#search-button"})
                    assert not await matches_expected(page, found[0], {"selector": "#search-button"})
                    raw = await page.screenshot(type="png", scale="device")
                    marked = mark_screenshot(raw, found, viewport=STUB["viewport"], scale=1.5)
                    assert marked.size == (1440, 864)
                    assert marked.png != raw
                    assert marked.boxes[0].box[0] == round((found[0].rect["x"] - PAD_CSS) * 1.5)
                    assert marked.boxes[0].tag == screens.layout(
                        [b.box for b in marked.boxes], marked.size,
                        meta=[{"role": c.role, "name": c.name} for c in found],
                    )[0].tag
                    assert marked.shot("Search").uri.startswith("data:image/png;base64,")
                    ordering = await browser.new_page(viewport=STUB["viewport"])
                    await ordering.set_content(
                        "<button id='left' style='position:absolute;left:20px;top:12px;"
                        "width:90px;height:34px'>Left</button>"
                        "<button id='right' style='position:absolute;left:300px;top:2px;"
                        "width:90px;height:34px'>Right</button>"
                        "<button id='middle' style='position:absolute;left:160px;top:8px;"
                        "width:90px;height:34px'>Middle</button>"
                    )
                    assert [c.name for c in await candidates(ordering)] == ["Left", "Middle", "Right"]
                    await ordering.close()
                    await page.locator("#search").fill("Amber Plan")
                    await page.locator("#search-button").click()
                    assert [c.selector for c in await candidates(page)] == ["#item", "#dismiss"]
                    await page.locator("#dismiss").click()
                    await page.locator("#item").click()
                    await page.locator("#delivery").select_option("Blue")
                    await page.locator("#publish").click()
                    assert [c.selector for c in await candidates(page)] == ["#confirm"]
                finally:
                    await browser.close()
        finally:
            server.shutdown()
            server.server_close()

    asyncio.run(run())


def test_off_path_oracle_is_explicit_but_not_a_harness_crash():
    async def run():
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            try:
                page = await browser.new_page()
                await page.set_content('<button id="back">Back</button>')
                found = await candidates(page)
                number, issue = await _oracle_box(
                    page, found, {"selector": "#missing", "action": "click", "done": False}
                )
                assert number is None and "absent from the current DOM" in issue
                number, issue = await _oracle_box(
                    page, found, {"selector": "#back", "action": "click", "done": False}
                )
                assert number == "1" and issue is None
            finally:
                await browser.close()

    asyncio.run(run())


def test_off_path_episode_records_oracle_issues_and_continues(tmp_path):
    scenario = {"id": "oracle-missing-stub", "path": "stub/oracle-missing.html",
                "viewport": STUB["viewport"], "scale": 1.5, "kind": "dom",
                "tasks": [{"index": 0, "optimal_steps": 3}]}

    class ScriptedClient:
        server = "scripted"

        async def ask(self, _state, _questions, **_kwargs):
            return Decision(
                answers={"element": {"choice": "1", "probabilities": {"1": 1.0}},
                         "done": {"noul": 0.1, "probabilities": {"yes": 0.1, "no": 0.9}},
                         "risky": {"noul": 0.1, "probabilities": {"yes": 0.1, "no": 0.9}}},
                usage={"input_tokens": 0, "visual_tokens": 0},
                wall_ms=0.0, server=self.server,
            )

    async def run():
        server, base = _serve(FIXTURES)
        try:
            async with async_playwright() as pw:
                browser = await pw.chromium.launch()
                try:
                    return await run_episode(
                        browser, base_url=base, scenario=scenario, task=scenario["tasks"][0],
                        seed=1, model="scripted", client=ScriptedClient(), output=tmp_path,
                        harness_sha="test-sha", record_video=False,
                    )
                finally:
                    await browser.close()
        finally:
            server.shutdown()
            server.server_close()

    summary = asyncio.run(run())
    assert summary["error"] is None and not summary["success"]
    assert summary["stop_reason"] == "repeated_action"
    assert summary["element_total"] == summary["oracle_issues"] == 3
    rows = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert len(rows) == 3
    assert all("absent from the current DOM" in row["oracle_issue"] for row in rows)
    assert all("oracle_issue" not in row["request_state"] for row in rows)


@pytest.mark.parametrize(("interrupts", "actions"), [(0, 6), (1, 7)])
def test_mock_episode_oracle_is_not_sent_or_logged(tmp_path, interrupts, actions):
    async def run():
        server, base = _serve(FIXTURES)
        try:
            async with async_playwright() as pw:
                browser = await pw.chromium.launch()
                try:
                    return await run_episode(
                        browser, base_url=base, scenario=STUB, task=STUB["tasks"][0],
                        seed=4, model="mock", client=MockClient(4), output=tmp_path,
                        harness_sha="test-sha", interrupts=interrupts, beat_ms=0,
                        record_video=True, convert_video=False,
                    )
                finally:
                    await browser.close()
        finally:
            server.shutdown()
            server.server_close()

    result = asyncio.run(run())
    assert result["success"] and result["error"] is None
    assert result["reason"] == "task completed"
    video_t0 = datetime.fromisoformat(result["video_t0"])
    assert result["steps"] == actions
    assert result["element_correct"] == result["element_total"] == actions
    assert (result["done_tp"], result["risk_tp"], result["risk_fn"], result["risk_fp"]) == (1, 1, 0, 0)
    assert result["gates"] == 1 and result["harness_sha"] == "test-sha"
    assert (tmp_path / "video.webm").stat().st_size > 1000
    rows = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert len(rows) == actions + 1
    assert rows[-1]["done_predicted"] and rows[-1]["candidates"] == []
    assert rows[-2]["gate_fired"] and rows[-2]["expected_risky"]
    assert [r["chosen_action"]["memory"] for r in rows if r["chosen_action"]["memory"]][0] == (
        'typed "Amber Plan" into box 1 (Search catalog)'
    )
    for row in rows:
        assert 0 <= row["video_ms"]
        assert datetime.fromisoformat(row["t_start"]) >= video_t0
        assert abs(row["video_ms"] -
                   (datetime.fromisoformat(row["t_start"]) - video_t0).total_seconds() * 1000) < 250
        assert set(row["request_state"]) == {"task", "previous_actions", "screenshot_note"}
        assert "expected" not in row["request_state"]
        assert "candidates" not in row["request_state"]
        assert not any("selector" in str(q) for q in row["request_questions"].values())
        assert row["wall_ms"] == row["main_ms"] + row["followup_ms"]
    with Image.open(tmp_path / "steps/001.png") as image:
        assert image.getpixel((1400, 820)) == (244, 246, 250)  # the HUD was never in the model image


def test_mock_noise_produces_measured_failure(tmp_path):
    async def run():
        server, base = _serve(FIXTURES)
        try:
            async with async_playwright() as pw:
                browser = await pw.chromium.launch()
                try:
                    return await run_episode(
                        browser, base_url=base, scenario=STUB, task=STUB["tasks"][0],
                        seed=1, model="mock", client=MockClient(seed=1, noise=1),
                        output=tmp_path, harness_sha="test-sha", record_video=False,
                    )
                finally:
                    await browser.close()
        finally:
            server.shutdown()
            server.server_close()

    result = asyncio.run(run())
    assert not result["success"]
    assert result["stop_reason"] == "model_done"
    assert result["reason"] == "said done too early" and result["video_t0"] is None
    assert result["steps"] == 0 and result["done_fp"] == 1
    assert json.loads((tmp_path / "trace.jsonl").read_text().splitlines()[0])["video_ms"] is None


@pytest.mark.parametrize(("detail", "stop_reason", "expected"), [
    ("target backend role detail not open", "model_done", "didn't open the target job"),
    ("cart has 3 lines", "step_limit", "cart still has the wrong items"),
    ("2 newsletters remain", "model_done", "newsletters still need archiving"),
    ("wave 20/20, hits 2", "game_finished", "took too many hits"),
    ("wrong key deleted: abc123", "model_done", "deleted the wrong key"),
    ("UnexpectedTypeError: stack trace", "model_done", "said done too early"),
])
def test_gallery_reason_keeps_app_detail_private_to_raw_field(detail, stop_reason, expected):
    assert _plain_reason({"success": False, "detail": detail}, stop_reason, None) == expected
    assert _plain_reason({"success": True, "detail": detail}, stop_reason, None) == "task completed"
    assert _plain_reason({"success": False, "detail": detail}, "error", "NetworkError") == (
        "episode could not finish"
    )


def test_wire_payload_validation_and_parallel_fanout():
    async def run():
        requests = []
        clients = []
        questions = screens.questions(2)
        state = {"task": "fixture", "previous_actions": [],
                 "screenshot_note": "Use boxes 1-2", "screenshot": "data:image/png;base64,AAAA"}
        try:
            for port in (12001, 12002, 12003):
                client = BlinkClient(f"http://127.0.0.1:{port}")

                async def handler(request, *, port=port):
                    payload = json.loads(request.content)
                    requests.append((port, payload))
                    await asyncio.sleep(0.01)
                    answers = {}
                    for key, q in payload["questions"].items():
                        answers[key] = ({"choice": "1", "probabilities": {"1": 0.75, "2": 0.25}}
                                        if q["type"] == "choice" else
                                        {"noul": 0.1, "probabilities": {"yes": 0.1, "no": 0.9}})
                    return httpx.Response(200, json={"answers": answers,
                                                     "usage": {"input_tokens": 6, "visual_tokens": 5}})

                await client.http.aclose()
                client.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
                clients.append(client)
            answer = await FanoutClient(clients).ask(state, questions)
            assert set(answer.answers) == {"element", "done", "risky"}
            assert answer.usage == {"input_tokens": 18, "visual_tokens": 15}
            assert set(answer.per_question_ms) == set(questions)
            assert [len(p["questions"]) for _, p in requests] == [1, 1, 1]
            assert {port for port, _ in requests} == {12001, 12002, 12003}
            assert all(p["model"] == "blink" and p["state"] == state for _, p in requests)
            assert answer.wall_ms < sum(answer.per_question_ms.values())
        finally:
            await asyncio.gather(*(c.close() for c in clients))

    asyncio.run(run())


def test_server_rejects_invalid_answers_and_urls():
    with pytest.raises(ValueError, match="http"):
        BlinkClient("ftp://example.com")
    with pytest.raises(ValueError, match="credentials"):
        BlinkClient("https://user:secret@localhost")

    async def run():
        remote = BlinkClient("https://example.com")
        assert remote.server == "https://example.com"
        await remote.close()
        client = BlinkClient("http://127.0.0.1:12004")
        await client.http.aclose()
        client.http = httpx.AsyncClient(transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, json={"answers": {}, "usage": {}})
        ))
        try:
            with pytest.raises(BlinkClientError, match="probabilities"):
                await client.ask({"task": "fixture"}, screens.questions(1))
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    ("path", "kind", "actions", "risky"),
    [("stub/canvas.html", "canvas", 3, 1), ("stub/game.html", "game", 5, 0)],
)
def test_canvas_targets_and_game_ticks_use_pixels_not_target_names(
    tmp_path, path, kind, actions, risky,
):
    scenario = {"id": f"{kind}-stub", "path": path, "viewport": STUB["viewport"],
                "scale": 1.5, "kind": kind, "cadence_ms": 20,
                "tasks": [{"index": 0, "optimal_steps": actions}]}

    async def run():
        server, base = _serve(FIXTURES)
        try:
            async with async_playwright() as pw:
                browser = await pw.chromium.launch()
                try:
                    return await run_episode(
                        browser, base_url=base, scenario=scenario, task=scenario["tasks"][0],
                        seed=2, model="mock", client=MockClient(2), output=tmp_path,
                        harness_sha="test-sha", record_video=False, beat_ms=0,
                    )
                finally:
                    await browser.close()
        finally:
            server.shutdown()
            server.server_close()

    summary = asyncio.run(run())
    assert summary["success"] and summary["steps"] == actions, summary
    assert summary["element_correct"] == summary["element_total"] == actions
    assert summary["gates"] == risky
    assert summary["candidates_source"] == "canvas hit regions"
    if kind == "game":
        assert summary["waves_survived"] == 5 and summary["hits"] == 0
        assert summary["done_tp"] == summary["risk_tp"] == 0
    rows = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    for row in rows:
        assert row["request_state"]["previous_actions"] == [
            prior["chosen_action"]["memory"] for prior in rows[:row["step"] - 1]
            if prior["chosen_action"]["memory"]
        ]
        assert "name" not in str(row["request_state"])
        assert "target" not in str(row["request_state"])
        assert all("(lane" not in memory and "(Open" not in memory
                   for memory in row["request_state"]["previous_actions"])
        if kind == "game":
            assert set(row["request_questions"]) == {"lane"}


def test_unoffered_text_value_is_a_scored_model_failure(tmp_path):
    registry = Path(__file__).parents[1] / "apps/registry.json"
    if not registry.exists():
        pytest.skip("Phone app has not merged")
    phone = next((s for s in json.loads(registry.read_text())["scenarios"]
                  if s["id"] == "phone"), None)
    if phone is None:
        pytest.skip("Phone app has not merged")

    class WrongTextClient(MockClient):
        async def ask(self, state, questions, **kwargs):
            decision = await super().ask(state, questions, **kwargs)
            if "element" not in questions:
                return decision
            probabilities = {option: (0.9 if option == "2" else
                             0.1 / (len(questions["element"]["criteria"]) - 1))
                             for option in questions["element"]["criteria"]}
            return Decision({**decision.answers,
                             "element": {"choice": "2", "probabilities": probabilities}},
                            decision.usage, decision.wall_ms, decision.server)

    async def run():
        server, base = _serve(registry.parent)
        try:
            async with async_playwright() as pw:
                browser = await pw.chromium.launch()
                try:
                    return await run_episode(
                        browser, base_url=base, scenario=phone, task=phone["tasks"][0],
                        seed=1, model="mock", client=WrongTextClient(1),
                        output=tmp_path, harness_sha="test-sha", record_video=False,
                    )
                finally:
                    await browser.close()
        finally:
            server.shutdown()
            server.server_close()

    result = asyncio.run(run())
    assert not result["success"] and result["error"] is None
    assert result["stop_reason"] == "no_task_value" and result["steps"] == 0
    assert result["element_total"] == 1 and result["element_correct"] == 0
    row = json.loads((tmp_path / "trace.jsonl").read_text().splitlines()[0])
    assert row["action"] == {"kind": "invalid_action", "box": "2"}
    assert "no values" in row["invalid_action"]
    assert row["chosen_action"]["memory"] is None


def test_video_hud_stays_clear_of_phone_nav_and_game_lanes():
    script = Path(__file__).parents[1] / "harness/overlay.js"

    async def run():
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            try:
                for viewport, kind, width, model in [
                    ({"width": 390, "height": 844}, "dom", 585, "blink-mimo-9b"),
                    ({"width": 960, "height": 576}, "game", 1440, "blink-mimo-9b"),
                    ({"width": 960, "height": 576}, "game", 1440, "blink-4b"),
                    ({"width": 960, "height": 576}, "game", 1440, "blink-27b"),
                    ({"width": 960, "height": 576}, "canvas", 1440, "blink-27b"),
                ]:
                    page = await browser.new_page(viewport=viewport)
                    await page.set_content("<html><body></body></html>")
                    await page.add_script_tag(path=str(script))
                    await page.evaluate(
                        "({width,kind,model}) => window.blinkCUAOverlay.show([],1.5,width,kind,model)",
                        {"width": width, "kind": kind, "model": model},
                    )
                    assert model in (await page.locator(".blink-cua-eyebrow").inner_text()).lower()
                    await page.evaluate("""() => window.blinkCUAOverlay.decide({
                      number:"1", action:"Click 1", probabilities:{"1":1},
                      done:0.1, risky:0.1, ms:123, gated:false
                    })""")
                    assert model in (await page.locator(".blink-cua-eyebrow").inner_text()).lower()
                    assert await page.locator(".blink-cua-brand").evaluate(
                        "(el) => el.scrollWidth <= el.clientWidth"
                    )
                    hud = await page.locator(".blink-cua-status").bounding_box()
                    if kind == "game":
                        assert hud["x"] >= 760
                    elif viewport["width"] < 500:
                        assert hud["y"] + hud["height"] <= viewport["height"] - 82
                    await page.close()
            finally:
                await browser.close()

    asyncio.run(run())


def test_short_video_poster_falls_back_even_when_ffmpeg_exits_zero(tmp_path, monkeypatch):
    webm = tmp_path / "video.webm"
    webm.write_bytes(b"fixture")
    poster = tmp_path / "poster.png"
    seeks = []

    def fake_ffmpeg(*args):
        seeks.append(args[1])
        if args[1] == "0":
            poster.write_bytes(b"\x89PNG\r\n\x1a\nframe")

    monkeypatch.setattr("examples.cua.harness.video._run", fake_ffmpeg)
    poster_frame(webm, poster)
    assert seeks == ["2", "1", "0"]
    assert poster.read_bytes().startswith(b"\x89PNG")

    poster.unlink()
    monkeypatch.setattr("examples.cua.harness.video._run", lambda *_args: None)
    with pytest.raises(RuntimeError, match="could not extract any frame"):
        poster_frame(webm, poster)


def test_video_conversion_bounds_encoder_threads(tmp_path, monkeypatch):
    from examples.cua.harness.video import convert, normalize_capture

    capture = tmp_path / "video.capture.webm"
    capture.write_bytes(b"capture")
    webm = tmp_path / "video.webm"
    commands = []

    def fake_ffmpeg(*args):
        commands.append(args)
        assert args[-3:-1] == ("-threads", "2")
        Path(args[-1]).write_bytes(b"encoded frame")

    monkeypatch.setattr("examples.cua.harness.video._run", fake_ffmpeg)
    normalize_capture(capture, webm, viewport={"width": 390, "height": 844}, scale=1.5)
    files = convert(webm)
    assert len(commands) == 4
    assert "scale=585:1266" in commands[0][commands[0].index("-vf") + 1]
    assert commands[1][commands[1].index("-vf") + 1] == (
        "pad=ceil(iw/2)*2:ceil(ih/2)*2:0:0:color=black"
    )
    assert all((tmp_path / name).is_file() for name in files.values())
