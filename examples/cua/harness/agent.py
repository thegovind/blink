"""One episode: screenshot -> typed decisions -> visible action -> independent check."""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

from playwright.async_api import Browser, Page

from .client import BlinkClient, Decision, FanoutClient, MockClient
from .dom import Candidate, candidates, matches_expected
from .mark import mark_screenshot, screens

Client = BlinkClient | FanoutClient | MockClient
_OVERLAY = Path(__file__).with_name("overlay.js")
_TEXT_TYPES = {"text", "search", "email", "url", "tel", "number", "password"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    position = (len(sorted_values) - 1) * percentile
    low = int(position)
    return round(sorted_values[low] + (sorted_values[min(low + 1, len(values) - 1)]
                                        - sorted_values[low]) * (position - low), 2)


def _action_kind(candidate: Candidate) -> str:
    if candidate.tag == "select":
        return "select"
    if candidate.tag == "textarea" or (candidate.tag == "input" and candidate.input_type in _TEXT_TYPES):
        return "type"
    return "click"


def _memory(kind: str, candidate: Candidate, value: str | None) -> str:
    target = f"box {candidate.n}"
    if candidate.tag != "canvas" and candidate.name:
        target += f" ({candidate.name})"
    if kind == "type":
        return f'typed "{value}" into {target}'
    if kind == "select":
        return f'selected "{value}" in {target}'
    return f"clicked {target}"


def _select_options(options: list[dict]) -> list[dict]:
    return [{"index": i, "label": str(option["label"]), "value": str(option["value"])}
            for i, option in enumerate(options) if not option["disabled"]]


async def _followup(page: Page, candidate: Candidate, state: dict, expected: dict,
                    client: Client, values: list[str]) -> tuple[Decision, dict, str, int | None]:
    kind = _action_kind(candidate)
    if kind == "type":
        if not values:
            raise ValueError(f"Box {candidate.n} is a text field, but the task has no values")
        options = [{"index": i, "label": value, "value": value} for i, value in enumerate(values)]
        instruction = (f"Which value from the task belongs in the selected field (box {candidate.n})?")
    else:
        raw = await page.locator(candidate.selector).evaluate(
            "(el) => [...el.options].map(o => ({label:o.label, value:o.value, disabled:o.disabled}))"
        )
        options = _select_options(raw)
        if not options:
            raise ValueError(f"Box {candidate.n} is a select with no enabled options")
        instruction = f"Which option should be selected in box {candidate.n} to make progress on the task?"
    question = {"value": {"type": "choice", "instructions": instruction,
                          "criteria": {str(i): option["label"] for i, option in enumerate(options, 1)}}}
    oracle_choice = None
    if isinstance(client, MockClient):
        wanted = expected.get("value")
        oracle_choice = next((str(i) for i, o in enumerate(options, 1)
                              if wanted is not None and wanted in (o["label"], o["value"])), None)
        if oracle_choice is None:
            if expected.get("selector") == candidate.selector:
                raise ValueError(f"Oracle value {wanted!r} is not among {options}")
            oracle_choice = "1"
    response = await client.ask(state, question, expected=expected,
                                oracle_choices={"value": oracle_choice})
    chosen = options[int(response.answers["value"]["choice"]) - 1]
    return response, question, chosen["label"], chosen["index"]


async def _oracle_box(page: Page, choices: list[Candidate], expected: dict) -> tuple[str | None, str | None]:
    if expected.get("done"):
        return None, None
    for candidate in choices:
        if await matches_expected(page, candidate, expected):
            return str(candidate.n), None
    selector = expected.get("selector")
    if selector:
        if not await page.locator(selector).count():
            return None, f"Oracle selector {selector!r} is absent from the current DOM"
        return None, f"Oracle selector {selector!r} is not among visible candidates"
    target = expected.get("target")
    if target:
        return None, f"Oracle canvas target {target!r} is not among visible candidates"
    return None, "Oracle has no visible next action while expected.done is false"


async def _show(page: Page, marked, *, scale: float, kind: str, model: str) -> None:
    if not await page.evaluate("Boolean(window.blinkCUAOverlay)"):
        await page.add_script_tag(path=str(_OVERLAY))
    await page.evaluate(
        "({marks,scale,width,kind,model}) => window.blinkCUAOverlay.show(marks,scale,width,kind,model)",
        {"marks": marked.overlay_marks(scale), "scale": scale,
         "width": marked.size[0], "kind": kind, "model": model},
    )


async def _display(page: Page, *, number: str | None, action: str, probabilities: dict,
                   p_done: float, p_risky: float, ms: float, gated: bool,
                   beat_ms: int) -> float:
    start = time.perf_counter()
    await page.evaluate(
        "details => window.blinkCUAOverlay.decide(details)",
        {"number": number, "action": action, "probabilities": probabilities,
         "done": p_done, "risky": p_risky, "ms": ms, "gated": gated},
    )
    if beat_ms:
        await page.wait_for_timeout(beat_ms)
        if gated:
            await page.evaluate("(n) => window.blinkCUAOverlay.approve(n)", number)
            await page.wait_for_timeout(min(300, max(160, beat_ms // 2)))
    await page.evaluate("window.blinkCUAOverlay.remove()")
    return round((time.perf_counter() - start) * 1000, 2)


async def _act(page: Page, kind: str, candidate: Candidate,
               value: str | None, option_index: int | None) -> None:
    if candidate.target is not None:
        await page.mouse.click(*candidate.center)
    elif kind == "select":
        await page.locator(candidate.selector).select_option(index=option_index, timeout=5000)
    elif kind == "type":
        await page.locator(candidate.selector).fill(value, timeout=5000)
    else:
        await page.locator(candidate.selector).click(timeout=5000)


async def _contract(page: Page) -> dict:
    return await page.evaluate("""() => {
      const s = window.blinkScenario;
      if (!s || !s.task || !Array.isArray(s.values) ||
          !['check', 'expected', 'state'].every(k => typeof s[k] === 'function'))
        throw new Error('Page does not implement the blinkScenario contract');
      return {id:s.id, seed:s.seed, taskIndex:s.taskIndex, task:s.task, values:s.values};
    }""")


def _scores(rows: list[dict], kind_game: bool) -> dict:
    scored = [r for r in rows if "answers" in r]
    element = [r for r in scored if not r["expected"].get("done")]
    actionable = [r for r in scored if r["action"]["kind"] not in ("done", "none", "invalid_action")]
    done_rows = [] if kind_game else scored
    risk_rows = [] if kind_game else element
    ms = [r["wall_ms"] for r in scored]
    return {
        "element_correct": sum(r["element_correct"] is True for r in element),
        "element_total": len(element),
        "oracle_issues": sum(bool(r.get("oracle_issue")) for r in element),
        "done_tp": sum(r["done_predicted"] and r["expected"].get("done") for r in done_rows),
        "done_fp": sum(r["done_predicted"] and not r["expected"].get("done") for r in done_rows),
        "done_fn": sum(not r["done_predicted"] and r["expected"].get("done") for r in done_rows),
        "done_tn": sum(not r["done_predicted"] and not r["expected"].get("done") for r in done_rows),
        "risk_tp": sum(r["gate_fired"] and r["expected_risky"] for r in risk_rows),
        "risk_fp": sum(r["gate_fired"] and not r["expected_risky"] for r in risk_rows),
        "risk_fn": sum(not r["gate_fired"] and r["expected_risky"] for r in risk_rows),
        "risk_tn": sum(not r["gate_fired"] and not r["expected_risky"] for r in risk_rows),
        "gates": sum(r["gate_fired"] for r in risk_rows),
        "steps": len(actionable),
        "decision_steps": len(scored),
        "wall_ms": ms,
        "p50_wall_ms": _percentile(ms, 0.5),
        "p95_wall_ms": _percentile(ms, 0.95),
    }


def _game_score(check: dict) -> tuple[int, int]:
    if isinstance(check.get("waves_survived"), int) and isinstance(check.get("hits"), int):
        return check["waves_survived"], check["hits"]
    match = re.fullmatch(r"wave (\d+)/\d+, hits (\d+)", str(check.get("detail", "")))
    if match:
        return int(match[1]), int(match[2])
    raise ValueError(f"Game check() does not expose waves/hits: {check}")


async def run_episode(
    browser: Browser, *, base_url: str, scenario: dict, task: dict, seed: int,
    model: str, client: Client, output: Path, harness_sha: str,
    interrupts: int = 0, beat_ms: int = 600, record_video: bool = True,
    convert_video: bool = True, mode: str = "batched", game_realtime: bool = False,
) -> dict:
    """Run and persist one episode, including errors rather than replacing failed results."""
    if beat_ms < 0 or interrupts not in (0, 1):
        raise ValueError("beat_ms must be >= 0 and interrupts must be 0 or 1")
    viewport, scale = scenario["viewport"], scenario["scale"]
    kind_game = scenario["kind"] == "game"
    if game_realtime and (not kind_game or beat_ms):
        raise ValueError("Real-time game mode requires a game scenario and beat_ms=0")
    cadence = scenario.get("cadence", {}).get("slow_ms", scenario.get("cadence_ms", scenario.get("tick_ms", 0)))
    cadence = int(cadence) if kind_game else 0
    max_steps = int(task["optimal_steps"]) * (4 if kind_game and "cadence" in scenario else 2) + (
        8 if kind_game and "cadence" in scenario else 4
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "steps").mkdir(exist_ok=True)
    trace_file = output / "trace.jsonl"
    trace_file.write_text("", encoding="utf-8")
    params = {"seed": seed, "task": task["index"], "interrupts": interrupts, "theme": "light"}
    if kind_game and "cadence" in scenario:
        params["speed"] = "slow"
        if not game_realtime:
            params["paused"] = "1"
    url = (base_url.rstrip("/") + "/" + scenario["path"].lstrip("/") + "?" + urlencode(params))
    rows: list[dict] = []
    history: list[str] = []
    consecutive: list[tuple] = []
    check: dict = {"done": False, "success": False, "detail": "Episode did not start"}
    stop_reason = "step_limit"
    error = None
    context = None
    video = None
    game_lockstep = False
    try:
        context_options = {"viewport": viewport, "device_scale_factor": scale,
                           "color_scheme": "light", "reduced_motion": "reduce"}
        if record_video:
            context_options.update(record_video_dir=str(output),
                                   record_video_size={"width": round(viewport["width"] * scale),
                                                      "height": round(viewport["height"] * scale)})
        context = await browser.new_context(**context_options)

        async def local_only(route):
            if route.request.url.startswith(base_url.rstrip("/") + "/"):
                await route.continue_()
            else:
                await route.abort()

        await context.route("**/*", local_only)
        page = await context.new_page()
        video = page.video
        await page.goto(url, wait_until="load", timeout=15000)
        contract = await _contract(page)
        if (contract["id"] != scenario["id"] or contract["seed"] != seed or
                contract["taskIndex"] != task["index"]):
            raise ValueError(f"App identity differs from requested job: {contract}")
        if kind_game and not game_realtime:
            game_lockstep = bool(await page.evaluate(
                "typeof blinkScenario.pause === 'function' && typeof blinkScenario.advance === 'function'"
            ))
            if game_lockstep:
                await page.evaluate("blinkScenario.pause(true)")
        for step in range(1, max_steps + 1):
            tick_start = time.perf_counter()
            start = _now()
            await page.evaluate("document.fonts.ready")
            expected = await page.evaluate("window.blinkScenario.expected()")
            check_before = await page.evaluate("window.blinkScenario.check()")
            state_name = await page.evaluate("window.blinkScenario.state()")
            if kind_game and check_before.get("done"):
                check = check_before
                stop_reason = "game_finished"
                break
            choices = await candidates(page)
            if game_realtime:
                raw = await page.screenshot(type="png", scale="device")
                current = await page.evaluate("""() => ({
                  expected: blinkScenario.expected(), check: blinkScenario.check(), state: blinkScenario.state()
                })""")
                expected, check_before, state_name = current["expected"], current["check"], current["state"]
                if check_before.get("done"):
                    check = check_before
                    stop_reason = "game_finished"
                    break
                oracle_n, oracle_issue = await _oracle_box(page, choices, expected)
            else:
                oracle_n, oracle_issue = await _oracle_box(page, choices, expected)
                raw = await page.screenshot(type="png", scale="device")
            if isinstance(client, MockClient) and not expected.get("done") and not oracle_n:
                raise ValueError(f"{oracle_issue}: {expected}")
            marked = mark_screenshot(raw, choices, scale=scale, viewport=viewport)
            image_path = f"steps/{step:03}.png"
            (output / image_path).write_bytes(marked.png)
            shot = marked.shot(contract["task"])
            request_state = screens.state(shot)
            request_state["previous_actions"] = history.copy()
            if not choices:
                request_state["screenshot_note"] = "Use the attached screenshot; there are no actionable boxes."
            if kind_game:
                questions = {"lane": {"type": "choice",
                                      "instructions": "Which marked lane is safe for the player right now?",
                                      "criteria": {str(c.n): f"box {c.n}" for c in choices}}}
                if len(choices) != 3:
                    raise ValueError(f"Game must offer exactly three lane boxes, found {len(choices)}")
            else:
                questions = screens.questions(len(choices))
                if not choices:
                    questions.pop("element")
            if record_video:
                await _show(page, marked, scale=scale, kind=scenario["kind"], model=model)
            main = await client.ask(request_state, questions, expected=expected,
                                    oracle_choices={"lane" if kind_game else "element": oracle_n})
            answers = main.answers.copy()
            followup_ms = 0.0
            followup_usage: dict = {}
            p_done = main.answers["done"]["noul"] if not kind_game else 0.0
            p_risky = main.answers["risky"]["noul"] if not kind_game else 0.0
            done = not kind_game and p_done >= screens.DONE_AT
            chosen_n = (main.answers.get("lane" if kind_game else "element", {}).get("choice")
                        if not done else None)
            chosen = next((c for c in choices if str(c.n) == chosen_n), None)
            action_kind = "done" if done else _action_kind(chosen) if chosen else "none"
            invalid_action = None
            if action_kind == "type" and not contract["values"]:
                invalid_action = f"Box {chosen.n} is a text field but the task offers no values"
                action_kind = "invalid_action"
            option_index = None
            value = None
            request_questions = questions.copy()
            if action_kind in ("type", "select"):
                extra, question, value, option_index = await _followup(
                    page, chosen, request_state, expected, client, contract["values"]
                )
                answers.update(extra.answers)
                request_questions.update(question)
                followup_ms = extra.wall_ms
                followup_usage = extra.usage
            wall_ms = round(main.wall_ms + followup_ms, 2)
            gate = not done and chosen is not None and not invalid_action and p_risky >= screens.RISKY_AT
            label = ("Task complete" if done else f"No value for {chosen_n}" if invalid_action else
                     f"{action_kind.capitalize()} {chosen_n}"
                     if chosen else "No action available")
            display_ms = 0.0
            if record_video:
                probs = main.answers.get("lane" if kind_game else "element", {}).get("probabilities", {})
                display_ms = await _display(
                    page, number=chosen_n, action=label, probabilities=probs,
                    p_done=p_done, p_risky=p_risky, ms=wall_ms, gated=gate, beat_ms=beat_ms,
                )
            if invalid_action:
                stop_reason = "no_task_value"
            elif done:
                stop_reason = "model_done"
            elif not chosen:
                stop_reason = "no_candidates"
            else:
                await _act(page, action_kind, chosen, value, option_index)
                memory = _memory(action_kind, chosen, value)
                history.append(memory)
                identity = (action_kind, chosen.selector or chosen.target, value)
                consecutive.append(identity)
                consecutive = consecutive[-3:]
                if not kind_game and len(consecutive) == 3 and len(set(consecutive)) == 1:
                    stop_reason = "repeated_action"
            game_display_ms = 0.0
            game_advance_ms = 0
            if kind_game and chosen:
                if game_lockstep:
                    intervals = 4 if record_video and cadence else 1
                    game_start = time.perf_counter()
                    for index in range(intervals):
                        increment = cadence // intervals + (index < cadence % intervals)
                        await page.evaluate("ms => blinkScenario.advance(ms)", increment)
                        if record_video and increment:
                            await page.wait_for_timeout(increment)
                    game_display_ms = round((time.perf_counter() - game_start) * 1000, 2)
                    game_advance_ms = cadence
                else:
                    remaining = cadence - (time.perf_counter() - tick_start) * 1000
                    if remaining > 0:
                        await page.wait_for_timeout(remaining)
                        game_display_ms = round(remaining, 2)
            check = await page.evaluate("window.blinkScenario.check()")
            if kind_game:
                _game_score(check)
                if check.get("done"):
                    stop_reason = "game_finished"
            row = {
                "step": step, "t_start": start, "t_end": _now(), "url": page.url,
                "state": state_name, "candidates": marked.trace_candidates(choices),
                "marked_screenshot": image_path,
                "request_state": {k: v for k, v in request_state.items() if k != "screenshot"},
                "request_questions": request_questions,
                "answers": answers,
                "usage": {key: main.usage.get(key, 0) + followup_usage.get(key, 0)
                          for key in ("input_tokens", "visual_tokens")},
                "main_usage": main.usage, "followup_usage": followup_usage,
                "chosen_action": {"kind": action_kind, "box": chosen_n,
                                  "selector": chosen.selector if chosen else None,
                                  "target": chosen.target if chosen else None,
                                  "value": value,
                                  "memory": history[-1] if chosen and not invalid_action else None},
                "action": {"kind": action_kind, "box": chosen_n},
                "invalid_action": invalid_action,
                "main_ms": main.wall_ms, "followup_ms": followup_ms, "wall_ms": wall_ms,
                "per_question_ms": main.per_question_ms,
                "display_ms": display_ms, "game_display_ms": game_display_ms,
                "game_advance_ms": game_advance_ms, "expected": expected,
                "oracle_box": oracle_n,
                "oracle_issue": oracle_issue,
                "element_correct": (oracle_n is not None and chosen_n == oracle_n
                                    if not expected.get("done") else None),
                "done_predicted": done, "gate_fired": gate,
                "expected_risky": bool(expected.get("risky")), "check": check,
            }
            rows.append(row)
            with trace_file.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            if stop_reason != "step_limit":
                break
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        stop_reason = "error"
        (output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        if context is not None:
            await context.close()
        if video is not None:
            recorded = Path(await video.path())
            os.replace(recorded, output / ("video.capture.webm" if convert_video else "video.webm"))

    summary = {
        "scenario": scenario["id"], "kind": scenario["kind"], "task_index": task["index"],
        "seed": seed, "model": model, "server": client.server,
        "interrupts": interrupts, "mode": "game-realtime" if game_realtime else mode,
        "game_realtime": game_realtime, "beat_ms": beat_ms,
        "success": bool(check.get("success")) and error is None,
        "done": bool(check.get("done")), "detail": check.get("detail"),
        "stop_reason": stop_reason, "error": error,
        "optimal_steps": int(task["optimal_steps"]),
        "step_limit": max_steps,
        "harness_sha": harness_sha,
        "observation": "marked screenshot pixels, task, screenshot note and agent's own action history",
        "candidates_source": "canvas hit regions" if scenario["kind"] in ("canvas", "game") else "DOM",
        "oracle_exposed_to_model": False,
        "done_at": screens.DONE_AT, "risky_at": screens.RISKY_AT,
        "game_clock": ("real-time: unpaused during inference and display" if game_realtime else
                       "paused during inference; advanced one cadence per decision" if game_lockstep else None),
        "game_cadence_ms": cadence if kind_game else None,
        "waves_survived": _game_score(check)[0] if kind_game and error is None else None,
        "hits": _game_score(check)[1] if kind_game and error is None else None,
        **_scores(rows, scenario["kind"] == "game"),
    }
    if record_video and convert_video and (output / "video.capture.webm").exists():
        from .video import convert, normalize_capture
        try:
            await asyncio.to_thread(normalize_capture, output / "video.capture.webm",
                                    output / "video.webm", viewport=viewport, scale=scale)
            summary["video"] = await asyncio.to_thread(convert, output / "video.webm")
            summary["video"]["capture"] = "video.capture.webm"
        except (OSError, RuntimeError, ImportError) as exc:
            summary["video_error"] = f"{type(exc).__name__}: {exc}"
    temp = output / "summary.json.tmp"
    temp.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temp, output / "summary.json")
    return summary
