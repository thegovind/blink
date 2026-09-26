"""Hold one of the app's own round trips open on the wire, the way a slow link does.

Nothing in the app is asked to wait: the browser sends each request exactly when it
always would, and the delay is applied to the round trip itself. BLINK_SLOW_SYNC is
not involved and stays off, so these checks run against the build as it ships.

The handlers are found in the page's own config rather than by a hard-coded index:
each one is named by the script gradio runs in the browser before it, and by how many
values go in and out.

    from hold import wiring, hold, watch
"""

from __future__ import annotations

import asyncio

ROWS = ".blk-qrow:not(.blk-qrow .blk-qrow)"
FOLD = ".blk-qmore button.label-wrap"
OPTS = ".blk-qopts textarea"
SHOWN = """(sel) => Array.from(document.querySelectorAll(sel))
    .filter((e) => e.offsetParent !== null).map((e) => e.textContent.trim())"""
CARD_KEYS = f"""() => ({SHOWN})('.blk-q .blk-q-head h4')"""
CARD_BY = f"""() => ({SHOWN})('.blk-by')"""
# the answer column has a problem strip of its own; the one beside the rows is not it
ANSWER_PROBLEM = f"""() => ({SHOWN})('.blk-main .blk-problems')"""
EMPTY = f"""() => ({SHOWN})('.blk-empty p')"""
ROW_NAMES = """() => Array.from(document.querySelectorAll('.blk-qrow'))
    .filter((g) => !g.parentElement.closest('.blk-qrow') && g.offsetParent !== null)
    .map((g) => g.querySelector('input').value)"""
TROUBLE = """() => (document.querySelector('.blk-problems') || {}).textContent || ''"""
EDITOR = """() => (document.querySelector('.blk-acc .cm-content') || {}).innerText || ''"""
SELECTED = """() => Array.from(
    document.querySelectorAll('button[role="tab"][aria-selected="true"]')
  ).map((b) => b.textContent.trim())"""
PICKED = """() => Array.from(document.querySelectorAll('.blk-seg input'))
    .filter((i) => i.offsetParent !== null && i.checked).map((i) => i.value)"""


async def wiring(page) -> dict:
    """Name every handler these checks need, from the config the page was built with."""
    cfg = await page.evaluate("() => window.gradio_config")
    deps = cfg["dependencies"]
    bumps = {d["id"] for d in deps
             if "__blinkAsk" in (d.get("js") or "") and not d["backend_fn"]}
    named: dict[str, list[int]] = {}
    runs: list[tuple[int, int]] = []
    for d in deps:
        js, after = d.get("js") or "", d.get("trigger_after")
        event = (d.get("targets") or [[None, None]])[0][1]
        n_in, n_out = len(d["inputs"]), len(d["outputs"])
        if '"rows"' in js and after is None and n_out == 3:
            named.setdefault("from_rows", []).append(d["id"])
        elif '"json"' in js and after is None and event == "input":
            named.setdefault("from_json", []).append(d["id"])
        elif '"json"' in js and after is None and event == "click":
            named.setdefault("preset", []).append(d["id"])
        elif '"json"' in js and after is not None:
            named.setdefault("hand_off", []).append(d["id"])
        elif after in bumps and n_out > 1:
            named.setdefault("ask", []).append(d["id"])       # the ask itself
        elif after in bumps:
            named.setdefault("ask_model", []).append(d["id"])  # choosing a model in Ask
        elif after is not None and not js and n_out == 1 and n_in > 4:
            runs.append((n_in, d["id"]))
    # the decide chain is two steps over the same inputs; the longer one is the run
    if runs:
        widest = max(n for n, _ in runs)
        named["decide"] = [i for n, i in runs if n == widest]
        named["waiting"] = [i for n, i in runs if n != widest]
    return named


def watch(page, seconds: float = 0.0, slow: list | None = None):
    """Record every request the app sends, holding the named ones open.

    Returns the list the payloads land in; `slow` is a list of handler ids."""
    sent: list[dict] = []
    held = set(slow or ())

    async def route(request):
        req = request.request
        body = None
        if req.method == "POST":
            try:
                body = req.post_data_json
            except Exception:  # noqa: BLE001 - not every post is a json payload
                body = None
        if isinstance(body, dict) and isinstance(body.get("data"), list):
            sent.append(body)
            if seconds and body.get("fn_index") in held:
                await asyncio.sleep(seconds)
        await request.continue_()

    return sent, route


async def hold(page, ids, seconds: float):
    """Delay the given handlers by `seconds`, and collect what the page sends."""
    sent, route = watch(page, seconds, ids)
    await page.route("**/gradio_api/**", route)
    return sent


def missing(named: dict, *want: str) -> list[str]:
    """Which handlers this build does not stamp, so a check can say so and stop.

    A build that marks no revisions is the bug itself, not a broken check."""
    return [w for w in want if not named.get(w)]


def ran(sent: list[dict], decide_ids) -> list[list]:
    """The payloads Decide was actually given: state, model, surface, revision, ..."""
    want = set(decide_ids)
    return [b["data"] for b in sent if b.get("fn_index") in want]


async def open_tab(page, label: str) -> None:
    """Narrow viewports move the tab bar behind a "More tabs" menu."""
    strip = page.get_by_role("tab", name=label, exact=True)
    if await strip.count() and await strip.first.is_visible():
        await strip.first.click()
        return
    await page.locator('[aria-label="More tabs"]').first.click()
    await page.wait_for_timeout(400)
    await page.get_by_role("button", name=label, exact=True).last.click()


async def open_json(page) -> None:
    editor = page.locator(".blk-acc .cm-content")
    if await editor.count() == 0 or not await editor.first.is_visible():
        await page.locator(".blk-acc button.label-wrap").first.click()
        await page.wait_for_timeout(700)


async def write_json(page, text: str) -> None:
    editor = page.locator(".cm-content").first
    await editor.scroll_into_view_if_needed()
    await editor.click()
    await page.keyboard.press("ControlOrMeta+a")
    await page.keyboard.press("Backspace")
    await page.keyboard.insert_text(text)


async def open_fold(page, row) -> None:
    if not await row.locator(OPTS).first.is_visible():
        await row.locator(FOLD).first.click()
        await page.wait_for_timeout(450)


def collector(page, problems: list) -> None:
    page.on("pageerror", lambda e: problems.append(f"pageerror: {e}"))
    page.on("console", lambda m: problems.append(f"console.error: {m.text}")
            if m.type == "error" else None)


def report(problems: list, good: str) -> int:
    noisy = [p for p in problems if "favicon" not in str(p).lower()]
    if noisy:
        print(f"\n{len(noisy)} problem(s):")
        for p in noisy:
            print(" -", p)
        return 1
    print(f"\n{good}")
    return 0
