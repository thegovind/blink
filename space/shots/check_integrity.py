"""What Decide runs has to be what the page shows, at every width.

    BLINK_MOCK=1 BLINK_PORT=7911 uv run --with gradio==6.28.0 python app.py &
    BLINK_URL=http://127.0.0.1:7911/ uv run --with playwright python shots/check_integrity.py

Every check here stands for a bug a reviewer found in a staged build.
"""

import asyncio
import json
import os
import sys

from playwright.async_api import async_playwright

URL = os.environ.get("BLINK_URL", "http://127.0.0.1:7911/").rstrip("/")
OUT = os.path.dirname(os.path.abspath(__file__))

ROWS = ".blk-qrow:not(.blk-qrow .blk-qrow)"
FOLD = ".blk-qmore button.label-wrap"
SHOWN = """(sel) => Array.from(document.querySelectorAll(sel))
    .filter((e) => e.offsetParent !== null)
    .map((e) => e.textContent.trim())"""
ROW_NAMES = """() => Array.from(document.querySelectorAll('.blk-qrow'))
    .filter((g) => !g.parentElement.closest('.blk-qrow') && g.offsetParent !== null)
    .map((g) => g.querySelector('input').value)"""
CARD_KEYS = f"""() => ({SHOWN})('.blk-q .blk-q-head h4')"""
CARD_OPTS = f"""() => ({SHOWN})('.blk-q .blk-opt .k')"""
TROUBLE = """() => (document.querySelector('.blk-problems') || {}).textContent || ''"""

NINE = json.dumps({f"q{i}": {"type": "noul", "instructions": f"ask {i}?"}
                   for i in range(9)}, indent=2)
NEWLINE = json.dumps({"a": {"type": "choice", "instructions": "Which one?", "criteria": {
    "keep": "one line\nand another", "drop": "plain"}}}, indent=2)
RANKING = json.dumps({"a": {"type": "ranking", "criteria": {"x": "1", "y": "2"}}}, indent=2)
COLLIDE = json.dumps({"q": {"type": "noul", "instructions": "first?"},
                      " q ": {"type": "noul", "instructions": "second?"}}, indent=2)


async def open_json(page):
    acc = page.locator(".blk-acc button.label-wrap").first
    if await page.locator(".blk-acc .cm-content").count() == 0 or not await page.locator(
            ".blk-acc .cm-content").first.is_visible():
        await acc.click()
        await page.wait_for_timeout(700)


async def write_json(page, text: str) -> None:
    editor = page.locator(".cm-content").first
    await editor.scroll_into_view_if_needed()
    await editor.click()
    await page.keyboard.press("ControlOrMeta+a")
    await page.keyboard.press("Backspace")
    await page.keyboard.insert_text(text)
    await page.keyboard.type(" ")
    await page.keyboard.press("Backspace")
    await page.wait_for_timeout(1800)


async def decide(page):
    await page.get_by_role("button", name="Decide", exact=True).first.click()
    await page.wait_for_timeout(2200)


async def main() -> int:
    problems: list[str] = []

    def ok(name, good):
        print(f"  {name:62s} {'ok' if good else 'WRONG'}")
        if not good:
            problems.append(name)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        page = await browser.new_page(viewport={"width": 1280, "height": 900},
                                      device_scale_factor=2)
        page.on("pageerror", lambda e: problems.append(f"pageerror: {e}"))
        page.on("console", lambda m: problems.append(f"console.error: {m.text}")
                if m.type == "error" else None)
        await page.goto(URL + "/?tab=playground", wait_until="load")
        await page.wait_for_selector(".blk-qrow", timeout=30000)
        await page.wait_for_timeout(1800)
        await open_json(page)

        # nine questions are nine questions, not the eight the rows can hold
        await write_json(page, NINE)
        rows = await page.evaluate(ROW_NAMES)
        locked = await page.evaluate(
            """() => (document.querySelector('.blk-locked') || {}).textContent || ''""")
        ok(f"nine questions show eight rows: {len(rows)}", len(rows) == 8)
        ok(f"and say so: {locked.strip()[:40]!r}", "8" in locked and "9" in locked)
        editable = await page.evaluate(
            f"""() => Array.from(document.querySelectorAll('{ROWS} input, {ROWS} textarea'))
                 .filter(i => i.offsetParent !== null).every(i => i.disabled)""")
        ok("the rows cannot quietly rewrite it", editable)
        await decide(page)
        keys = await page.evaluate(CARD_KEYS)
        ok(f"all nine are answered: {len(keys)}", len(keys) == 9)
        await page.screenshot(path=os.path.join(OUT, "integrity-nine.png"), full_page=True)

        # a newline inside a description is part of that description
        await write_json(page, NEWLINE)
        await decide(page)
        opts = await page.evaluate(CARD_OPTS)
        ok(f"a newline stays one option: {opts}", opts == ["keep", "drop"])

        # a type blink does not have is refused, not turned into one it does
        await write_json(page, RANKING)
        trouble = await page.evaluate(TROUBLE)
        ok(f"an unknown type is named: {trouble.strip()[:34]!r}", "type" in trouble)
        await decide(page)
        after = await page.evaluate(TROUBLE)
        keys = await page.evaluate(CARD_KEYS)
        ok("Decide repeats it instead of running", "type" in after)
        ok(f"and answers nothing: {keys}", keys == [])
        await page.screenshot(path=os.path.join(OUT, "integrity-refused.png"),
                              full_page=True)

        # two names that read alike are a mistake, not a silent replacement
        await write_json(page, COLLIDE)
        trouble = await page.evaluate(TROUBLE)
        ok(f"a name collision is named: {trouble.strip()[:32]!r}",
           "q" in trouble and "two questions" in trouble.lower())
        await decide(page)
        ok(f"and nothing runs: {await page.evaluate(CARD_KEYS)}",
           await page.evaluate(CARD_KEYS) == [])

        # after bad JSON, Decide must not fall back to whatever ran last
        await page.get_by_role("button", name="Support ticket", exact=True).first.click()
        await page.wait_for_selector(".blk-q .blk-q-head h4", timeout=25000)
        await page.wait_for_timeout(1200)
        keys = await page.evaluate(CARD_KEYS)
        ok(f"a preset answers again: {keys[:1]}", keys[:1] == ["queue"])
        await open_json(page)
        await write_json(page, '{"queue": {"type": "choice",')
        trouble = await page.evaluate(TROUBLE)
        ok(f"broken json is named: {trouble.strip()[:24]!r}", "JSON" in trouble)
        rows_now = await page.evaluate(ROW_NAMES)
        ok(f"the rows are left alone: {rows_now[:1]}", rows_now[:1] == ["queue"])
        await decide(page)
        keys = await page.evaluate(CARD_KEYS)
        still = await page.evaluate(TROUBLE)
        ok(f"Decide runs nothing: {keys}", keys == [])
        ok("and says why", "JSON" in still)

        # one click, one row
        await page.get_by_role("button", name="Invoice email", exact=True).first.click()
        await page.wait_for_selector(".blk-q .blk-q-head h4", timeout=25000)
        await page.wait_for_timeout(1200)
        async def add_one(want: int) -> int:
            """One click, then wait for it to land: a second click cannot overtake it."""
            await page.get_by_role("button", name="Add question", exact=True).first.click()
            try:
                await page.wait_for_function(
                    """(n) => Array.from(document.querySelectorAll('.blk-qrow'))
                         .filter((g) => !g.parentElement.closest('.blk-qrow')
                                        && g.offsetParent !== null).length === n""",
                    arg=want, timeout=15000)
            except Exception:  # noqa: BLE001 - reported by the caller
                pass
            await page.wait_for_timeout(300)
            return len(await page.evaluate(ROW_NAMES))

        before = await page.evaluate(ROW_NAMES)
        got = await add_one(len(before) + 1)
        ok(f"add question adds one row: {len(before)} -> {got}", got == len(before) + 1)
        got = await add_one(len(before) + 2)
        ok(f"and one more: {got}", got == len(before) + 2)
        await page.screenshot(path=os.path.join(OUT, "integrity-add.png"), full_page=True)

        await browser.close()

    noisy = [p for p in problems if "favicon" not in str(p).lower()]
    if noisy:
        print(f"\n{len(noisy)} problem(s):")
        for p in noisy:
            print(" -", p)
        return 1
    print("\nwhat runs is what the page shows")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
