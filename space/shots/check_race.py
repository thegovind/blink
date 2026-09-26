"""Click Decide while validation is still in flight.

    BLINK_MOCK=1 BLINK_SLOW_SYNC=2 BLINK_PORT=7923 uv run --with gradio==6.28.0 python app.py &
    BLINK_URL=http://127.0.0.1:7923/ uv run --with playwright python shots/check_race.py

BLINK_SLOW_SYNC holds every validation round trip open for that many seconds, so a
click lands in the window a fast machine hides. Nothing the visitor can see may be
answered by a request they replaced.
"""

import asyncio
import json
import os
import sys

from playwright.async_api import async_playwright

URL = os.environ.get("BLINK_URL", "http://127.0.0.1:7923/").rstrip("/")
OUT = os.path.dirname(os.path.abspath(__file__))
SLOW = float(os.environ.get("BLINK_SLOW_SYNC") or 2)

SHOWN = """(sel) => Array.from(document.querySelectorAll(sel))
    .filter((e) => e.offsetParent !== null).map((e) => e.textContent.trim())"""
CARD_KEYS = f"""() => ({SHOWN})('.blk-q .blk-q-head h4')"""
CARD_OPTS = f"""() => ({SHOWN})('.blk-q .blk-opt .k')"""
TROUBLE = """() => (document.querySelector('.blk-problems') || {}).textContent || ''"""
ROWS = ".blk-qrow:not(.blk-qrow .blk-qrow)"
FOLD = ".blk-qmore button.label-wrap"
OPTS = ".blk-qopts textarea"

GOOD = json.dumps({"mood": {"type": "noul", "instructions": "Is it angry?"}}, indent=2)


async def write_json(page, text: str) -> None:
    editor = page.locator(".cm-content").first
    await editor.scroll_into_view_if_needed()
    await editor.click()
    await page.keyboard.press("ControlOrMeta+a")
    await page.keyboard.press("Backspace")
    await page.keyboard.insert_text(text)


async def settle(page, extra: float = 1.0) -> None:
    await page.wait_for_timeout(int((SLOW * 2 + extra) * 1000))


async def main() -> int:
    problems: list[str] = []

    def ok(name, good):
        print(f"  {name:64s} {'ok' if good else 'WRONG'}")
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
        await page.wait_for_timeout(2000)
        before = await page.evaluate(CARD_KEYS)
        ok(f"the page starts with an answer: {before}", len(before) == 3)

        await page.locator(".blk-acc button.label-wrap").first.click()
        await page.wait_for_timeout(700)

        # bad JSON, then Decide before validation can land
        await write_json(page, '{"mood": {"type":')
        await page.get_by_role("button", name="Decide", exact=True).first.click()
        await settle(page)
        keys = await page.evaluate(CARD_KEYS)
        trouble = await page.evaluate(TROUBLE)
        ok(f"a racing Decide answers nothing: {keys}", keys == [])
        ok(f"and says why: {trouble.strip()[:22]!r}", "JSON" in trouble)
        await page.screenshot(path=os.path.join(OUT, "race-json.png"), full_page=True)

        # a valid paste, raced the same way, must run the paste and not the old rows
        await write_json(page, GOOD)
        await page.get_by_role("button", name="Decide", exact=True).first.click()
        await settle(page)
        keys = await page.evaluate(CARD_KEYS)
        ok(f"a racing valid paste runs itself: {keys}", keys == ["mood"])

        # a row edit raced by Decide runs the edit, not the editor it has not reached
        first = page.locator(ROWS).first
        if not await first.locator(OPTS).first.is_visible():
            await first.locator(FOLD).first.click()
            await page.wait_for_timeout(400)
        await first.locator("input").nth(0).click()
        await page.keyboard.press("ControlOrMeta+a")
        await page.keyboard.type("temper", delay=5)
        await page.get_by_role("button", name="Decide", exact=True).first.click()
        await settle(page)
        keys = await page.evaluate(CARD_KEYS)
        ok(f"a racing row edit runs the edit: {keys}", keys == ["temper"])

        # a row made invalid and raced by Decide answers nothing
        await first.locator("input").nth(1).click()
        await page.wait_for_timeout(300)
        await page.get_by_role("option", name="choice", exact=True).first.click()
        await page.wait_for_timeout(300)
        await first.locator(OPTS).first.click()
        await page.keyboard.press("ControlOrMeta+a")
        await page.keyboard.type("only_one: just the one", delay=5)
        await page.get_by_role("button", name="Decide", exact=True).first.click()
        await settle(page)
        keys = await page.evaluate(CARD_KEYS)
        trouble = await page.evaluate(TROUBLE)
        ok(f"a racing bad row answers nothing: {keys}", keys == [])
        ok(f"and says why: {trouble.strip()[:34]!r}", "two options" in trouble)
        await page.screenshot(path=os.path.join(OUT, "race-rows.png"), full_page=True)

        await browser.close()

    noisy = [p for p in problems if "favicon" not in str(p).lower()]
    if noisy:
        print(f"\n{len(noisy)} problem(s):")
        for p in noisy:
            print(" -", p)
        return 1
    print("\nnothing outruns validation")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
