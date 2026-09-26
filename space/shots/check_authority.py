"""Which surface Decide runs is the one whose content changed last — nothing else.

    BLINK_MOCK=1 BLINK_PORT=7921 uv run --with gradio==6.28.0 python app.py &
    BLINK_URL=http://127.0.0.1:7921/ uv run --with playwright python shots/check_authority.py

Opening Edit as JSON is not an edit, and clicking a row is not one either. With the row
sync held open on the wire the two disagree, and the click alone used to decide which
one ran. The State carries a marker no earlier run can have, so the answer on screen
says whether a fresh decision happened and what it was given.
"""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from playwright.async_api import async_playwright  # noqa: E402

import hold  # noqa: E402

URL = os.environ.get("BLINK_URL", "http://127.0.0.1:7921/").rstrip("/")
OUT = os.path.dirname(os.path.abspath(__file__))
SIZES = ({"width": 1456, "height": 900}, {"width": 390, "height": 844})
DELAY = 2.0
OLD = ["intent", "suspicious", "urgency"]
PASTE = json.dumps({"mood": {"type": "noul", "instructions": "Is it angry?"}}, indent=2)


async def settle(page, extra: float = 2.0) -> None:
    await page.wait_for_timeout(int((DELAY * 2 + extra) * 1000))


async def run(pw, size, ok) -> list:
    problems: list = []
    browser = await pw.chromium.launch()
    page = await browser.new_page(viewport=size, device_scale_factor=2)
    hold.collector(page, problems)
    tag = f"{size['width']}px"
    await page.goto(URL + "/?tab=playground", wait_until="load")
    await page.wait_for_selector(".blk-qrow", timeout=30000)
    await page.wait_for_timeout(2200)

    named = await hold.wiring(page)
    gaps = hold.missing(named, "from_rows", "from_json", "decide")
    ok(f"{tag} every edit is stamped with a revision: {gaps}", not gaps)
    if gaps:
        await browser.close()
        return problems
    sent = await hold.hold(page, named["from_rows"] + named["from_json"], DELAY)

    # (1) a row made invalid, the editor opened on top of it, then Decide
    marker = f"blink-authority-{size['width']}"
    box = page.locator(".blk-state textarea").first
    await box.fill(marker)
    await page.wait_for_timeout(400)
    first = page.locator(hold.ROWS).first
    await hold.open_fold(page, first)
    await first.locator(hold.OPTS).first.click()
    await page.keyboard.press("ControlOrMeta+a")
    await page.keyboard.type("only: one option", delay=5)
    # the sync is still on the wire; opening the editor must not take the request over
    await page.locator(".blk-acc button.label-wrap").first.click()
    await page.get_by_role("button", name="Decide", exact=True).first.click()
    await settle(page)

    keys = await page.evaluate(hold.CARD_KEYS)
    refused = " ".join(await page.evaluate(hold.ANSWER_PROBLEM))
    asked = hold.ran(sent, named["decide"])
    ok(f"{tag} the old questions are not answered: {keys}",
       all(k not in keys for k in OLD))
    ok(f"{tag} the invalid row is what Decide reads: {refused.strip()[:34]!r}",
       keys == [] and "two options" in refused)
    ok(f"{tag} Decide was handed the rows, not the editor: "
       f"{[a[2] for a in asked]}", bool(asked) and asked[-1][2] == "rows")
    ok(f"{tag} and the state it ran is the new one: {asked[-1][0]!r}",
       asked[-1][0] == marker)
    await page.screenshot(path=os.path.join(OUT, f"authority-rows-{size['width']}.png"),
                          full_page=True)

    # (2) a valid paste, then a click on an old row: the paste is still the request
    await page.get_by_role("button", name="Support ticket", exact=True).first.click()
    await settle(page)
    await hold.open_json(page)
    await hold.write_json(page, PASTE)
    old_rows = await page.evaluate(hold.ROW_NAMES)
    first = page.locator(hold.ROWS).first
    await first.locator("input").nth(0).click()  # a click, and nothing typed
    await page.get_by_role("button", name="Decide", exact=True).first.click()
    await settle(page)

    keys = await page.evaluate(hold.CARD_KEYS)
    asked = hold.ran(sent, named["decide"])
    ok(f"{tag} a click on a row is not an edit: {old_rows[:1]} -> {keys}",
       keys == ["mood"])
    ok(f"{tag} and the editor stayed the surface: {asked[-1][2]!r}",
       asked[-1][2] == "json")
    await page.screenshot(path=os.path.join(OUT, f"authority-json-{size['width']}.png"),
                          full_page=True)

    wide = await page.evaluate(
        "() => document.documentElement.scrollWidth <= window.innerWidth")
    ok(f"{tag} no sideways scroll", wide)
    await browser.close()
    return problems


async def main() -> int:
    problems: list = []

    def ok(name, good):
        print(f"  {name:76s} {'ok' if good else 'WRONG'}")
        if not good:
            problems.append(name)

    async with async_playwright() as pw:
        for size in SIZES:
            problems += await run(pw, size, ok)
    return hold.report(problems, "only a changed value moves the request")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
