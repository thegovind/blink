"""A reply that arrives late must not undo the request that replaced it.

    BLINK_MOCK=1 BLINK_PORT=7921 uv run --with gradio==6.28.0 python app.py &
    BLINK_URL=http://127.0.0.1:7921/ uv run --with playwright python shots/check_ordering.py

The validation of a paste is held open on the wire while a preset is loaded on top of
it. When the old reply finally lands, the rows, the row count, the editor, the error
strip and the answers all have to be the preset, and nothing else. The State carries a
marker so the run afterwards proves which request was actually executed.
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
DELAY = float(os.environ.get("BLINK_HOLD") or 2)
TICKET = ["queue", "urgency", "refund_eligible", "angry"]
OBSOLETE = json.dumps({"obsolete": {"type": "noul", "instructions": "Old request?"}},
                      indent=2)
BROKEN = '{"queue": {"type": "choice",'


async def settle(page, extra: float = 2.5) -> None:
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
    gaps = hold.missing(named, "from_json", "decide")
    ok(f"{tag} every edit is stamped with a revision: {gaps}", not gaps)
    if gaps:
        await browser.close()
        return problems
    sent = await hold.hold(page, named["from_json"], DELAY)

    # (1) a paste still being validated, and a preset loaded straight over it
    await hold.open_json(page)
    await hold.write_json(page, OBSOLETE)
    await page.get_by_role("button", name="Support ticket", exact=True).first.click()
    await settle(page)

    rows = await page.evaluate(hold.ROW_NAMES)
    keys = await page.evaluate(hold.CARD_KEYS)
    editor = await page.evaluate(hold.EDITOR)
    trouble = await page.evaluate(hold.TROUBLE)
    ok(f"{tag} the late paste does not take the rows: {rows}", rows == TICKET)
    ok(f"{tag} the editor is the preset too: {sorted(json.loads(editor))[:2]}",
       sorted(json.loads(editor)) == sorted(TICKET))
    ok(f"{tag} the answers are the preset's: {keys}", keys == TICKET)
    ok(f"{tag} and no error is left over: {trouble.strip()[:30]!r}",
       trouble.strip() == "")

    # the run afterwards is a fresh one, and it is the four the page shows
    marker = f"blink-ordering-{size['width']}"
    await page.locator(".blk-state textarea").first.fill(marker)
    await page.wait_for_timeout(400)
    await page.get_by_role("button", name="Decide", exact=True).first.click()
    await settle(page)
    asked = hold.ran(sent, named["decide"])
    keys = await page.evaluate(hold.CARD_KEYS)
    rows = await page.evaluate(hold.ROW_NAMES)
    ok(f"{tag} Decide runs the request on screen: {keys}",
       keys == TICKET and rows == TICKET)
    ok(f"{tag} and it is a fresh run: {asked[-1][0][:24]!r}",
       bool(asked) and asked[-1][0] == marker)
    await page.screenshot(path=os.path.join(OUT, f"ordering-preset-{size['width']}.png"),
                          full_page=True)

    # (2) an older broken paste must not leave its warning on a preset that worked
    await hold.open_json(page)
    await hold.write_json(page, BROKEN)
    await page.get_by_role("button", name="Invoice email", exact=True).first.click()
    await settle(page)
    rows = await page.evaluate(hold.ROW_NAMES)
    trouble = await page.evaluate(hold.TROUBLE)
    keys = await page.evaluate(hold.CARD_KEYS)
    ok(f"{tag} the preset loads over a broken paste: {rows[:2]}", len(rows) == 3)
    ok(f"{tag} no stale JSON warning is left: {trouble.strip()[:30]!r}",
       "JSON" not in trouble)
    ok(f"{tag} and it answered its own rows: {keys}", keys == rows)
    await page.screenshot(path=os.path.join(OUT, f"ordering-broken-{size['width']}.png"),
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
    return hold.report(problems, "a late reply never undoes the request that replaced it")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
