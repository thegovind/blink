"""An ask answers for the model it was sent with, or it does not answer at all.

    BLINK_MOCK=1 BLINK_PORT=7921 BLINK_MODELS=thegovind/blink-4b,thegovind/blink-mimo-9b \
        uv run --with gradio==6.28.0 python app.py &
    BLINK_URL=http://127.0.0.1:7921/ uv run --with playwright python shots/check_ask_model.py

The ask itself is held open on the wire while a different model is chosen inside Ask.
When it finally lands, its answer belongs to a model nobody has selected any more, so
neither the panel nor the hand-off next door may take it.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from playwright.async_api import async_playwright  # noqa: E402

import hold  # noqa: E402

URL = os.environ.get("BLINK_URL", "http://127.0.0.1:7921/").rstrip("/")
OUT = os.path.dirname(os.path.abspath(__file__))
SIZES = ({"width": 1456, "height": 900}, {"width": 390, "height": 844})
DELAY = float(os.environ.get("BLINK_HOLD_ASK") or 5)
OTHER = """() => {
  const seg = Array.from(document.querySelectorAll('.blk-seg input'))
      .filter((i) => i.offsetParent !== null);
  const off = seg.find((i) => !i.checked);
  return off ? off.value : '';
}"""


def short(model: str) -> str:
    return model.rsplit("/", 1)[-1]


async def type_ask(page, text: str) -> None:
    box = page.locator("#ask-input input").first
    await box.click()
    await page.keyboard.press("ControlOrMeta+a")
    await page.keyboard.type(text, delay=8)
    await page.wait_for_timeout(250)
    await page.locator("#ask-run").first.click()


async def pick_other(page) -> str:
    want = await page.evaluate(OTHER)
    if not want:
        return ""
    values = await page.evaluate(
        """() => Array.from(document.querySelectorAll('.blk-seg input'))
             .filter((i) => i.offsetParent !== null).map((i) => i.value)""")
    await page.locator(".blk-seg label").filter(visible=True).nth(
        values.index(want)).click()
    return want


async def run(pw, size, ok) -> list:
    problems: list = []
    browser = await pw.chromium.launch()
    page = await browser.new_page(viewport=size, device_scale_factor=2)
    hold.collector(page, problems)
    tag = f"{size['width']}px"
    await page.goto(URL + "/?tab=ask", wait_until="load")
    await page.wait_for_selector("#ask-input input", timeout=30000)
    await page.wait_for_timeout(2200)

    named = await hold.wiring(page)
    gaps = hold.missing(named, "ask")
    ok(f"{tag} the ask panel counts its own revisions: {gaps}", not gaps)
    if gaps:
        await browser.close()
        return problems
    first = (await page.evaluate(hold.PICKED) or [""])[0]
    await hold.hold(page, named["ask"], DELAY)

    # the ask goes out as the first model; the second is chosen while it is in flight
    await type_ask(page, "How many r in strawberry")
    await page.wait_for_timeout(900)
    want = await pick_other(page)
    ok(f"{tag} a second model can be chosen mid-ask: {short(want)}", bool(want))
    await page.wait_for_timeout(int((DELAY + 8) * 1000))

    by = await page.evaluate(hold.CARD_BY)
    empty = await page.evaluate(hold.EMPTY)
    opened = await page.locator("#ask-open").first.is_visible()
    ok(f"{tag} no answer from the model that was replaced: {by}",
       all(b != short(first) for b in by))
    ok(f"{tag} the panel stays empty rather than lying: {empty}",
       by == [] and empty == ["Ask again."])
    await page.screenshot(path=os.path.join(OUT, f"ask-model-held-{size['width']}.png"),
                          full_page=True)

    # nothing may be carried next door either, whether the button is there or not
    if opened:
        await page.locator("#ask-open").first.click()
        await page.wait_for_timeout(3000)
        carried = await page.evaluate(hold.CARD_BY)
        ok(f"{tag} the hand-off carries no mismatched answer: {carried}",
           all(b != short(first) for b in carried))
        await hold.open_tab(page, "Ask")
        await page.wait_for_timeout(1200)
    else:
        ok(f"{tag} there is no draft to hand over: {opened}", True)

    # and with the new model chosen, a fresh ask answers as that model, all the way over
    await page.unroute("**/gradio_api/**")
    await type_ask(page, "How many r in strawberry")
    await page.wait_for_selector("#ask-open", timeout=30000)
    await page.wait_for_timeout(2600)
    by = await page.evaluate(hold.CARD_BY)
    ok(f"{tag} a fresh ask answers as the chosen model: {by}",
       bool(by) and all(b == short(want) for b in by))
    await page.locator("#ask-open").first.click()
    await page.wait_for_timeout(3200)
    carried = await page.evaluate(hold.CARD_BY)
    rows = await page.evaluate(hold.ROW_NAMES)
    ok(f"{tag} and the hand-off carries that one: {carried} {rows}",
       carried == [short(want)] and rows == ["answer"])
    await page.screenshot(path=os.path.join(OUT, f"ask-model-fresh-{size['width']}.png"),
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
    return hold.report(problems, "an answer belongs to the model it was asked of")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
