"""One model for the whole page, and every panel says what to do in its own words.

    BLINK_MOCK=1 BLINK_PORT=7921 BLINK_MODELS=thegovind/blink-4b,thegovind/blink-mimo-9b \
        uv run --with gradio==6.28.0 python app.py &
    BLINK_URL=http://127.0.0.1:7921/ uv run --with playwright python shots/check_selection.py

Choosing a model on one tab has to reach every other control, not just the first of
them, and the answer the next request produces has to be that model's. Ask's own empty
column has to name an action Ask has.
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
MAST = """() => (document.querySelector('.blk-meta .key') || {}).textContent || ''"""
OTHER = """() => {
  const seg = Array.from(document.querySelectorAll('.blk-seg input'))
      .filter((i) => i.offsetParent !== null);
  const off = seg.find((i) => !i.checked);
  return off ? off.value : '';
}"""


def short(model: str) -> str:
    return model.rsplit("/", 1)[-1]


async def ask(page, text: str) -> None:
    box = page.locator("#ask-input input").first
    await box.click()
    await page.keyboard.press("ControlOrMeta+a")
    await page.keyboard.type(text, delay=8)
    await page.wait_for_timeout(300)
    await page.locator("#ask-run").first.click()


async def pick_other(page) -> str:
    """Click the model that is not already chosen on the tab in view."""
    want = await page.evaluate(OTHER)
    if not want:
        return ""
    seg = page.locator(".blk-seg label").filter(visible=True)
    values = await page.evaluate(
        """() => Array.from(document.querySelectorAll('.blk-seg input'))
             .filter((i) => i.offsetParent !== null).map((i) => i.value)""")
    await seg.nth(values.index(want)).click()
    await page.wait_for_timeout(3200)
    return want


async def run(pw, size, ok) -> list:
    problems: list = []
    browser = await pw.chromium.launch()
    page = await browser.new_page(viewport=size, device_scale_factor=2)
    hold.collector(page, problems)
    tag = f"{size['width']}px"
    await page.goto(URL + "/?tab=use-cases", wait_until="load")
    await page.wait_for_selector(".blk-seg input", timeout=30000)
    await page.wait_for_timeout(2200)

    # S1: chosen on Use cases, in force everywhere
    want = await pick_other(page)
    ok(f"{tag} there are two models to choose from: {short(want)}", bool(want))
    mast = await page.evaluate(MAST)
    ok(f"{tag} the masthead follows the choice: {mast}", mast == want)

    await hold.open_tab(page, "Ask")
    await page.wait_for_selector("#ask-input input", timeout=20000)
    await page.wait_for_timeout(1200)
    picked = await page.evaluate(hold.PICKED)
    ok(f"{tag} Ask's own control moved too: {[short(p) for p in picked]}",
       picked == [want])

    await ask(page, "How many r in strawberry")
    await page.wait_for_selector("#ask-open", timeout=25000)
    await page.wait_for_timeout(2200)
    by = await page.evaluate(hold.CARD_BY)
    ok(f"{tag} and Ask runs that model: {by}",
       bool(by) and all(b == short(want) for b in by))
    await page.screenshot(path=os.path.join(OUT, f"selection-ask-{size['width']}.png"),
                          full_page=True)

    # S3: a model chosen on another tab leaves Ask an action Ask actually has
    await hold.open_tab(page, "Use cases")
    await page.wait_for_timeout(1200)
    back = await pick_other(page)
    ok(f"{tag} the choice can be changed back: {short(back)}", bool(back))
    await hold.open_tab(page, "Ask")
    await page.wait_for_timeout(1500)
    empty = await page.evaluate(hold.EMPTY)
    ok(f"{tag} Ask says what to do next: {empty}", empty == ["Ask again."])
    ok(f"{tag} and never names Decide: {empty}",
       all("Decide" not in e for e in empty))
    ok(f"{tag} with no answer left under it: {await page.evaluate(hold.CARD_BY)}",
       await page.evaluate(hold.CARD_BY) == [])
    await page.screenshot(path=os.path.join(OUT, f"selection-stale-{size['width']}.png"),
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
    return hold.report(problems, "one choice, every control, and each panel's own words")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
