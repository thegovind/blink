"""Navigation has to work where the tab strip does not fit.

    BLINK_MOCK=1 BLINK_PORT=7911 uv run --with gradio==6.28.0 python app.py &
    BLINK_URL=http://127.0.0.1:7911/ uv run --with playwright python shots/check_mobile.py

At 390 px gradio folds the overflow tabs behind a menu, so nothing may depend on a
rendered tab button being there to click.
"""

import asyncio
import os
import sys

from playwright.async_api import async_playwright

URL = os.environ.get("BLINK_URL", "http://127.0.0.1:7911/").rstrip("/")
OUT = os.path.dirname(os.path.abspath(__file__))
PHONE = {"width": 390, "height": 844}
DESK = {"width": 1456, "height": 900}
# the home cards, in their order on the page (ui.HOME_LINKS)
CARDS = ["use-cases", "ask", "playground", "results", "how-it-works"]

SELECTED = """() => Array.from(
    document.querySelectorAll('button[role="tab"][aria-selected="true"]')
  ).map((b) => b.textContent.trim())"""
ROW_NAMES = """() => Array.from(document.querySelectorAll('.blk-qrow'))
    .filter((g) => !g.parentElement.closest('.blk-qrow') && g.offsetParent !== null)
    .map((g) => g.querySelector('input').value)"""
SHOWN_CARDS = f"""() => Array.from(document.querySelectorAll('.blk-q .blk-q-head h4'))
    .filter((e) => e.offsetParent !== null).map((e) => e.textContent.trim())"""
STALE = """() => Array.from(document.querySelectorAll('.blk-empty'))
    .filter((e) => e.offsetParent !== null).length"""


async def settle(page, timeout: int = 25000) -> None:
    """Wait out the run: the card on screen belongs to the previous request."""
    try:
        await page.wait_for_selector('[aria-busy="true"]', timeout=4000)
        await page.wait_for_selector('[aria-busy="true"]', state="detached",
                                     timeout=timeout)
    except Exception:  # noqa: BLE001 - a refused request never shows one
        pass
    # visible only: tabs opened earlier keep their own answers, hidden, before this one
    await page.wait_for_selector(
        ".blk-q .blk-q-head h4:visible, .blk-problems:visible, .blk-empty:visible", timeout=timeout)
    await page.wait_for_timeout(700)


async def open_tab(page, label: str) -> None:
    """Narrow viewports move the tab bar behind a "More tabs" menu."""
    strip = page.get_by_role("tab", name=label, exact=True)
    if await strip.count() and await strip.first.is_visible():
        await strip.first.click()
        return
    await page.locator('[aria-label="More tabs"]').first.click()
    await page.wait_for_timeout(400)
    await page.get_by_role("button", name=label, exact=True).last.click()


def seg(page):
    return page.locator(".blk-seg label").filter(visible=True)


async def switch_model(page) -> str:
    """Click the model that is not already chosen, or nothing happens at all."""
    state = await page.evaluate(
        """() => Array.from(document.querySelectorAll('.blk-seg input'))
             .filter((i) => i.offsetParent !== null).map((i) => i.checked)""")
    want = state.index(False) if False in state else 0
    await seg(page).nth(want).click()
    await page.wait_for_timeout(3000)
    return await seg(page).nth(want).inner_text()


async def run(pw, size, ok) -> None:
    browser = await pw.chromium.launch()
    page = await browser.new_page(viewport=size, device_scale_factor=2)
    tag = f"{size['width']}px"
    reloads = []
    page.on("load", lambda _: reloads.append(1))
    await page.goto(URL + "/", wait_until="load")
    await page.wait_for_selector(".blk-top h1", timeout=30000)
    await page.wait_for_timeout(2000)
    loads_at_start = len(reloads)

    # a home card moves tab without fetching the document again
    for slug, label in (("results", "Results"), ("use-cases", "Use cases"),
                        ("how-it-works", "How it works"), ("ask", "Ask"),
                        ("playground", "Playground")):
        await open_tab(page, "Home")
        await page.wait_for_timeout(500)
        card = page.locator("button.blk-tile-hit").nth(CARDS.index(slug))
        await card.scroll_into_view_if_needed()
        await card.click()
        await page.wait_for_timeout(1100)
        picked = await page.evaluate(SELECTED)
        search = await page.evaluate("() => window.location.search")
        ok(f"{tag} card -> {label}: {picked[:1]}", picked[:1] == [label])
        ok(f"{tag} card syncs the url: {search}", f"tab={slug}" in search)
    ok(f"{tag} no card reloaded the page", len(reloads) == loads_at_start)

    # the overflow menu selects exactly like the strip, url and all
    for slug, label in (("ask", "Ask"), ("results", "Results"),
                        ("how-it-works", "How it works")):
        await open_tab(page, "Playground")
        await page.wait_for_timeout(500)
        await open_tab(page, label)
        await page.wait_for_timeout(1000)
        picked = await page.evaluate(SELECTED)
        search = await page.evaluate("() => window.location.search")
        ok(f"{tag} menu -> {label}: {picked[:1]}", picked[:1] == [label])
        ok(f"{tag} menu syncs the url: {search}", f"tab={slug}" in search)

    # the hand-off moves tab and takes the request and its answer with it
    await open_tab(page, "Ask")
    await page.wait_for_selector("#ask-input input", timeout=20000)
    await page.wait_for_timeout(900)
    box = page.locator("#ask-input input").first
    await box.click()
    await page.keyboard.type("How many r in strawberry", delay=8)
    await page.wait_for_timeout(300)
    await page.locator("#ask-run").first.click()
    await page.wait_for_selector("#ask-open", timeout=25000)
    await page.wait_for_timeout(1600)
    await page.locator("#ask-open").first.click()
    await page.wait_for_timeout(2600)
    picked = await page.evaluate(SELECTED)
    rows = await page.evaluate(ROW_NAMES)
    cards = await page.evaluate(SHOWN_CARDS)
    state = await page.locator(".blk-state textarea").first.input_value()
    search = await page.evaluate("() => window.location.search")
    ok(f"{tag} hand-off lands on Playground: {picked[:1]}", picked[:1] == ["Playground"])
    ok(f"{tag} hand-off syncs the url: {search}", "tab=playground" in search)
    ok(f"{tag} hand-off brings the request: {rows} {state!r}",
       rows == ["answer"] and state == "strawberry")
    ok(f"{tag} and no stale answer beside it: {cards}", cards == ["answer"])
    ok(f"{tag} nothing reloaded", len(reloads) == loads_at_start)
    await page.screenshot(path=os.path.join(OUT, f"mobile-handoff-{size['width']}.png"),
                          full_page=True)

    # leaving the tab and coming back must not bring the old rows back with it
    await open_tab(page, "Results")
    await page.wait_for_timeout(800)
    await open_tab(page, "Playground")
    await page.wait_for_timeout(1100)
    ok(f"{tag} the rows survive a tab round trip: {await page.evaluate(ROW_NAMES)}",
       await page.evaluate(ROW_NAMES) == ["answer"])

    # and neither must the way in through a home card
    await open_tab(page, "Home")
    await page.wait_for_timeout(700)
    play_card = page.locator("button.blk-tile-hit").nth(CARDS.index("playground"))
    await play_card.scroll_into_view_if_needed()
    await play_card.click()
    await page.wait_for_timeout(1400)
    rows = await page.evaluate(ROW_NAMES)
    ok(f"{tag} the rows survive a card round trip: {rows}", rows == ["answer"])
    # an extra row left visible would take an edit Decide then ignored
    await page.get_by_role("button", name="Decide", exact=True).first.click()
    await settle(page)
    ok(f"{tag} and Decide answers those rows: {await page.evaluate(SHOWN_CARDS)}",
       await page.evaluate(SHOWN_CARDS) == ["answer"])

    # the blank starter has to be a starter, not a locked view
    await page.get_by_role("button", name="Blank", exact=True).first.click()
    await settle(page)
    rows = await page.evaluate(ROW_NAMES)
    locked = await page.evaluate(
        """() => (document.querySelector('.blk-locked') || {}).textContent || ''""")
    editable = await page.evaluate(
        """() => Array.from(document.querySelectorAll('.blk-qrow input'))
             .filter(i => i.offsetParent !== null).some(i => !i.disabled)""")
    ok(f"{tag} blank gives one editable row: {rows}", rows == ["label"])
    ok(f"{tag} blank is not locked: {locked.strip()[:28]!r}", locked.strip() == "")
    ok(f"{tag} blank rows accept typing", editable)
    await page.get_by_role("button", name="Add question", exact=True).first.click()
    await page.wait_for_timeout(900)
    ok(f"{tag} and add question works after it: {await page.evaluate(ROW_NAMES)}",
       len(await page.evaluate(ROW_NAMES)) == 2)
    await page.screenshot(path=os.path.join(OUT, f"blank-{size['width']}.png"),
                          full_page=True)

    # every held-out value is readable without panning
    await open_tab(page, "Home")
    await page.wait_for_timeout(1100)
    seen = await page.evaluate("""() => {
        const w = window.innerWidth;
        const out = [];
        for (const el of document.querySelectorAll('.blk-half text, .blk-half td, .blk-half th')) {
            if (el.offsetParent === null && el.tagName !== 'text') continue;
            const b = el.getBoundingClientRect();
            if (b.width === 0) continue;
            out.push([el.textContent.trim(), Math.round(b.right) <= w + 1]);
        }
        return out;
    }""")
    for want in ("78.3", "73.5", "68.5", "63.44", "76.6", "80.0", "0.096"):
        hit = [v for t, v in seen if want in t]
        ok(f"{tag} home shows {want} in view", bool(hit) and all(hit))
    await page.screenshot(path=os.path.join(OUT, f"mobile-home-{size['width']}.png"),
                          full_page=True)

    # switching model stops every other answer claiming to be this one's
    if await page.locator(".blk-seg input").count() >= 2:

        await open_tab(page, "Playground")
        await page.wait_for_timeout(900)
        before = await page.evaluate(SHOWN_CARDS)
        ok(f"{tag} playground has an answer: {before[:1]}", bool(before))
        await open_tab(page, "Ask")
        await page.wait_for_timeout(900)
        await switch_model(page)
        by = await page.evaluate(
            """() => Array.from(document.querySelectorAll('.blk-by'))
                 .filter(e => e.offsetParent !== null).map(e => e.textContent.trim())""")
        ok(f"{tag} the ask answer names the new model: {by}",
           bool(by) and all("mimo" in b for b in by))
        await open_tab(page, "Playground")
        await page.wait_for_timeout(900)
        stale = await page.evaluate(STALE)
        cards = await page.evaluate(SHOWN_CARDS)
        ok(f"{tag} the playground answer is cleared: {stale} {cards}",
           stale == 1 and cards == [])
        await page.screenshot(path=os.path.join(OUT, f"mobile-stale-{size['width']}.png"),
                              full_page=True)

        # an answer another tab invalidated must not come back through the hand-off
        await open_tab(page, "Ask")
        await page.wait_for_timeout(900)
        box = page.locator("#ask-input input").first
        await box.click()
        await page.keyboard.press("ControlOrMeta+a")
        await page.keyboard.type("How many r in strawberry", delay=8)
        await page.locator("#ask-run").first.click()
        await page.wait_for_selector("#ask-open", timeout=25000)
        await page.wait_for_timeout(1800)
        asked_by = await page.evaluate(
            """() => Array.from(document.querySelectorAll('.blk-by'))
                 .filter(e => e.offsetParent !== null).map(e => e.textContent.trim())""")
        await open_tab(page, "Use cases")
        await page.wait_for_timeout(900)
        await switch_model(page)
        await open_tab(page, "Ask")
        await page.wait_for_timeout(900)
        ok(f"{tag} the ask answer went stale: {await page.evaluate(STALE)}",
           await page.evaluate(STALE) == 1)
        await page.locator("#ask-open").first.click()
        await page.wait_for_timeout(2600)
        left = await page.evaluate(
            """() => Array.from(document.querySelectorAll('.blk-by'))
                 .filter(e => e.offsetParent !== null).map(e => e.textContent.trim())""")
        ok(f"{tag} and the hand-off does not bring it back: {asked_by} -> {left}",
           left == [] and await page.evaluate(STALE) == 1)
        ok(f"{tag} the request still arrives: {await page.evaluate(ROW_NAMES)}",
           await page.evaluate(ROW_NAMES) == ["answer"])
        await page.screenshot(path=os.path.join(OUT, f"handoff-stale-{size['width']}.png"),
                              full_page=True)
    # the claim about generated text belongs to blink's pass, not the drafter's
    async def claim():
        return await page.evaluate(
            """() => (document.querySelector('.blk-meta') || {}).textContent || ''""")

    await open_tab(page, "Ask")
    await page.wait_for_timeout(900)
    ok(f"{tag} ask does not claim zero generated: {(await claim())[-34:]!r}",
       "0 generated tokens" not in await claim())
    await open_tab(page, "Playground")
    await page.wait_for_timeout(900)
    ok(f"{tag} the playground still does", "0 generated tokens" in await claim())
    await page.goto(URL + "/?tab=ask", wait_until="load")
    await page.wait_for_selector("#ask-input input", timeout=25000)
    await page.wait_for_timeout(2000)
    ok(f"{tag} a deep link to ask sets the claim too",
       "0 generated tokens" not in await claim())

    await page.close()
    await browser.close()


async def main() -> int:
    problems: list[str] = []

    def ok(name, good):
        print(f"  {name:70s} {'ok' if good else 'WRONG'}")
        if not good:
            problems.append(name)

    async with async_playwright() as pw:
        for size in (PHONE, DESK):
            await run(pw, size, ok)

    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems:
            print(" -", p)
        return 1
    print("\nnavigation works where the strip does not fit")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
