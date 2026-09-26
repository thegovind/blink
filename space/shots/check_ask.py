"""Drive the Ask box: a draft becomes a visible, editable request and then an answer.

    BLINK_MOCK=1 BLINK_PORT=7899 uv run --with gradio==6.28.0 python app.py &
    BLINK_URL=http://127.0.0.1:7899/ uv run --with playwright python shots/check_ask.py
"""

import asyncio
import os
import sys

from playwright.async_api import async_playwright

URL = os.environ.get("BLINK_URL", "http://127.0.0.1:7899/").rstrip("/")
OUT = os.path.dirname(os.path.abspath(__file__))

ROWS = ".blk-qrow:not(.blk-qrow .blk-qrow)"
FOLD = ".blk-qmore button.label-wrap"
OPTS = ".blk-qopts textarea"
# stable hooks, so the same check runs against the deployed Space
ASK_INPUT = "#ask-input input"
ASK_RUN = "#ask-run"
ASK_OPEN = "#ask-open"
PROVENANCE = "#ask-provenance"
# both tabs keep their answers in the page, so only read the one on screen
SHOWN = """(sel) => Array.from(document.querySelectorAll(sel))
    .filter((e) => e.offsetParent !== null)
    .map((e) => e.textContent.trim())"""
DRAFT_NAMES = f"""() => ({SHOWN})('.blk-draft-q b')"""
DRAFT_KINDS = f"""() => ({SHOWN})('.blk-draft-q .blk-kind')"""
DRAFT_ASKS = f"""() => ({SHOWN})('.blk-draft-q .blk-q-head p')"""
DRAFT_OPTS = f"""() => ({SHOWN})('.blk-draft-q li')"""
ROW_NAMES = """() => Array.from(document.querySelectorAll('.blk-qrow'))
    .filter((g) => !g.parentElement.closest('.blk-qrow') && g.offsetParent !== null)
    .map((g) => g.querySelector('input').value)"""
CARD_KEYS = f"""() => ({SHOWN})('.blk-q .blk-q-head h4')"""
CARD_ASKED = f"""() => ({SHOWN})('.blk-q .blk-q-head p')"""
CARD_OPTS = f"""() => ({SHOWN})('.blk-q .blk-opt .k')"""
NOTE = """() => (document.querySelector('.blk-ask-note') || {}).textContent || ''"""


async def type_into(page, locator, text: str) -> None:
    await locator.click()
    await page.keyboard.press("ControlOrMeta+a")
    await page.keyboard.type(text, delay=10)
    await page.wait_for_timeout(400)


async def ask(page, text: str, settle: int = 3000) -> None:
    await type_into(page, page.locator(ASK_INPUT).first, text)
    await page.locator(ASK_RUN).first.click()
    await page.wait_for_timeout(settle)


async def open_ask(page, url: str) -> None:
    """Free-form asks have their own tab now; the playground stays clean."""
    await page.goto(url, wait_until="load")
    await page.wait_for_selector(ASK_INPUT, timeout=30000)
    await page.wait_for_timeout(1800)


async def main() -> int:
    problems: list[str] = []

    def ok(name, good):
        print(f"  {name:58s} {'ok' if good else 'WRONG'}")
        if not good:
            problems.append(name)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        page = await browser.new_page(viewport={"width": 1280, "height": 900}, device_scale_factor=2)
        page.on("pageerror", lambda e: problems.append(f"pageerror: {e}"))
        page.on("console", lambda m: problems.append(f"console.error: {m.text}")
                if m.type == "error" else None)
        await open_ask(page, URL + "/?tab=ask")
        ok("the playground has no ask box",
           await page.evaluate("""() => {
               const box = document.querySelector('#ask-input');
               const play = document.querySelector('.blk-state');
               return !(box && play && box.closest('.tabitem') === play.closest('.tabitem'));
           }"""))

        # an ask becomes a visible request and an answer
        await ask(page, "How many r in strawberry")
        names = await page.evaluate(DRAFT_NAMES)
        kinds = await page.evaluate(DRAFT_KINDS)
        asked = await page.evaluate(DRAFT_ASKS)
        dopts = await page.evaluate(DRAFT_OPTS)
        opts = await page.evaluate(CARD_OPTS)
        keys = await page.evaluate(CARD_KEYS)
        cards = await page.evaluate(CARD_ASKED)
        ok(f"the drafted question is shown: {names} {kinds}",
           names == ["answer"] and kinds == ["choice"])
        ok(f"it shows what was asked: {asked[:1]}",
           asked and asked[0].startswith("How many r in strawberry"))
        ok(f"it shows the options: {dopts}", [d.split(":")[0] for d in dopts] == ["1", "2", "3", "4"])
        ok(f"card answers the drafted question: {keys}", keys == ["answer"])
        ok(f"options are 1-4: {opts}", opts == ["1", "2", "3", "4"])
        ok(f"card repeats the wording: {cards[:1]}",
           cards[0].startswith("How many r in strawberry"))
        drafted = await page.evaluate(
            f"""() => (document.querySelector('{PROVENANCE}') || {{}}).textContent || ''""")
        ok(f"provenance shown: {drafted!r}", "drafted by" in drafted and "generated tokens" in drafted)
        zero = await page.evaluate(
            """() => Array.from(document.querySelectorAll('.blk-stats span'))
                 .map(s => s.textContent.trim()).join(' | ')""")
        ok(f"blink still reports none of its own: {zero[:60]}", "0 generated" in zero)
        ok("the draft can be opened next door",
           await page.evaluate("() => !!document.querySelector('#ask-open')"))
        await page.screenshot(path=os.path.join(OUT, "ask-desktop.png"), full_page=True)

        # something blink cannot decide says so, and leaves the draft alone
        await ask(page, "write me a poem", settle=2500)
        note = await page.evaluate(NOTE)
        ok(f"inline error: {note.strip()[:44]!r}", "few possible answers" in note)
        ok("the draft is left alone after the error",
           await page.evaluate(DRAFT_NAMES) == ["answer"])
        await page.screenshot(path=os.path.join(OUT, "ask-error.png"), full_page=True)

        # the draft moves next door, where it can be edited and re-run
        await open_ask(page, URL + "/?tab=ask")
        await ask(page, "How many r in strawberry")
        await page.locator(ASK_OPEN).first.click()
        await page.wait_for_timeout(2000)
        picked = await page.evaluate(
            """() => Array.from(document.querySelectorAll('button[role="tab"][aria-selected="true"]'))
                 .map((b) => b.textContent.trim())""")
        rows = await page.evaluate(ROW_NAMES)
        state = await page.locator(".blk-state textarea").first.input_value()
        ok(f"the hand-off switches tab: {picked[:1]}", picked[:1] == ["Playground"])
        ok(f"one drafted row lands there: {rows}", rows == ["answer"])
        ok(f"state holds the subject: {state!r}", state == "strawberry")

        first = page.locator(ROWS).first
        if not await first.locator(OPTS).first.is_visible():
            await first.locator(FOLD).first.click()
            await page.wait_for_timeout(450)
        await type_into(page, first.locator(OPTS).first, "two: two\nthree: three")
        await page.get_by_role("button", name="Decide", exact=True).first.click()
        await page.wait_for_timeout(2200)
        opts = await page.evaluate(CARD_OPTS)
        ok(f"edit to a drafted option is honoured: {opts}", opts == ["two", "three"])
        await page.screenshot(path=os.path.join(OUT, "ask-edited.png"), full_page=True)

        # the deep link opens the ask tab, fills the box and waits
        await open_ask(page, URL + "/?ask=Is%20this%20email%20phishing%3F")
        filled = await page.locator(ASK_INPUT).first.input_value()
        picked = await page.evaluate(
            """() => Array.from(document.querySelectorAll('button[role="tab"][aria-selected="true"]'))
                 .map((b) => b.textContent.trim())""")
        ok(f"?ask= pre-fills: {filled!r}", filled == "Is this email phishing?")
        ok(f"?ask= opens the ask tab: {picked[:1]}", picked[:1] == ["Ask"])
        ok("?ask= does not run", await page.evaluate(DRAFT_NAMES) == [])

        # narrow
        await page.set_viewport_size({"width": 390, "height": 844})
        await open_ask(page, URL + "/?tab=ask")
        await ask(page, "How many r in strawberry", settle=3500)
        ok("narrow drafts too", await page.evaluate(DRAFT_NAMES) == ["answer"])
        ok("narrow does not scroll sideways",
           await page.evaluate("() => document.documentElement.scrollWidth <= window.innerWidth"))
        await page.screenshot(path=os.path.join(OUT, "ask-narrow.png"), full_page=True)

        await browser.close()

    noisy = [p for p in problems if "favicon" not in str(p).lower()]
    if noisy:
        print(f"\n{len(noisy)} problem(s):")
        for p in noisy:
            print(" -", p)
        return 1
    print("\nan ask becomes a request you can see, open next door, edit and re-run")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
