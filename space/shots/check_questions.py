"""Drive the playground form: rows edit, JSON edits, presets, and what Decide answers.

    BLINK_MOCK=1 BLINK_PORT=7898 uv run --with gradio==6.28.0 python app.py &
    BLINK_URL=http://127.0.0.1:7898/ uv run --with playwright python shots/check_questions.py
"""

import asyncio
import os
import sys

from playwright.async_api import async_playwright

URL = os.environ.get("BLINK_URL", "http://127.0.0.1:7898/").rstrip("/")
OUT = os.path.dirname(os.path.abspath(__file__))

CARD_KEYS = """() => Array.from(document.querySelectorAll('.blk-q .blk-q-head h4'))
    .map((h) => h.textContent.trim())"""
CARD_OPTS = """() => Array.from(document.querySelectorAll('.blk-q .blk-opt .k'))
    .map((k) => k.textContent.trim())"""
# a gradio Group is two nested divs with the same class; only the outer one is a row
ROW_NAMES = """() => Array.from(document.querySelectorAll('.blk-qrow'))
    .filter((g) => !g.parentElement.closest('.blk-qrow'))
    .map((g) => g.querySelector('input').value)"""
ROW_ASKS = """() => Array.from(document.querySelectorAll('.blk-qrow'))
    .filter((g) => !g.parentElement.closest('.blk-qrow'))
    .map((g) => g.querySelector('.blk-qask textarea').value)"""
ROWS = ".blk-qrow:not(.blk-qrow .blk-qrow)"
FOLD = ".blk-qmore button.label-wrap"
# the question is on screen with its name and type; the options box opens from the fold
NAME = 0
ASK = ".blk-qask textarea"
OPTS = ".blk-qopts textarea"


async def open_fold(page, row) -> None:
    """Options sit behind a per-question fold now; open it before typing."""
    if not await row.locator(OPTS).first.is_visible():
        await row.locator(FOLD).first.click()
        await page.wait_for_timeout(450)


async def type_into(page, locator, text: str) -> None:
    """Gradio listens for real key events; fill() sets the value without raising one."""
    await locator.click()
    await page.keyboard.press("ControlOrMeta+a")
    await page.keyboard.type(text, delay=12)
    await page.wait_for_timeout(500)


async def settled(page, timeout: int = 25000) -> None:
    """Wait out the run itself.

    The skeleton is the only honest signal: the card already on screen belongs to the
    previous request, so its presence proves nothing."""
    try:
        await page.wait_for_selector('[aria-busy="true"]', timeout=4000)
        await page.wait_for_selector('[aria-busy="true"]', state="detached",
                                     timeout=timeout)
    except Exception:  # noqa: BLE001 - a refused request never shows one
        pass
    await page.wait_for_selector(".blk-q .blk-q-head h4, .blk-problems, .blk-empty",
                                 timeout=timeout)
    await page.wait_for_timeout(700)


async def decide(page):
    await page.get_by_role("button", name="Decide", exact=True).first.click()
    await settled(page)


async def main() -> int:
    problems: list[str] = []
    ok = lambda name, good: print(f"  {name:52s} {'ok' if good else 'WRONG'}") or (
        None if good else problems.append(name)
    )

    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        page = await browser.new_page(viewport={"width": 1280, "height": 900}, device_scale_factor=2)
        page.on("pageerror", lambda e: problems.append(f"pageerror: {e}"))
        page.on("console", lambda m: problems.append(f"console.error: {m.text}")
                if m.type == "error" else None)
        await page.goto(URL + "/?tab=playground", wait_until="load")
        await page.wait_for_selector(".blk-qrow", timeout=30000)
        await page.wait_for_timeout(1800)

        # the questions are on screen without opening anything
        rows = await page.evaluate(ROW_NAMES)
        ok(f"questions visible on load: {rows}", rows == ["intent", "suspicious", "urgency"])
        asked = await page.evaluate(ROW_ASKS)
        ok(f"each row still says what it asks: {asked[:1]}", all(asked) and len(asked) == 3)
        await page.screenshot(path=os.path.join(OUT, "questions-desktop.png"), full_page=True)

        # (c) editing only the state leaves the questions on screen, before and after
        box = page.locator("textarea").first
        await box.fill("A short note about a delivery that never arrived.")
        await page.wait_for_timeout(500)
        before = await page.evaluate(ROW_NAMES)
        await decide(page)
        after = await page.evaluate(ROW_NAMES)
        keys = await page.evaluate(CARD_KEYS)
        ok("state edit keeps the questions visible", before == after == ["intent", "suspicious", "urgency"])
        ok(f"answers still name the questions: {keys}", keys == ["intent", "suspicious", "urgency"])

        # (a) edit a question in the rows: rename it and give it new options
        first = page.locator(ROWS).first
        await type_into(page, first.locator("input").nth(NAME), "topic")
        await type_into(page, first.locator(ASK).first, "What is this note about?")
        await open_fold(page, first)
        await type_into(page, first.locator(OPTS).first,
                        "delivery: A parcel or delivery\nbilling: Money")
        await page.wait_for_timeout(600)
        await decide(page)
        keys = await page.evaluate(CARD_KEYS)
        opts = await page.evaluate(CARD_OPTS)
        ok(f"row edit renames the answer: {keys[:1]}", keys[0] == "topic")
        ok(f"row edit changes the options: {opts[:2]}",
           opts[:2] == ["delivery", "billing"])
        await page.screenshot(path=os.path.join(OUT, "questions-edited.png"), full_page=True)

        # (b) the same through the JSON view
        await page.get_by_role("button", name="Edit as JSON").first.click()
        await page.wait_for_selector(".cm-content", state="visible", timeout=15000)
        await page.wait_for_timeout(900)
        editor = page.locator(".cm-content").first
        await editor.scroll_into_view_if_needed()
        await editor.click()
        await page.keyboard.press("ControlOrMeta+a")
        await page.keyboard.press("Backspace")
        # insert rather than type: the editor closes brackets as you go, and half a
        # request is a request the page must refuse
        await page.keyboard.insert_text(
            '{"mood": {"type": "noul", "instructions": "Is it angry?"}}')
        await page.keyboard.type(" ")
        await page.keyboard.press("Backspace")
        await page.wait_for_timeout(2000)
        doc = (await editor.inner_text()).strip()
        ok(f"the editor holds what we wrote: {doc[:26]!r}", doc.endswith("}}"))
        rows = await page.evaluate(ROW_NAMES)
        ok(f"json edit rewrites the rows: {rows}", rows == ["mood"])
        await decide(page)
        keys = await page.evaluate(CARD_KEYS)
        ok(f"json edit changes the answer: {keys}", keys == ["mood"])

        # (d) a preset puts everything back, visibly
        await page.get_by_role("button", name="Support ticket", exact=True).first.click()
        await settled(page)
        rows = await page.evaluate(ROW_NAMES)
        ok(f"preset refills the rows: {rows}",
           rows == ["queue", "urgency", "refund_eligible", "angry"])
        keys = await page.evaluate(CARD_KEYS)
        ok(f"preset answers its own questions: {keys}",
           keys == ["queue", "urgency", "refund_eligible", "angry"])

        # validation shows up inline rather than blowing up
        first = page.locator(ROWS).first
        await open_fold(page, first)
        await type_into(page, first.locator(OPTS).first, "only_one: just the one")
        await page.wait_for_timeout(1200)
        trouble = await page.evaluate(
            """() => (document.querySelector('.blk-problems') || {}).textContent || ''""")
        ok(f"inline problem shown: {trouble.strip()[:40]!r}", "two options" in trouble)
        await decide(page)
        still = await page.evaluate(
            """() => (document.querySelector('.blk-problems') || {}).textContent || ''""")
        ok("Decide repeats the problem instead of crashing", "two options" in still)
        await page.screenshot(path=os.path.join(OUT, "questions-invalid.png"), full_page=True)

        # blank starter
        await page.get_by_role("button", name="Blank", exact=True).first.click()
        await settled(page)
        rows = await page.evaluate(ROW_NAMES)
        ok(f"blank starter gives one question: {rows}", rows == ["label"])

        # narrow
        await page.set_viewport_size({"width": 390, "height": 844})
        await page.goto(URL + "/?tab=playground", wait_until="load")
        await page.wait_for_selector(".blk-qrow", timeout=30000)
        await page.wait_for_timeout(1800)
        rows = await page.evaluate(ROW_NAMES)
        ok(f"narrow keeps the questions: {rows}", len(rows) == 3)
        wide = await page.evaluate("() => document.documentElement.scrollWidth <= window.innerWidth")
        ok("narrow does not scroll sideways", wide)
        await page.screenshot(path=os.path.join(OUT, "questions-narrow.png"), full_page=True)

        await browser.close()

    noisy = [p for p in problems if "favicon" not in str(p).lower()]
    if noisy:
        print(f"\n{len(noisy)} problem(s):")
        for p in noisy:
            print(" -", p)
        return 1
    print("\nthe questions are visible, editable, and what the answers are about")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
