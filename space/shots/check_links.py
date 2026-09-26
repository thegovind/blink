"""The project's links: GitHub and Docs beside the name on every tab, the API docs in the API tab,
and the source and the docs beside its steps and in How it works. Every one opens in a new tab, fits a phone, reads
the same when the system is dark, and resolves. How it works also links each model's card, in ui.MODEL_CARDS order;
each opens in a new tab and resolves.

    BLINK_MOCK=1 BLINK_PORT=7911 uv run --with gradio==6.28.0 python app.py &
    BLINK_URL=http://127.0.0.1:7911/ uv run --with playwright python shots/check_links.py

Screenshots go to BLINK_SHOTS (default: this folder). Exits non-zero on any problem, a console error included. The
link targets are fetched without credentials, to check that each one answers 200.
"""

import asyncio
import os
import sys
import urllib.request

from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import live  # noqa: E402  (live-Space allowances: shots/live.py)

URL = os.environ.get("BLINK_URL", "http://127.0.0.1:7911/").rstrip("/")
OUT = os.environ.get("BLINK_SHOTS") or HERE
DESK = {"width": 1456, "height": 900}
PHONE = {"width": 390, "height": 844}
TABS = (("home", "Home"), ("playground", "Playground"), ("ask", "Ask"), ("use-cases", "Use cases"),
        ("results", "Results"), ("how-it-works", "How it works"), ("api", "API"))
PROJECT = [["GitHub", "https://github.com/thegovind/blink"], ["Docs", "https://thegovind.github.io/blink/"]]
WIRE = [["API docs", "https://thegovind.github.io/blink/api/"]]
# How it works' "Full details": each model's card on the Hub, as ui.MODEL_CARDS lists them
CARDS = [[f"{m} card", f"https://huggingface.co/thegovind/{m}"] for m in ("blink-4b", "blink-27b", "blink-mimo-9b")]

# every link in a row: its label, where it goes, how it opens, and whether it can be seen
LINKS = """(sel) => Array.from(document.querySelectorAll(sel)).map((a) => {
  const r = a.getBoundingClientRect();
  return {text: a.textContent.trim(), href: a.getAttribute('href'), target: a.getAttribute('target'),
          rel: a.getAttribute('rel'), shown: a.offsetParent !== null && r.width > 0,
          box: [Math.round(r.left), Math.round(r.top), Math.round(r.right), Math.round(r.bottom)],
          inTab: !!a.closest('[role="tabpanel"]')};
})"""
# the name and the header links: on one line, apart, inside the window
HEADER = """() => {
  const word = document.querySelector('.blk-top .blk-word').getBoundingClientRect();
  const row = document.querySelector('.blk-toplinks').getBoundingClientRect();
  return {word: [Math.round(word.left), Math.round(word.top), Math.round(word.right), Math.round(word.bottom)],
          links: [Math.round(row.left), Math.round(row.top), Math.round(row.right), Math.round(row.bottom)],
          window: document.documentElement.clientWidth, page: document.documentElement.scrollWidth};
}"""
PAINT = """() => Array.from(document.querySelectorAll('.blk-toplinks a, .blk-links a'))
  .filter((a) => a.offsetParent !== null)
  .map((a) => {
    const s = getComputedStyle(a);
    let bg = 'none';
    for (let n = a; n; n = n.parentElement) {
      const c = getComputedStyle(n).backgroundColor;
      if (c && c !== 'rgba(0, 0, 0, 0)' && c !== 'transparent') { bg = c; break; }
    }
    return [a.textContent.trim(), s.color, bg, s.borderTopColor, s.textDecorationLine, s.fontSize, s.fontWeight];
  })"""
# the header names the model it was drawn for
SHOWS_MODEL = "(m) => ((document.querySelector('.blk-meta .key') || {}).textContent || '').includes(m)"
OPEN_RUN = """() => {
  const d = Array.from(document.querySelectorAll('details.blk-d'))
    .find((x) => x.offsetParent !== null && x.querySelector('.blk-links'));
  if (d) d.open = true;
  return !!d;
}"""


def pairs(found):
    return [[x["text"], x["href"]] for x in found]


def unsafe(found):
    return [x["text"] for x in found if x["target"] != "_blank" or "noopener" not in (x["rel"] or "").split()]


async def fresh(browser, size, scheme="light"):
    ctx = await browser.new_context(viewport=size, color_scheme=scheme, reduced_motion="reduce",
                                    device_scale_factor=2)
    page = await ctx.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
    page.on("console", lambda m: errors.append(f"console.error: {m.text}") if live.app_error(m) else None)
    return ctx, page, errors


async def land(page, link: str) -> None:
    await page.goto(URL + link, wait_until="domcontentloaded", timeout=180_000)
    await page.wait_for_selector(".blk-top h1", timeout=60_000)
    await page.wait_for_selector(".blk-toplinks a", timeout=20_000)
    await page.wait_for_timeout(1200)


async def open_tab(page, label: str) -> None:
    """A narrow window moves the tab strip behind a "More tabs" menu."""
    strip = page.get_by_role("tab", name=label, exact=True)
    if await strip.count() and await strip.first.is_visible():
        await strip.first.click()
        return
    await page.locator('[aria-label="More tabs"]').first.click()
    await page.wait_for_timeout(400)
    await page.get_by_role("button", name=label, exact=True).last.click()


async def clip_shot(page, selector: str, path: str, pad: int = 24, above: int = 0) -> None:
    """A picture of one region: the element, a margin round it and `above` px of what precedes it."""
    box = await page.evaluate("""(s) => {
      const r = document.querySelector(s).getBoundingClientRect();
      return {y: r.top + window.scrollY, height: r.height, width: document.documentElement.clientWidth};
    }""", selector)
    top = max(0, box["y"] - pad - above)
    await page.screenshot(path=path, clip={"x": 0, "y": top, "width": box["width"],
                                           "height": box["y"] + box["height"] + pad - top}, full_page=True)


async def main() -> int:
    problems: list[str] = []
    os.makedirs(OUT, exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch()

        # beside the name on every tab, opened by link; the header is never part of a tab
        for size, name in ((DESK, "desk"), (PHONE, "phone")):
            for slug, _ in TABS:
                ctx, page, errors = await fresh(browser, size)
                await land(page, f"/?tab={slug}")
                found = await page.evaluate(LINKS, ".blk-toplinks a")
                head = await page.evaluate(HEADER)
                ok = (pairs(found) == PROJECT and not unsafe(found) and all(x["shown"] for x in found)
                      and not any(x["inTab"] for x in found))
                # on the first screen, on the name's line, clear of it, and inside the window
                word, row = head["word"], head["links"]
                placed = (row[3] <= size["height"] and row[1] < word[3] and row[3] > word[1]
                          and row[0] >= word[2] and row[2] <= head["window"] and head["page"] <= head["window"])
                print(f"  {name:5} ?tab={slug:13} {pairs(found)} word {word} links {row} "
                      f"{'ok' if ok and placed else 'WRONG'}")
                if not ok:
                    problems.append(f"{name} {slug}: header links {found}")
                if not placed:
                    problems.append(f"{name} {slug}: header links not beside the name in the window: {head}")
                problems += [f"{name} {slug}: {e}" for e in errors]
                await ctx.close()

        # still there after moving between tabs and changing the model, which redraw the header
        for size, name in ((DESK, "desk"), (PHONE, "phone")):
            ctx, page, errors = await fresh(browser, size)
            await land(page, "/")
            for slug, label in TABS[1:] + TABS[:1]:
                await open_tab(page, label)
                await page.wait_for_timeout(700)
                found = pairs(await page.evaluate(LINKS, ".blk-toplinks a"))
                # a phone's menu holds the tab it opened, so the address says where it went
                search = await page.evaluate("() => window.location.search")
                if found != PROJECT or f"tab={slug}" not in search:
                    problems.append(f"{name} click {label}: address {search!r}, header links {found}")
            await open_tab(page, "Playground")
            await page.wait_for_timeout(700)
            seg = page.locator(".blk-seg label").filter(has_text="blink-mimo-9b")
            if await seg.count():
                await seg.first.click()
                # the live Space redraws the header about 1.6 s after the switch; give it up to 10 s
                try:
                    await page.wait_for_function(SHOWS_MODEL, arg="blink-mimo-9b", timeout=10_000)
                    late = ""
                except PlaywrightTimeoutError:
                    late = ", not redrawn within 10 s"
                key = await page.evaluate("() => (document.querySelector('.blk-meta .key') || {}).textContent")
                found = pairs(await page.evaluate(LINKS, ".blk-toplinks a"))
                print(f"  {name:5} after tab clicks and a model switch ({key}): {found}{late}")
                if found != PROJECT or "blink-mimo-9b" not in (key or "") or late:
                    problems.append(f"{name} model switch: header {key!r}, links {found}{late}")
            else:
                print(f"  {name:5} after tab clicks: {found} (one model: no switch)")
            problems += [f"{name} clicks: {e}" for e in errors]
            await ctx.close()

        # the API tab: the full reference by the lede, the source and the docs beside the steps
        # How it works: each model's card, and the source and the docs where it downloads and runs the model
        for size, name in ((DESK, "desk"), (PHONE, "phone")):
            ctx, page, errors = await fresh(browser, size)
            await land(page, "/?tab=api")
            lede = await page.evaluate(LINKS, ".blk-note + .blk-links a")
            steps = await page.evaluate(LINKS, ".blk-stepshead .blk-links a")
            fit = await page.evaluate("() => document.documentElement.scrollWidth <= window.innerWidth")
            order = await page.evaluate("""() => {
              const at = (s) => { const e = document.querySelector(s); return e ? e.getBoundingClientRect().top : -1; };
              return [at('.blk-h2'), at('.blk-note + .blk-links'), at('.blk-apifig'), at('.blk-stepshead'),
                      at('ol.blk-steps')];
            }""")
            ok = (pairs(lede) == WIRE and pairs(steps) == PROJECT and not unsafe(lede + steps)
                  and all(x["shown"] for x in lede + steps) and fit and order == sorted(order) and min(order) >= 0)
            print(f"  {name:5} API lede {pairs(lede)} steps {pairs(steps)} fits={fit} {'ok' if ok else 'WRONG'}")
            if not ok:
                problems.append(f"{name} API: lede {lede}, steps {steps}, fits {fit}, order {order}")
            await ctx.close()

            ctx, page, more = await fresh(browser, size)
            await land(page, "/?tab=how-it-works")
            cards = await page.evaluate(LINKS, "p.blk-note a")
            listed = pairs(cards) == CARDS and not unsafe(cards) and all(x["shown"] for x in cards)
            print(f"  {name:5} How it works cards {pairs(cards)} {'ok' if listed else 'WRONG'}")
            if not listed:
                problems.append(f"{name} How it works: model cards {cards}")
            closed = await page.evaluate(LINKS, "details.blk-d .blk-links a")
            opened = await page.evaluate(OPEN_RUN)
            await page.wait_for_timeout(300)
            how = await page.evaluate(LINKS, "details.blk-d[open] .blk-links a")
            summary = await page.evaluate("""() => {
              const d = document.querySelector('details.blk-d[open]');
              return d ? d.querySelector('summary').textContent.trim() : '';
            }""")
            fit = await page.evaluate("() => document.documentElement.scrollWidth <= window.innerWidth")
            ok = (opened and pairs(how) == PROJECT and not unsafe(how) and all(x["shown"] for x in how)
                  and summary == "Running it yourself" and pairs(closed) == PROJECT and fit)
            print(f"  {name:5} How it works '{summary}' {pairs(how)} fits={fit} {'ok' if ok else 'WRONG'}")
            if not ok:
                problems.append(f"{name} How it works: fold {summary!r}, links {how}, fits {fit}")
            problems += [f"{name} API/How: {e}" for e in errors + more]
            await ctx.close()

        # a dark system sees the same links; pictures of each, light and dark, desk and phone
        for size, name in ((DESK, "desk"), (PHONE, "phone")):
            for slug in ("home", "api", "how-it-works"):
                seen = {}
                for how, scheme, query in (("light", "light", ""), ("system dark", "dark", ""),
                                           ("?__theme=dark", "light", "&__theme=dark")):
                    ctx, page, errors = await fresh(browser, size, scheme)
                    await land(page, f"/?tab={slug}{query}")
                    if slug == "how-it-works":
                        await page.evaluate(OPEN_RUN)
                        await page.wait_for_timeout(300)
                    marked = await page.evaluate("() => document.body.classList.contains('dark')")
                    if how != "light" and not marked:
                        problems.append(f"{name} {slug} {how}: the page was never marked dark")
                    seen[how] = await page.evaluate(PAINT)
                    if how != "?__theme=dark":
                        tone = "light" if how == "light" else "dark"
                        base = os.path.join(OUT, f"links-{size['width']}-{tone}")
                        if slug == "home":
                            await clip_shot(page, ".blk-top", f"{base}-header.png", pad=16)
                        elif slug == "api":
                            await clip_shot(page, ".blk-note + .blk-links", f"{base}-api-lede.png", above=90)
                            await clip_shot(page, ".blk-stepshead", f"{base}-api-steps.png", above=40)
                        else:
                            await clip_shot(page, "details.blk-d[open] .blk-links", f"{base}-how.png", above=140)
                    problems += [f"{name} {slug} {how}: {e}" for e in errors]
                    await ctx.close()
                for how in ("system dark", "?__theme=dark"):
                    same = seen[how] == seen["light"] and len(seen["light"]) >= 2
                    print(f"  {name:5} {slug:13} {how:14} {len(seen[how])} links drawn "
                          f"{'the same' if same else 'DIFFERENTLY'}")
                    if not same:
                        problems.append(f"{name} {slug} {how}: {seen[how]} != {seen['light']}")
        await browser.close()

    # every target answers
    for label, href in PROJECT + WIRE + CARDS:
        try:
            req = urllib.request.Request(href, headers={"User-Agent": "blink-check-links"})
            with urllib.request.urlopen(req, timeout=30) as r:
                status, final = r.status, r.geturl()
        except Exception as exc:  # noqa: BLE001 - reported as a problem
            status, final = getattr(exc, "code", str(exc)), href
        ok = status == 200
        print(f"  {label:16} {href} -> {status}{'' if final == href else ' via ' + final} {'ok' if ok else 'WRONG'}")
        if not ok:
            problems.append(f"{label}: {href} answered {status}")

    if problems:
        print(f"\n{len(problems)} problem(s):")
        for line in problems:
            print(" -", line)
        return 1
    print("\nthe links are on every tab, open in a new tab, fit a phone, read the same in dark, and resolve")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
