"""Open the app by link and check the right tab comes up, then that clicks update the address.

    BLINK_MOCK=1 BLINK_PORT=7897 uv run --with gradio==6.28.0 python app.py &
    BLINK_URL=http://127.0.0.1:7897/ uv run --with playwright python shots/check_deeplinks.py

Exits non-zero if a link opens the wrong view, if a tab click leaves the address behind,
or if the console reports anything.
"""

import asyncio
import os
import sys

from playwright.async_api import async_playwright

URL = os.environ.get("BLINK_URL", "http://127.0.0.1:7897/").rstrip("/")
OUT = os.path.dirname(os.path.abspath(__file__))

# (link, expected top tab, expected sub tab or None)
LINKS = [
    ("/", "Home", None),
    ("/?tab=playground", "Playground", None),
    ("/?tab=ask", "Ask", None),
    ("/#ask", "Ask", None),
    ("/?tab=results", "Results", None),
    ("/#results", "Results", None),
    ("/?tab=RESULTS", "Results", None),
    ("/?tab=use-cases&case=nextclick", "Use cases", "Next click"),
    ("/?case=rag", "Use cases", "Passage check"),
    ("/?tab=use-cases&case=policy", "Use cases", "Policy checks"),
    ("/?tab=how-it-works", "How it works", None),
    ("/?tab=api", "API", None),
    ("/#api", "API", None),
    ("/?tab=nonsense", "Home", None),
    ("/?__theme=light&tab=results", "Results", None),
]

SELECTED = """() => Array.from(
    document.querySelectorAll('button[role="tab"][aria-selected="true"]')
  ).map((b) => b.textContent.trim())"""


async def main() -> int:
    problems: list[str] = []
    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        page = await browser.new_page(viewport={"width": 1280, "height": 900}, device_scale_factor=2)
        page.on("pageerror", lambda e: problems.append(f"pageerror: {e}"))
        page.on("console", lambda m: problems.append(f"console.error: {m.text}")
                if m.type == "error" else None)

        async def opens(link, want_tab, want_case):
            """Wait for the view the link names rather than for a guess at how long."""
            await page.goto(URL + link, wait_until="load")
            await page.wait_for_selector(".blk-top h1", timeout=30000)
            wanted = [want_tab] + ([want_case] if want_case else [])
            try:
                await page.wait_for_function(
                    """(want) => {
                        const got = Array.from(document.querySelectorAll(
                            'button[role="tab"][aria-selected="true"]'))
                            .map((b) => b.textContent.trim());
                        return want.every((w, i) => got[i] === w);
                    }""", arg=wanted, timeout=20000)
            except Exception:  # noqa: BLE001 - reported below with what it did open
                pass
            await page.wait_for_timeout(700)
            return await page.evaluate(SELECTED)

        for link, want_tab, want_case in LINKS:
            picked = await opens(link, want_tab, want_case)
            got_tab = picked[0] if picked else None
            got_case = picked[1] if len(picked) > 1 else None
            ok = got_tab == want_tab and (want_case is None or got_case == want_case)
            print(f"  {link:38s} -> {picked} {'ok' if ok else 'WRONG'}")
            if not ok:
                problems.append(f"{link} opened {picked}, wanted {want_tab}/{want_case}")
            # the address a link leaves behind has to be the link
            search = await page.evaluate("() => window.location.search")
            if want_case and f"case={want_case and want_case}" and "case=" not in search:
                problems.append(f"{link} dropped its case: {search}")
            if not want_case and "case=" in search:
                problems.append(f"{link} invented a case: {search}")
            if ok and search:
                again = await opens("/" + search, want_tab, want_case)
                if again != picked:
                    problems.append(f"{link} -> {search} reopened {again}")
                print(f"  {'reload ' + search:38s} -> {again}")
            if link == "/?tab=results":
                await page.screenshot(path=os.path.join(OUT, "deeplink-results.png"))
            if link.endswith("case=nextclick"):
                await page.screenshot(path=os.path.join(OUT, "deeplink-next-click.png"))
            if "__theme" in link:
                if "__theme=light" not in search:
                    problems.append(f"lost __theme: {search}")

        # a card on the home page moves to its tab without a reload
        await page.goto(URL + "/", wait_until="load")
        await page.wait_for_selector("button.blk-tile-hit", timeout=30000)
        await page.wait_for_timeout(1500)
        marked = await page.evaluate("() => { window.__stayed = true; return true; }")
        await page.locator("button.blk-tile-hit").nth(3).click()
        await page.wait_for_timeout(1200)
        picked = await page.evaluate(SELECTED)
        stayed = await page.evaluate("() => window.__stayed === true")
        search = await page.evaluate("() => window.location.search")
        print(f"  card -> Results      -> {picked} {search}")
        if picked[:1] != ["Results"]:
            problems.append(f"the results card opened {picked}")
        if not stayed:
            problems.append("the results card reloaded the page")
        if "tab=results" not in search:
            problems.append(f"the results card left {search!r}")
        outside = await page.evaluate(
            """() => Array.from(document.querySelectorAll('.blk-cards.out a'))
                 .map((a) => [a.getAttribute('target'), a.getAttribute('rel')])""")
        inapp = await page.evaluate(
            """() => document.querySelectorAll('.blk-cards a[data-tab]').length""")
        print(f"  in-app cards as links  -> {inapp}")
        if inapp:
            problems.append("an in-app card is still a link that reloads")
        print(f"  outside links        -> {len(outside)} in a new tab")
        if not outside or any(t != "_blank" or "noopener" not in (r or "") for t, r in outside):
            problems.append(f"outside links are not safe: {outside}")

        # clicking a tab has to leave a link behind
        await page.goto(URL + "/", wait_until="load")
        await page.wait_for_selector(".blk-top h1", timeout=30000)
        await page.wait_for_timeout(1400)
        before = await page.evaluate("() => history.length")
        for label, want in (("Results", "tab=results"), ("How it works", "tab=how-it-works")):
            await page.get_by_role("tab", name=label, exact=True).first.click()
            await page.wait_for_timeout(900)
            search = await page.evaluate("() => window.location.search")
            print(f"  click {label:14s} -> {search}")
            if want not in search:
                problems.append(f"clicking {label} left {search!r}")
        await page.get_by_role("tab", name="Use cases", exact=True).first.click()
        await page.wait_for_timeout(900)
        await page.get_by_role("tab", name="Next click", exact=True).first.click()
        await page.wait_for_timeout(900)
        search = await page.evaluate("() => window.location.search")
        print(f"  click Next click     -> {search}")
        for want in ("tab=use-cases", "case=nextclick"):
            if want not in search:
                problems.append(f"sub-tab click left {search!r}")
        after = await page.evaluate("() => history.length")
        if after != before:
            problems.append(f"tab clicks added {after - before} history entries")
        print(f"  history entries added: {after - before}")

        # and the link that address gives has to reopen the same view
        await page.goto(URL + "/" + search, wait_until="load")
        await page.wait_for_selector(".blk-top h1", timeout=30000)
        await page.wait_for_timeout(1500)
        picked = await page.evaluate(SELECTED)
        print(f"  round trip           -> {picked}")
        if picked[:2] != ["Use cases", "Next click"]:
            problems.append(f"round trip opened {picked}")
        await browser.close()

    noisy = [p for p in problems if "favicon" not in p.lower()]
    if noisy:
        print(f"\n{len(noisy)} problem(s):")
        for p in sorted(set(noisy)):
            print(" -", p)
        return 1
    print("\nevery link opens its view, every click leaves one behind")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
