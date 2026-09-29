"""Open the app by link and check the right tab comes up, then that clicks update the address.

    BLINK_MOCK=1 BLINK_PORT=7897 uv run --with gradio==6.28.0 python app.py &
    BLINK_URL=http://127.0.0.1:7897/ uv run --with playwright python shots/check_deeplinks.py

Exits non-zero if a link opens the wrong view, if a tab click leaves the address behind,
or if the console reports anything.

Where Screen click is served (blink-mimo-9b among BLINK_MODELS), it sits on the Computer use tab and every
preset has a link, ?tab=computer-use&shot=<key>: it must open that preset, an unknown shot the first one, and the
older ?tab=use-cases&case=screen&shot=<key> must land the same way and be rewritten to the new form. Picking a
preset must rewrite the address in place and tell the embedding page; a visitor's own screenshot, or leaving the
tab, drops the shot. BLINK_HUB_URL (the Space's huggingface.co page) also opens one older shot link through the
Hub and checks that the Hub's address follows a pick.
"""

import asyncio
import json
import os
import sys

from playwright.async_api import async_playwright
import live  # noqa: E402  (live-Space allowances: shots/live.py)

URL = os.environ.get("BLINK_URL", "http://127.0.0.1:7897/").rstrip("/")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("BLINK_SHOTS") or HERE
HUB = os.environ.get("BLINK_HUB_URL", "").rstrip("/")
with open(os.path.join(HERE, "..", "screens", "screens.json"), encoding="utf-8") as fh:
    PRESETS = json.load(fh)["presets"]

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

# what Screen click holds, and the address it left
HELD = """() => ({
  shot: ((document.querySelector('#screen-out .blk-scr') || {}).dataset || {}).shot || '',
  task: (document.querySelector('#screen-task textarea') || {}).value || '',
  search: window.location.search,
  posted: (window.__posted || []).slice(-1)[0] || null,
  entries: history.length,
})"""

# every message the page sends its embedder, kept instead of sent: top level, the parent is the page
POSTS = """(() => {
  window.__posted = [];
  window.postMessage = function (msg, origin) { window.__posted.push({msg, origin}); };
})()"""


def query(search: str) -> dict:
    from urllib.parse import parse_qs

    return {k: v[0] for k, v in parse_qs(search.lstrip("?")).items()}


async def screen_links(browser, problems: list) -> None:
    """Every Screen click preset by link, on the Computer use tab, the older use-case links landing there,
    a pick rewriting the address, and what drops the shot."""
    ctx = await browser.new_context(viewport={"width": 1280, "height": 900})
    await ctx.add_init_script(POSTS)
    page = await ctx.new_page()
    page.on("pageerror", lambda e: problems.append(f"pageerror: {e}"))
    page.on("console", lambda m: problems.append(f"console.error: {m.text}") if live.app_error(m) else None)

    async def holding(link, key):
        # a live Space can take a while to serve a page; locally these are the usual waits
        await page.goto(URL + link, wait_until="load", timeout=live.answer_wait(60000))
        await page.wait_for_selector(".blk-top h1", timeout=live.answer_wait(30000))
        try:
            await page.wait_for_selector(f'#screen-out .blk-scr[data-shot="{key}"]', timeout=live.answer_wait(30000))
        except Exception:  # noqa: BLE001 - reported below with what it holds
            pass
        await page.wait_for_timeout(900)
        return await page.evaluate(SELECTED), await page.evaluate(HELD)

    # the use cases open on the text Next click; Screen click, where it is served, is on the Computer use tab
    picked, held = await holding("/?tab=use-cases", "none")
    print(f"  {'/?tab=use-cases':52s} -> {picked} {'ok' if picked[:2] == ['Use cases', 'Next click'] else 'WRONG'}")
    if picked[:2] != ["Use cases", "Next click"]:
        problems.append(f"?tab=use-cases opened {picked}, not Next click")
    picked, held = await holding("/?tab=computer-use", PRESETS[0]["key"])
    served = picked[:1] == ["Computer use"] and held["shot"] == PRESETS[0]["key"]
    if not served:
        print(f"  Screen click is not served here ({picked}); its links are not checked")
        await ctx.close()
        return
    # the home page's computer-use block: its runs and Screen click on their tab, the text Next click in place
    for button, want, q_want in (("#home-watch", ["Computer use"], {"tab": "computer-use"}),
                                 ("#home-screen", ["Computer use"], {"tab": "computer-use"}),
                                 ("#home-nextclick", ["Use cases", "Next click"], {"case": "nextclick"})):
        await page.goto(URL + "/", wait_until="load", timeout=live.answer_wait(60000))
        await page.wait_for_selector("#home-nextclick", timeout=live.answer_wait(30000))
        if not await page.locator(button).count():
            print(f"  home {button:47s} -> not on this page (no runs to show)")
            continue
        await page.wait_for_timeout(1200)
        await page.evaluate("() => { window.__stayed = true; }")
        await page.locator(button).click()
        await page.wait_for_timeout(2500)
        got = await page.evaluate(SELECTED)
        q = query(await page.evaluate("() => window.location.search"))
        stayed = await page.evaluate("() => window.__stayed === true")
        good = got[:len(want)] == want and all(q.get(k) == v for k, v in q_want.items()) and stayed
        if button == "#home-screen":
            top = await page.evaluate("() => Math.round((document.getElementById('cua-try') || document.body)"
                                      ".getBoundingClientRect().top)")
            good = good and 0 <= top < 450  # and Screen click is in view
        print(f"  home {button:47s} -> {got[:2]} {q} {'ok' if good else 'WRONG'}")
        if not good:
            problems.append(f"home {button} opened {got} at {q} (stayed: {stayed})")
    # Screen click's older links land on the new tab, on their preset, and are written the new way
    for link in ("/?tab=use-cases&case=screen", "/?case=screen"):
        picked, held = await holding(link, PRESETS[0]["key"])
        q = query(held["search"])
        good = picked[:1] == ["Computer use"] and held["shot"] == PRESETS[0]["key"] and "case" not in q \
            and "shot" not in q and q.get("tab") == "computer-use"
        print(f"  {link:52s} -> {picked} {held['shot']} {held['search']} {'ok' if good else 'WRONG'}")
        if not good:
            problems.append(f"{link} opened {picked} {held['shot']!r} and left {held['search']!r}")
    by_task = {p["key"]: p["task"] for p in PRESETS}
    for key in [p["key"] for p in PRESETS] + ["nonsense"]:
        want = key if key in by_task else PRESETS[0]["key"]
        for link in (f"/?tab=computer-use&shot={key}", f"/?tab=use-cases&case=screen&shot={key}"):
            picked, held = await holding(link, want)
            q = query(held["search"])
            good = (picked[:1] == ["Computer use"] and held["shot"] == want and held["task"] == by_task[want]
                    and q.get("shot") == want and q.get("tab") == "computer-use" and "case" not in q)
            print(f"  {link:52s} -> {held['shot']} {held['search']} {'ok' if good else 'WRONG'}")
            if not good:
                problems.append(f"{link} held {held['shot']!r} / {held['task']!r} at {held['search']!r} ({picked})")
        if key == "done":
            await page.screenshot(path=os.path.join(OUT, "deeplink-screen-done.png"))

    # a pick rewrites the address in place and tells the embedding page
    before = held["entries"]
    for p in PRESETS[1:3]:
        await page.get_by_role("button", name=p["label"], exact=True).first.click()
        await page.wait_for_timeout(1200)
        held = await page.evaluate(HELD)
        posted = held["posted"] or {}
        told = query((posted.get("msg") or {}).get("queryString", ""))
        good = (query(held["search"]).get("shot") == p["key"] and told.get("shot") == p["key"]
                and posted.get("origin") == "https://huggingface.co")
        print(f"  pick {p['label']:47s} -> {held['search']} told {told.get('shot')} {'ok' if good else 'WRONG'}")
        if not good:
            problems.append(f"picking {p['label']} left {held['search']!r} and told {posted}")
    if held["entries"] != before:
        problems.append(f"picking presets added {held['entries'] - before} history entries")

    # a visitor's own screenshot has no link; a preset brings the shot back
    await page.locator(".blk-screenrow input[type=file]").set_input_files(
        os.path.join(HERE, "..", "screens", f"{PRESETS[0]['key']}.png"))
    try:
        await page.wait_for_selector('#screen-out .blk-shot[data-draw="1"]', timeout=live.answer_wait(20000))
    except Exception:  # noqa: BLE001
        pass
    await page.wait_for_timeout(900)
    held = await page.evaluate(HELD)
    q = query(held["search"])
    print(f"  upload a screenshot{'':33s} -> {held['search']}")
    if "shot" in q or q.get("tab") != "computer-use" or not held["shot"].startswith("upload-"):
        problems.append(f"an upload held {held['shot']!r} and left {held['search']!r}")
    last = PRESETS[-1]
    await page.get_by_role("button", name=last["label"], exact=True).first.click()
    await page.wait_for_timeout(1200)
    held = await page.evaluate(HELD)
    print(f"  pick {last['label']:47s} -> {held['search']}")
    if query(held["search"]).get("shot") != last["key"]:
        problems.append(f"a preset after an upload left {held['search']!r}")

    # leaving the tab drops the shot; coming back brings the same preset back
    for tab, want in (("Use cases", {"tab": "use-cases"}), ("Computer use", {"tab": "computer-use", "shot": last["key"]}),
                      ("Results", {"tab": "results"})):
        await page.get_by_role("tab", name=tab, exact=True).first.click()
        await page.wait_for_timeout(1000)
        q = query(await page.evaluate("() => window.location.search"))
        good = all(q.get(k) == v for k, v in want.items()) and ("shot" in q) == ("shot" in want)
        print(f"  click {tab:46s} -> {q} {'ok' if good else 'WRONG'}")
        if not good:
            problems.append(f"clicking {tab} left {q}")
    await ctx.close()


async def hub_frame(page, key: str, seconds: int = 180):
    """The Space's frame on the Hub's page once it holds the preset: the Hub may swap its frame while it loads."""
    for _ in range(seconds * 2):
        for frame in page.frames:
            if ".hf.space" not in frame.url:
                continue
            try:
                held = await frame.evaluate(
                    "() => ((document.querySelector('#screen-out .blk-scr') || {}).dataset || {}).shot || ''")
            except Exception:  # noqa: BLE001 - a frame the Hub has swapped out
                continue
            if held == key:
                return frame
        await page.wait_for_timeout(500)
    return None


async def hub_link(browser, problems: list) -> None:
    """The same shot link through the Hub's page: its frame opens the preset, and a pick moves the Hub's address."""
    page = await browser.new_page(viewport={"width": 1280, "height": 900})
    key = PRESETS[-1]["key"]
    link = f"{HUB}?tab=use-cases&case=screen&shot={key}"  # the older form the cards and docs carry
    await page.goto(link, wait_until="load", timeout=live.answer_wait(60000))
    frame = await hub_frame(page, key)
    if frame is None:
        problems.append(f"{link}: the Space's frame never held {key}")
        await page.close()
        return
    print(f"  {link} -> the Space's frame holds {key} ({frame.url.split('?')[-1]})")
    other = PRESETS[1]
    await frame.get_by_role("button", name=other["label"], exact=True).first.click()
    moved = ""
    for _ in range(40):
        await page.wait_for_timeout(500)
        moved = page.url
        if f"shot={other['key']}" in moved:
            break
    print(f"  pick {other['label']} in the Hub's frame -> {moved}")
    if f"shot={other['key']}" not in moved or "tab=computer-use" not in moved:
        problems.append(f"a pick in the Hub's frame left the Hub at {moved}")
    await page.screenshot(path=os.path.join(OUT, "deeplink-hub-screen.png"))
    await page.close()

async def main() -> int:
    problems: list[str] = []
    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        page = await browser.new_page(viewport={"width": 1280, "height": 900}, device_scale_factor=2)
        page.on("pageerror", lambda e: problems.append(f"pageerror: {e}"))
        page.on("console", lambda m: problems.append(f"console.error: {m.text}")
                if live.app_error(m) else None)

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

        await page.goto(URL + "/", wait_until="load")
        await page.wait_for_selector(".blk-top h1", timeout=30000)
        links = LINKS
        if await page.get_by_role("tab", name="Computer use", exact=True).count():
            links = LINKS + [("/?tab=computer-use", "Computer use", None), ("/#computer-use", "Computer use", None)]
        for link, want_tab, want_case in links:
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
        await page.locator("button.blk-tile-hit").filter(has_text="All results").first.click()
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
        # Next click may be the case the tab opens on; leave it first, so the click is one
        await page.get_by_role("tab", name="Policy checks", exact=True).first.click()
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
        await screen_links(browser, problems)
        if HUB:
            await hub_link(browser, problems)
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
