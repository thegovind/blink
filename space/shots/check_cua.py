"""The Computer use tab at a desk and on a phone: the runs, the player, the explainer, Screen click and the tiles.

    cd space && BLINK_MOCK=1 BLINK_MODELS=thegovind/blink-4b,thegovind/blink-mimo-9b BLINK_GALLERY=/path/gallery.json \\
        BLINK_PORT=7941 uv run --with gradio==6.28.0 python app.py &
    BLINK_URL=http://127.0.0.1:7941/ BLINK_SHOTS=/tmp/shots uv run --with playwright python shots/check_cua.py

Against the live Space: BLINK_URL=https://thegovind-blink.hf.space/ BLINK_WAIT_MS=150000 (nothing here presses
Decide, so no model runs). Checks, where the page has runs to show:
- the tab follows Home; the strip's reels loop muted, inline, with a poster, and never with sound;
- a card opens its run: the address names it, the video is muted, inline and has a poster, and every file the
  run names (poster, video, steps) is served;
- the step-through draws one mark per box, exactly on the boxes of the marked screenshot, lights the box blink
  picked, and keeps the address in step (?scenario=..&step=N), which reopens the same step;
- another model and a failed run open by link; closing brings the strip back and clears the address;
- the explainer's six nodes play once in view and end on the image-token count;
- a tile with sound never starts by itself; old Screen click links land on the tab with their preset in view;
- nothing scrolls sideways, and the console stays clean.
Screenshots go to BLINK_SHOTS (default: this folder) as cua-<width>-*.png. Exits non-zero on any problem.
"""

import asyncio
import json
import os
import sys

import live  # live-Space allowances: shots/live.py
from playwright.async_api import async_playwright

URL = os.environ.get("BLINK_URL", "http://127.0.0.1:7941/").rstrip("/")
OUT = os.environ.get("BLINK_SHOTS") or os.path.dirname(os.path.abspath(__file__))
DESK = {"width": 1440, "height": 900}
PHONE = {"width": 390, "height": 844}

SELECTED = """() => Array.from(document.querySelectorAll('button[role="tab"][aria-selected="true"]'))
  .map((b) => b.textContent.trim())"""
DATA = """() => { const r = document.querySelector('.blk-cuax'); return r ? JSON.parse(r.dataset.g || 'null') : undefined; }"""
OVERFLOW = """() => ({page: document.documentElement.scrollWidth, win: window.innerWidth})"""
# where each mark sits on the screenshot, in the screenshot's own pixels
MARKS = """() => {
  const img = document.querySelector('.blk-cuax-screen .blk-shot img');
  if (!img) return null;
  const r = img.getBoundingClientRect();
  const k = img.naturalWidth / r.width;
  return {w: img.naturalWidth, h: img.naturalHeight, done: img.complete && img.naturalWidth > 0,
    marks: Array.from(document.querySelectorAll('.blk-cuax-screen .blk-som')).map((el) => {
      const b = el.getBoundingClientRect();
      return {n: Number(el.dataset.n), win: el.classList.contains('win'),
              box: [(b.left - r.left) * k, (b.top - r.top) * k, (b.right - r.left) * k, (b.bottom - r.top) * k]};
    })};
}"""
STATUS = """async (urls) => Promise.all(urls.map((u) => fetch(u, {method: 'GET', credentials: 'omit'})
  .then((r) => r.status).catch(() => 0)))"""


def query(page_url: str) -> dict:
    from urllib.parse import parse_qs, urlsplit

    return {k: v[0] for k, v in parse_qs(urlsplit(page_url).query).items()}


async def check(pw, size: dict, problems: list) -> None:
    tag = f"{size['width']}px"
    phone = size is PHONE

    def ok(name, good):
        print(f"  {'ok ' if good else 'BAD'} {tag} {name}")
        if not good:
            problems.append(f"{tag} {name}")

    browser = await pw.chromium.launch()
    ctx = await browser.new_context(viewport=size, device_scale_factor=1, has_touch=phone, is_mobile=phone)
    page = await ctx.new_page()
    errors = []
    page.on("console", lambda m: errors.append(m.text) if live.app_error(m) else None)
    page.on("pageerror", lambda e: errors.append(str(e)))

    async def land(link, selector=".blk-cuax"):
        await page.goto(URL + link, wait_until="load", timeout=live.answer_wait(60000))
        await page.wait_for_selector(selector, timeout=live.answer_wait(60000))
        await page.wait_for_timeout(1200)

    await land("/?tab=computer-use")
    ok(f"the tab opens by link: {await page.evaluate(SELECTED)}", (await page.evaluate(SELECTED))[:1] == ["Computer use"])
    if not phone:
        names = await page.evaluate("() => Array.from(document.querySelectorAll('button[role=\"tab\"]'))"
                                    ".map((b) => b.textContent.trim())")
        ok(f"it follows Home: {names[:3]}", names[:2] == ["Home", "Computer use"])
    g = await page.evaluate(DATA)
    served = await page.locator("#screen-out").count() > 0
    await page.screenshot(path=os.path.join(OUT, f"cua-{size['width']}-top.png"))
    if g:
        reels = await page.evaluate("""() => Array.from(document.querySelectorAll('.blk-cuax-reel video')).map((v) => ({
            muted: v.muted, inline: v.hasAttribute('playsinline'), loop: v.loop, poster: !!v.getAttribute('poster'),
            auto: v.hasAttribute('autoplay')}))""")
        ok(f"{len(reels)} reels loop muted, inline, with a poster", reels and all(
            r["muted"] and r["inline"] and r["loop"] and r["poster"] and not r["auto"] for r in reels))
        cards = await page.locator(".blk-cuax-card").count()
        ok(f"a card for every scenario ({cards} of {len(g['scenarios'])})", cards == len(g["scenarios"]))
        dots = await page.evaluate("() => Array.from(document.querySelectorAll('.blk-cuax-card')).map((c) => "
                                   "c.querySelectorAll('.blk-cuax-dots i').length)")
        runs = [s["models"].get(g["vision_model"], next(iter(s["models"].values())))["stats"]["episodes"]
                for s in g["scenarios"]]
        ok(f"one dot per run on every card {dots}", dots == runs)

        # a card opens its run, and the address names it
        first = g["scenarios"][0]
        await page.locator(".blk-cuax-card").first.click()
        await page.wait_for_selector(".blk-cuax-player", timeout=10000)
        await page.wait_for_timeout(1500)
        q = query(page.url)
        ok(f"a card opens its run: {q}", q.get("scenario") == first["id"] and q.get("tab") == "computer-use")
        item = first["models"].get(g["vision_model"]) or next(iter(first["models"].values()))
        ep = g["episodes"][item["showcase"]]
        video = await page.evaluate("""() => { const v = document.querySelector('.blk-cuax-screen video');
            return v ? {muted: v.muted, inline: v.hasAttribute('playsinline'), poster: v.getAttribute('poster'),
                        src: (v.querySelector('source') || {}).src || ''} : null; }""")
        if ep.get("video"):
            ok(f"its video is muted, inline, with a poster: {video and video['src'][-40:]}",
               video and video["muted"] and video["inline"] and video["poster"] and video["src"].endswith(".mp4"))
        files = [g["media_base"] + ep[k] for k in ("poster", "video", "preview", "steps_url") if ep.get(k)]
        status = await page.evaluate(STATUS, files)
        ok(f"every file the run names is served {status}", all(s == 200 for s in status))
        await page.locator(".blk-cuax-player").screenshot(path=os.path.join(OUT, f"cua-{size['width']}-player-video.png"))

        # the steps: one mark per box, on the boxes, the pick lit
        await page.locator(".blk-cuax-views button", has_text="Steps").click()
        await page.wait_for_selector(".blk-cuax-screen .blk-shot img", timeout=live.answer_wait(20000))
        await page.wait_for_function("() => { const i = document.querySelector('.blk-cuax-screen .blk-shot img');"
                                     " return i && i.complete && i.naturalWidth > 0; }", timeout=live.answer_wait(20000))
        steps = await page.evaluate("async (u) => (await fetch(u, {credentials: 'omit'})).json()",
                                    g["media_base"] + ep["steps_url"])
        geo = await page.evaluate(MARKS)
        s = steps[0]
        pad = ep.get("pad") or 6
        off = [m["n"] for m, b in zip(geo["marks"], s["boxes"])
               if any(abs(a - c) > pad + 2 for a, c in zip(m["box"], (b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad)))
               and not phone]
        ok(f"step 1 draws {len(geo['marks'])} marks for {len(s['boxes'])} boxes, on them {off[:3]}",
           geo["done"] and len(geo["marks"]) == len(s["boxes"]) and not off)
        lit = [m["n"] for m in geo["marks"] if m["win"]]
        ok(f"the pick is lit: {lit} (blink picked {s['pick']}, {s['act']})",
           lit == ([s["pick"]] if s["pick"] and s["act"] != "done" else []))
        q = query(page.url)
        ok(f"the address follows the steps: {q.get('step')}", q.get("step") == "1")
        await page.locator(".blk-cuax-player").screenshot(path=os.path.join(OUT, f"cua-{size['width']}-player-steps.png"))
        if len(steps) > 1:
            await page.locator('.blk-cuax-walk [data-do="next"]').click()
            await page.wait_for_timeout(900)
            q = query(page.url)
            on = await page.evaluate("() => (document.querySelector('.blk-cuax-rail .on') || {}).textContent")
            ok(f"next moves to step 2: address {q.get('step')}, rail {on}", q.get("step") == "2" and on == "2")
            link = page.url.split(URL, 1)[1]
            await land(link, ".blk-cuax-screen .blk-shot")
            on = await page.evaluate("() => (document.querySelector('.blk-cuax-rail .on') || {}).textContent")
            ok(f"that address reopens step 2: {link} -> {on}", on == "2")
            # the steps play on their own from there, and stop when asked
            await page.locator(".blk-cuax-autoplay").click()
            await page.wait_for_timeout(1700 * min(2, len(steps) - 2) + 600)
            moved = await page.evaluate("() => (document.querySelector('.blk-cuax-rail .on') || {}).textContent")
            await page.locator(".blk-cuax-autoplay").click()
            await page.wait_for_timeout(2000)
            held = await page.evaluate("() => (document.querySelector('.blk-cuax-rail .on') || {}).textContent")
            ok(f"play walks the steps ({on} -> {moved}) and pause holds ({held})",
               int(moved) == min(len(steps), 2 + min(2, len(steps) - 2)) and held == moved)
        if phone:
            order = await page.evaluate("""() => ['.blk-cuax-screen', '.blk-cuax-rail', '.blk-cuax-side']
              .map((s) => Math.round(document.querySelector(s).getBoundingClientRect().top))""")
            ok(f"the steps sit between the screenshot and its answers {order}", order == sorted(order))

        # another model, and a failed run, by link
        other = next(((sc, m) for sc in g["scenarios"] for m in sc["models"] if m != g["vision_model"]), None)
        if other:
            sc, m = other
            await land(f"/?tab=computer-use&scenario={sc['id']}&model={m}", ".blk-cuax-player")
            who = await page.evaluate("() => (document.querySelector('.blk-cuax-who') || {}).textContent || ''")
            ok(f"a link opens {m} on {sc['id']}: {who!r}", who.startswith(m))
        failed = next(((sc, m) for sc in g["scenarios"] for m, item in sc["models"].items() if item.get("failure")),
                      None)
        if failed:
            sc, m = failed
            await land(f"/?tab=computer-use&scenario={sc['id']}&model={m}&run=failed", ".blk-cuax-player")
            await page.wait_for_timeout(600)
            said = await page.evaluate("() => (document.querySelector('.blk-cuax-outcome') || {}).className || ''")
            ok(f"a failed run opens as one: {said!r}", "no" in said.split())
        await page.locator(".blk-cuax-x").click()
        await page.wait_for_timeout(700)
        q = query(page.url)
        ok(f"closing brings the strip back and clears the address: {q}",
           await page.locator(".blk-cuax-strip").count() == 1 and "scenario" not in q)

    # how it reads a screenshot
    how = page.locator(".blk-cx")
    if await how.count():
        await how.scroll_into_view_if_needed()
        await page.wait_for_timeout(4500)
        nodes = await page.locator(".blk-cx-flow > li").count()
        count = await page.evaluate("() => { const c = document.querySelector('.blk-cx-count');"
                                    " return c ? [c.textContent, Number(c.dataset.to).toLocaleString('en-US')] : null; }")
        seen = await page.evaluate("() => getComputedStyle(document.querySelector('.blk-cx-flow > li:last-child')).opacity")
        ok(f"the explainer shows six nodes, ends on {count}, fully in view ({seen})",
           nodes == 6 and (count is None or count[0] == count[1]) and float(seen) > 0.99)
        await how.screenshot(path=os.path.join(OUT, f"cua-{size['width']}-explainer.png"))

    # beyond clicks: a tile with sound waits for a tap
    tiles = page.locator(".blk-cuax-tile")
    if await tiles.count():
        spec = json.loads(await page.evaluate("() => document.querySelector('.blk-cuax-more').dataset.t"))
        for i, t in enumerate(spec["more"]):
            await tiles.nth(i).scroll_into_view_if_needed()
            await tiles.nth(i).click()
            await page.wait_for_timeout(600)
            v = await page.evaluate("""() => { const v = document.querySelector('.blk-cuax-open video');
                return v ? {auto: v.hasAttribute('autoplay'), muted: v.muted, paused: v.paused,
                            controls: v.hasAttribute('controls'), inline: v.hasAttribute('playsinline')} : null; }""")
            if t.get("video"):
                good = v and v["controls"] and v["inline"] and (not t.get("sound") or (v["paused"] and not v["auto"]))
                ok(f"tile {t['id']}: its video {'waits for a tap' if t.get('sound') else 'plays muted'} {v}", good)
            if i == len(spec["more"]) - 1:
                await page.locator(".blk-cuax-more").screenshot(path=os.path.join(OUT, f"cua-{size['width']}-tiles.png"))
            await tiles.nth(i).click()  # and closes again
            await page.wait_for_timeout(300)

    # Screen click lives here: its old links land on the tab with the preset in view
    if served:
        await land("/?tab=use-cases&case=screen&shot=done", "#screen-out .blk-scr[data-shot=\"done\"]")
        await page.wait_for_timeout(2500)
        top = await page.evaluate("() => Math.round(document.getElementById('cua-try').getBoundingClientRect().top)")
        q = query(page.url)
        ok(f"an old Screen click link lands here, in view ({top}px), as {q}",
           (await page.evaluate(SELECTED))[:1] == ["Computer use"] and 0 <= top < size["height"] / 2
           and q.get("tab") == "computer-use" and q.get("shot") == "done" and "case" not in q)
    room = await page.evaluate(OVERFLOW)
    ok(f"nothing scrolls sideways {room}", room["page"] <= room["win"])
    ok(f"no console errors {errors[:3]}", not errors)
    await ctx.close()
    await browser.close()


async def main() -> int:
    problems: list[str] = []
    os.makedirs(OUT, exist_ok=True)
    async with async_playwright() as pw:
        for size in (DESK, PHONE):
            await check(pw, size, problems)
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems:
            print(" -", p)
        return 1
    print("\nthe Computer use tab plays, steps, links and explains")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
