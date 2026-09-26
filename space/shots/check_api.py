"""The API tab: a link opens it, it shows everything api_doc holds, it reads the same when the system is dark,
and it fits a phone.

    BLINK_MOCK=1 BLINK_PORT=7962 uv run --with gradio==6.28.0 python app.py &
    BLINK_URL=http://127.0.0.1:7962/ uv run --with playwright python shots/check_api.py

Screenshots go to BLINK_SHOTS (default: this folder). Exits non-zero on any problem, a console error included.
"""

import asyncio
import os
import sys

from playwright.async_api import async_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import api_doc  # noqa: E402  (pure python: the tab's copy and examples)

URL = os.environ.get("BLINK_URL", "http://127.0.0.1:7962/").rstrip("/")
OUT = os.environ.get("BLINK_SHOTS") or HERE
DESK = {"width": 1456, "height": 900}
PHONE = {"width": 390, "height": 844}

SELECTED = """() => Array.from(document.querySelectorAll('button[role="tab"][aria-selected="true"]'))
    .map((b) => b.textContent.trim())"""
SHOWN = """() => {
  const fig = document.querySelector('.blk-apifig');
  return !!fig && fig.offsetParent !== null;
}"""
COUNTS = """() => {
  const q = (s) => Array.from(document.querySelectorAll(s)).filter((e) => e.offsetParent !== null);
  return {
    heading: (q('.blk-h2').find((h) => h.textContent.includes('TypeSafe')) || {}).textContent || '',
    rows: q('table.blk-apitable tbody tr').length,
    marks: q('.blk-apifig .blk-mk').length,
    steps: q('ol.blk-steps > li').length,
    folds: q('.blk-api details.blk-d').length,
    draft: q('.blk-copytag').length,
  };
}"""
OPEN_ALL = "() => document.querySelectorAll('.blk-api details').forEach((d) => { d.open = true; })"
# anything in the tab poking past the right edge of the window; code blocks scroll inside themselves
OVERFLOW = """() => {
  const w = document.documentElement.clientWidth;
  const out = [];
  const root = document.querySelector('.blk-apifig').closest('[role="tabpanel"]') || document.body;
  for (const e of root.querySelectorAll('*')) {
    if (e.offsetParent === null || e.closest('.blk-pre')) continue;
    const r = e.getBoundingClientRect();
    if (r.width && r.right > w + 1) {
      out.push(`${e.tagName.toLowerCase()}.${[...e.classList].join('.')} right=${Math.round(r.right)}`);
    }
  }
  return {page: document.documentElement.scrollWidth, window: w, out: out.slice(0, 8)};
}"""
PAINT = """() => {
  const surface = (e) => {
    for (let n = e; n; n = n.parentElement) {
      const bg = getComputedStyle(n).backgroundColor;
      if (bg && bg !== 'rgba(0, 0, 0, 0)' && bg !== 'transparent') return bg;
    }
    return 'none';
  };
  const seen = [];
  for (const e of document.querySelectorAll('body *')) {
    if (e.offsetParent === null || e.children.length) continue;
    const text = (e.textContent || '').trim();
    if (!text) continue;
    seen.push([text.slice(0, 40), getComputedStyle(e).color, surface(e)]);
    if (seen.length >= 1500) break;
  }
  return seen;
}"""


async def fresh(browser, size, scheme="light"):
    ctx = await browser.new_context(viewport=size, color_scheme=scheme, reduced_motion="reduce",
                                    device_scale_factor=1)
    page = await ctx.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
    page.on("console", lambda m: errors.append(f"console.error: {m.text}") if m.type == "error" else None)
    return ctx, page, errors


async def land(page, link: str) -> None:
    await page.goto(URL + link, wait_until="domcontentloaded", timeout=180_000)
    await page.wait_for_selector(".blk-top h1", timeout=60_000)
    try:
        await page.wait_for_function(SHOWN, timeout=20_000)
    except Exception:  # noqa: BLE001 - reported by the caller with what did open
        pass
    await page.wait_for_timeout(900)


async def open_tab(page, label: str) -> None:
    """A narrow window moves the tab strip behind a "More tabs" menu."""
    strip = page.get_by_role("tab", name=label, exact=True)
    if await strip.count() and await strip.first.is_visible():
        await strip.first.click()
        return
    await page.locator('[aria-label="More tabs"]').first.click()
    await page.wait_for_timeout(400)
    await page.get_by_role("button", name=label, exact=True).last.click()


async def main() -> int:
    problems: list[str] = []
    os.makedirs(OUT, exist_ok=True)
    want = {"heading": api_doc.COPY["heading"], "rows": len(api_doc.TABLE_ROWS),
            "marks": 2 * len(api_doc.TABLE_ROWS) + len(api_doc.COPY["legend"]),
            "steps": len(api_doc.COPY["steps"]), "folds": 6, "draft": 1 if api_doc.DRAFT else 0}
    async with async_playwright() as p:
        browser = await p.chromium.launch()

        # links open the tab, on a desk and on a phone (where its button may sit in the menu)
        for size, name in ((DESK, "desk"), (PHONE, "phone")):
            for link in ("/?tab=api", "/#api", "/?tab=API", "/?__theme=dark&tab=api"):
                ctx, page, errors = await fresh(browser, size)
                await land(page, link)
                shown, picked = await page.evaluate(SHOWN), await page.evaluate(SELECTED)
                ok = shown and (name == "phone" or picked[:1] == ["API"])
                print(f"  {name:5} {link:26} -> shown={shown} selected={picked} {'ok' if ok else 'WRONG'}")
                if not ok:
                    problems.append(f"{name} {link}: the API tab did not open (selected {picked})")
                problems += [f"{name} {link}: {e}" for e in errors]
                await ctx.close()

        # it shows everything api_doc holds, and every fold opens
        for size, name in ((DESK, "desk"), (PHONE, "phone")):
            ctx, page, errors = await fresh(browser, size)
            await land(page, "/?tab=api")
            got = await page.evaluate(COUNTS)
            got["heading"] = got["heading"].replace(api_doc.COPY["draft"], "").strip()
            ok = got == want
            print(f"  {name:5} content {got} {'ok' if ok else 'WRONG, want ' + str(want)}")
            if not ok:
                problems.append(f"{name}: content {got} != {want}")
            await page.screenshot(path=os.path.join(OUT, f"api-{name}.png"), full_page=True)
            summaries = page.locator(".blk-api details.blk-d > summary")
            for i in range(await summaries.count()):
                await summaries.nth(i).click()
            await page.wait_for_timeout(400)
            opened = await page.evaluate(
                "() => Array.from(document.querySelectorAll('.blk-api details')).filter((d) => d.open).length")
            codes = await page.evaluate(
                "() => Array.from(document.querySelectorAll('.blk-api .blk-pre'))"
                ".filter((e) => e.offsetParent !== null).length")
            print(f"  {name:5} folds opened by click: {opened}/6, code blocks shown: {codes}")
            if opened != 6 or codes < 12:
                problems.append(f"{name}: {opened} folds opened, {codes} code blocks shown")
            fit = await page.evaluate(OVERFLOW)
            ok = fit["page"] <= fit["window"] and not fit["out"]
            print(f"  {name:5} fit: page {fit['page']}px in a {fit['window']}px window {'ok' if ok else fit['out']}")
            if not ok:
                problems.append(f"{name}: wider than the window: {fit}")
            if name == "phone":
                stacked = await page.evaluate(
                    "() => getComputedStyle(document.querySelector('table.blk-apitable thead')).display")
                if stacked != "none":
                    problems.append(f"phone: the table is not stacked (thead display {stacked})")
            await page.screenshot(path=os.path.join(OUT, f"api-{name}-open.png"), full_page=True)
            problems += [f"{name} content: {e}" for e in errors]
            await ctx.close()

        # reached by clicking, and the address follows
        for size, name in ((DESK, "desk"), (PHONE, "phone")):
            ctx, page, errors = await fresh(browser, size)
            await land(page, "/")
            await open_tab(page, "API")
            try:
                await page.wait_for_function(SHOWN, timeout=15_000)
                await page.wait_for_function("() => window.location.search.includes('tab=api')", timeout=10_000)
                print(f"  {name:5} click -> {await page.evaluate('() => window.location.search')} ok")
            except Exception:  # noqa: BLE001
                problems.append(f"{name}: clicking API left {await page.evaluate('() => window.location.search')!r}")
            problems += [f"{name} click: {e}" for e in errors]
            await ctx.close()

        # a dark system sees the same tab, folds open included
        for size, name in ((DESK, "desk"), (PHONE, "phone")):
            seen = {}
            for how, scheme, link in (("light", "light", "/?tab=api"), ("system dark", "dark", "/?tab=api"),
                                      ("?__theme=dark", "light", "/?tab=api&__theme=dark")):
                ctx, page, errors = await fresh(browser, size, scheme)
                await land(page, link)
                await page.evaluate(OPEN_ALL)
                await page.wait_for_timeout(400)
                marked = await page.evaluate("() => document.body.classList.contains('dark')")
                if how != "light" and not marked:
                    problems.append(f"{name} {how}: the page was never marked dark")
                seen[how] = await page.evaluate(PAINT)
                if how == "system dark" and name == "desk":
                    await page.screenshot(path=os.path.join(OUT, "api-desk-dark.png"), full_page=True)
                problems += [f"{name} {how}: {e}" for e in errors]
                await ctx.close()
            light = seen["light"]
            for how in ("system dark", "?__theme=dark"):
                pairs = [(a, b) for a, b in zip(light, seen[how], strict=False) if a[0] == b[0]]
                off = [(a[0], a[1:], b[1:]) for a, b in pairs if a[1:] != b[1:]]
                ok = not off and len(pairs) >= 0.9 * max(len(light), 1)
                verdict = "ok" if ok else "WRONG"
                print(f"  {name:5} {how:14} {len(pairs):4} texts, {len(off)} drawn differently {verdict}")
                if not ok:
                    problems.append(f"{name} {how}: {len(off)} of {len(pairs)} differ, e.g. {off[:3]}")
        await browser.close()
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for line in problems:
            print(" -", line)
        return 1
    print("\nthe API tab opens by link and by click, shows all of api_doc, reads the same in dark, and fits a phone")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
