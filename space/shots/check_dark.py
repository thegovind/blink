"""A visitor whose system is dark sees the same page as everyone else.

    BLINK_MOCK=1 BLINK_PORT=7921 uv run --with gradio==6.28.0 python app.py &
    BLINK_URL=http://127.0.0.1:7921/ uv run --with playwright python shots/check_dark.py

Gradio marks the page dark when the system prefers it, or when the address carries
`?__theme=dark`, as the Hub can pass. The page has one palette, so every visible piece of
text has to be drawn in the same colour, on the same surface, either way.
"""
import asyncio
import os
import sys

from playwright.async_api import async_playwright

URL = os.environ.get("BLINK_URL", "http://127.0.0.1:7921/").rstrip("/")
SIZES = ({"width": 1456, "height": 900}, {"width": 390, "height": 844})
TABS = ("home", "playground", "ask", "use-cases", "results", "how-it-works", "api")
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
    if (seen.length >= 800) break;
  }
  return seen;
}"""
DARK = "() => document.body.classList.contains('dark')"


async def paint(browser, size, scheme, query):
    ctx = await browser.new_context(viewport=size, color_scheme=scheme, reduced_motion="reduce")
    page = await ctx.new_page()
    await page.goto(f"{URL}/?{query}", wait_until="domcontentloaded", timeout=180_000)
    await page.wait_for_selector(".blk-top h1", timeout=60_000)
    await page.wait_for_timeout(2500)
    marked, seen = await page.evaluate(DARK), await page.evaluate(PAINT)
    await ctx.close()
    return marked, seen


async def main() -> int:
    problems = []
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        for size in SIZES:
            for tab in TABS:
                where = f"{size['width']}px {tab}"
                _, light = await paint(browser, size, "light", f"tab={tab}")
                for how, scheme, query in (("system dark", "dark", f"tab={tab}"),
                                           ("?__theme=dark", "light", f"tab={tab}&__theme=dark")):
                    marked, dark = await paint(browser, size, scheme, query)
                    if not marked:
                        problems.append(f"{where} {how}: the page was never marked dark")
                    pairs = [(a, b) for a, b in zip(light, dark) if a[0] == b[0]]
                    off = [(a[0], a[1:], b[1:]) for a, b in pairs if a[1:] != b[1:]]
                    ok = not off and len(pairs) >= 0.9 * max(len(light), 1)
                    print(f"  {where:24} {how:14} {len(pairs):4} texts, {len(off)} drawn differently"
                          f"{'':6}{'ok' if ok else 'WRONG'}")
                    if not ok:
                        problems.append(f"{where} {how}: {len(off)} of {len(pairs)} differ, e.g. {off[:3]}")
        await browser.close()
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for line in problems:
            print(" -", line)
        return 1
    print("\ndark systems see the page everyone else sees")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
