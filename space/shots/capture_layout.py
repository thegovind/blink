"""Screenshot the interface at the widths the owner looks at, and check the grid.

    BLINK_MOCK=1 BLINK_MODELS=a,b BLINK_PORT=7896 uv run --with gradio==6.28.0 python app.py &
    BLINK_URL=http://127.0.0.1:7896/ uv run --with playwright python shots/capture_layout.py
"""

import asyncio
import json
import os
import sys

from playwright.async_api import async_playwright

URL = os.environ.get("BLINK_URL", "http://127.0.0.1:7896/")
OUT = os.path.dirname(os.path.abspath(__file__))
SIZES = [(1456, 900), (1920, 1080), (1280, 800), (390, 844)]
TABS = [("Home", "home"), ("Playground", "play"), ("Ask", "ask"),
        ("Use cases", "case"), ("Results", "results")]


async def open_tab(page, label: str) -> None:
    """Narrow viewports move the tab bar behind a "More tabs" menu."""
    strip = page.get_by_role("tab", name=label, exact=True)
    if await strip.count():
        await strip.first.click()
        return
    await page.locator('[aria-label="More tabs"]').first.click()
    await page.wait_for_timeout(300)
    await page.get_by_role("button", name=label, exact=True).last.click()


async def main() -> int:
    problems: list[str] = []
    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        for w, hgt in SIZES:
            page = await browser.new_page(viewport={"width": w, "height": hgt}, device_scale_factor=2)
            page.on("pageerror", lambda e: problems.append(f"pageerror: {e}"))
            page.on("console", lambda m: problems.append(f"console.error: {m.text}")
                    if m.type == "error" else None)
            await page.goto(URL, wait_until="load")
            await page.wait_for_selector(".blk-top h1", timeout=30000)
            await page.wait_for_timeout(1800)
            for i, (label, stem) in enumerate(TABS):
                if i:  # the page opens on the first tab; clicking it again only churns
                    await open_tab(page, label)
                await page.wait_for_timeout(1500 if stem in ("results", "home") else 900)
                path = os.path.join(OUT, f"layout-{w}-{stem}.png")
                # a full-page capture replays CSS animations in Chromium; take the settled frame
                await page.screenshot(path=path, full_page=(stem in ("results", "home")),
                                      animations="disabled")
                print("wrote", os.path.relpath(path))
            await open_tab(page, "Playground")
            await page.wait_for_timeout(1000)
            grid = await page.evaluate("""() => {
                const x = s => { const el = Array.from(document.querySelectorAll(s))
                        .find((e) => e.offsetParent !== null || e.tagName === 'svg');
                    if (!el) return null; const b = el.getBoundingClientRect();
                    return [Math.round(b.x), Math.round(b.right)]; };
                const c = document.querySelector('.gradio-container').getBoundingClientRect();
                return {win: window.innerWidth, container: [Math.round(c.x), Math.round(c.right)],
                        top: x('.blk-top'), rule: x('.blk-rule'), note: x('.blk-note'),
                        seg: x('.blk-seg .wrap'), chips: x('.blk-chips'), demo: x('.blk-demo'),
                        state: x('.blk-state textarea'), decide: x('.blk-decide'),
                        sum: x('.blk-qsum'), acc: x('.blk-acc'), qrow: x('.blk-qblock')};
            }""")
            print(f"  grid@{w}: {json.dumps(grid)}")
            lefts = {k: v[0] for k, v in grid.items() if isinstance(v, list) and k != "container"}
            if len(set(lefts.values())) > 1:
                problems.append(f"{w}px ragged left edge: {lefts}")
            # the home page shares the same left edge, and its chips stay on one line
            await open_tab(page, "Home")
            await page.wait_for_timeout(900)
            home = await page.evaluate("""() => {
                const x = s => { const el = Array.from(document.querySelectorAll(s))
                        .find((e) => e.getBoundingClientRect().width > 0);
                    if (!el) return null; const b = el.getBoundingClientRect();
                    return [Math.round(b.x), Math.round(b.right)]; };
                return {lede: x('.blk-home-lede'), eyebrow: x('.blk-eyebrow'),
                        cards: x('.blk-cards'), chart: x('.blk-figure')};
            }""")
            print(f"  home@{w}: {json.dumps(home)}")
            hl = {k: v[0] for k, v in home.items() if isinstance(v, list)}
            if len(set(hl.values())) > 1:
                problems.append(f"{w}px home ragged left edge: {hl}")
            # a figure keeps its own width on a phone and scrolls inside its pane
            over = {k: v for k, v in home.items()
                    if k != "chart" and isinstance(v, list) and v[1] > w}
            if over:
                problems.append(f"{w}px home runs past the window: {over}")
            wide = await page.evaluate(
                "() => document.documentElement.scrollWidth <= window.innerWidth + 1")
            if not wide:
                problems.append(f"{w}px home scrolls sideways")
            # a value you have to pan to is a value the page did not show
            values = await page.evaluate("""() => {
                const out = [];
                for (const el of document.querySelectorAll(
                        '.blk-half text, .blk-half td, .blk-half th')) {
                    const b = el.getBoundingClientRect();
                    if (b.width === 0) continue;
                    out.push([el.textContent.trim(), Math.round(b.right)]);
                }
                return out;
            }""")
            for want in ("78.3", "73.5", "68.5", "63.44", "59.51",
                         "76.6", "80.0", "0.096", "0.119"):
                hits = [right for text, right in values if want in text]
                if not hits:
                    problems.append(f"{w}px home never shows {want}")
                elif max(hits) > w + 1:
                    problems.append(f"{w}px home hides {want} at x={max(hits)}")
            print(f"  values@{w}: {len(values)} on screen")
            await open_tab(page, "Playground")
            await page.wait_for_timeout(800)
            tall = await page.evaluate("""() => Array.from(
                document.querySelectorAll('.blk-chips button'))
                .filter((b) => b.offsetParent !== null)
                .map((b) => Math.round(b.getBoundingClientRect().height))""")
            print(f"  chips@{w}: {tall}")
            if tall and max(tall) - min(tall) > 2:
                problems.append(f"{w}px a preset wrapped: {tall}")
            await page.close()
        await browser.close()

    noisy = [p for p in problems if "favicon" not in p.lower()]
    if noisy:
        print(f"\n{len(noisy)} problem(s):")
        for p in sorted(set(noisy))[:20]:
            print(" -", p)
        return 1
    print("\nno console errors, one left edge everywhere")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
