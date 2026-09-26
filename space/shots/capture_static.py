"""Drive the built static page: every tab, every chip, plus an edited input.

    cd space && python -m http.server 8123 --directory static &
    BLINK_URL=http://127.0.0.1:8123/ uv run --with playwright python shots/capture_static.py

Fails with a non-zero exit if the console reports anything, if a chip does not swap in
exactly one pre-rendered answer, or if editing a box does not raise the notice.
"""

import asyncio
import os
import sys

from playwright.async_api import async_playwright

URL = os.environ.get("BLINK_URL", "http://127.0.0.1:8123/")
OUT = os.path.dirname(os.path.abspath(__file__))

TOP = [("playground", "Playground"), ("usecases", "Use cases"), ("results", "Results"),
       ("how", "How it works")]


async def main() -> int:
    problems: list[str] = []
    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        page = await browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=2)
        page.on("console", lambda m: problems.append(f"console.{m.type}: {m.text}")
                if m.type in ("error", "warning") else None)
        page.on("pageerror", lambda e: problems.append(f"pageerror: {e}"))
        page.on("requestfailed", lambda r: problems.append(f"requestfailed: {r.url} {r.failure}"))

        await page.goto(URL, wait_until="load")
        await page.wait_for_selector(".blk-top h1", timeout=20000)
        await page.wait_for_timeout(900)

        async def shot(name: str, full: bool = False):
            path = os.path.join(OUT, f"static-{name}.png")
            await page.screenshot(path=path, full_page=full)
            print("wrote", os.path.relpath(path))

        async def visible_keys(group: str) -> list[str]:
            return await page.evaluate(
                """g => Array.from(document.querySelectorAll('[data-slot="' + g + '"] .blk-out.on'))
                          .map(e => e.dataset.key)""",
                group,
            )

        async def chips(group: str):
            return await page.query_selector_all(f'[data-chips="{group}"] button')

        async def click_chips(group: str, stem: str, save: bool):
            for i, chip in enumerate(await chips(group)):
                label = (await chip.inner_text()).strip()
                await chip.click()
                await page.wait_for_timeout(700)
                keys = await visible_keys(group)
                if len(keys) != 1:
                    problems.append(f"{group} chip {label!r} shows {len(keys)} answers")
                if save:
                    await shot(f"{stem}-{i + 1}")
                print(f"  {group:16s} {label:28s} -> {keys}")

        # --- playground
        await shot("01-playground")
        await click_chips("playground", "01-playground-chip", True)
        state = page.locator("#pg-state")
        await state.fill("A state nobody recorded an answer for.")
        await page.wait_for_timeout(600)
        if await visible_keys("playground") != ["__miss__"]:
            problems.append("editing the playground state did not raise the notice")
        await shot("01-playground-edited")
        await page.get_by_role("button", name="Decide").first.click()
        await page.wait_for_timeout(400)
        if await visible_keys("playground") != ["__miss__"]:
            problems.append("Decide on an edited state did not keep the notice")
        pg_chips = await chips("playground")
        await pg_chips[0].click()
        await page.wait_for_timeout(500)
        first_key = await page.evaluate(
            """() => document.querySelector('[data-chips="playground"] button').dataset.key"""
        )
        if await visible_keys("playground") != [first_key]:
            problems.append("the first chip did not restore its answer")

        # --- use cases
        await page.get_by_role("tab", name="Use cases").click()
        await page.wait_for_timeout(600)
        sub = await page.query_selector_all('.blk-tabs.sub button')
        for i, tab in enumerate(sub):
            title = (await tab.inner_text()).strip()
            await tab.click()
            await page.wait_for_timeout(700)
            group = (await tab.get_attribute("data-target")).replace("pane-", "")
            await shot(f"02-{i + 1}-{group.replace('case-', '')}")
            await click_chips(group, f"02-{i + 1}-{group.replace('case-', '')}-chip", i == 0)
        first_case = await page.query_selector('.blk-tabs.sub button')
        await first_case.click()
        await page.wait_for_timeout(500)
        box = page.locator('#case-support-state')
        await box.fill("Something the recording never saw.")
        await page.wait_for_timeout(600)
        if await visible_keys("case-support") != ["__miss__"]:
            problems.append("editing a use-case state did not raise the notice")
        await shot("02-1-support-edited")

        # --- results and how it works
        await page.get_by_role("tab", name="Results").click()
        await page.wait_for_timeout(1400)
        await shot("03-results")
        await shot("03-results-full", full=True)
        for d in await page.query_selector_all("#pane-results details"):
            await d.evaluate("e => e.open = true")
        await page.wait_for_timeout(400)
        await shot("03-results-open", full=True)

        await page.get_by_role("tab", name="How it works").click()
        await page.wait_for_timeout(900)
        await shot("04-how")
        for d in await page.query_selector_all("#pane-how details"):
            await d.evaluate("e => e.open = true")
        await page.wait_for_timeout(400)
        await shot("04-how-open", full=True)

        await browser.close()

    noisy = [p for p in problems if "favicon" not in p.lower()]
    if noisy:
        print(f"\n{len(noisy)} problem(s):")
        for p in sorted(set(noisy))[:40]:
            print(" -", p)
        return 1
    print("\nno console errors")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
