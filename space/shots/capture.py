"""Screenshot the interface at 1440x900 and report console errors.

    cd space && BLINK_MOCK=1 BLINK_PORT=7871 uv run --with "gradio==6.28.0" python app.py &
    BLINK_URL=http://127.0.0.1:7871/ uv run --with playwright python shots/capture.py

BLINK_SHOTS=replay captures the two replay-only states instead: a preset that has a
recorded answer, and an edited state that does not.
"""

import asyncio
import os
import sys

from playwright.async_api import async_playwright

URL = os.environ.get("BLINK_URL", "http://127.0.0.1:7860/")
MODE = os.environ.get("BLINK_SHOTS", "all")
OUT = os.path.dirname(os.path.abspath(__file__))

TOP = ["Playground", "Use cases", "Results", "How it works"]
CASES = [
    "Support triage",
    "Email & phishing",
    "Policy decisions",
    "RAG passage filter",
    "Moderation & safety",
    "Model routing",
    "Next click",
]


async def main() -> int:
    problems: list[str] = []
    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        page = await browser.new_page(
            viewport={"width": 1440, "height": 900}, device_scale_factor=2
        )
        page.on(
            "console",
            lambda m: problems.append(f"console.{m.type}: {m.text}")
            if m.type in ("error", "warning")
            else None,
        )
        page.on("pageerror", lambda e: problems.append(f"pageerror: {e}"))
        page.on(
            "requestfailed",
            lambda r: problems.append(f"requestfailed: {r.url} {r.failure}"),
        )

        await page.goto(URL, wait_until="load")
        await page.wait_for_selector(".blk-top h1", timeout=30000)
        await page.wait_for_timeout(2500)

        async def tab(name: str, nth: int = 0):
            await page.get_by_role("tab", name=name, exact=True).nth(nth).click()
            await page.wait_for_timeout(1600)

        async def shot(name: str, full: bool = False):
            path = os.path.join(OUT, f"{name}.png")
            await page.screenshot(path=path, full_page=full)
            print("wrote", os.path.relpath(path))

        if MODE == "replay":
            await page.get_by_role("button", name="Support ticket", exact=True).click()
            await page.wait_for_selector(".blk-answers", timeout=20000)
            await page.mouse.wheel(0, -4000)
            await page.wait_for_timeout(1200)
            await shot("05-playground-replay-preset")

            state = page.locator("textarea").first
            await state.fill(
                "Our nightly export job has failed four nights running and nobody has "
                "replied to the last two tickets."
            )
            await page.get_by_role("button", name="Decide", exact=True).click()
            await page.wait_for_selector(".blk-notice", timeout=20000)
            await page.mouse.wheel(0, -4000)
            await page.wait_for_timeout(1200)
            await shot("06-playground-replay-edited")

            await tab("Use cases")
            await tab("Support triage")
            await page.wait_for_selector(".blk-verdict", timeout=20000)
            await page.mouse.wheel(0, -4000)
            await page.wait_for_timeout(1200)
            await shot("07-use-case-replay")

            await page.wait_for_timeout(800)
            await browser.close()
            return report(problems)

        await shot("01-playground")

        await tab("Use cases")
        for i, case in enumerate(CASES):
            await tab(case)
            await shot(f"02-{i + 1}-{case.split()[0].lower().strip('&')}")

        await tab("Results")
        await shot("03-results")
        await shot("03-results-full", full=True)
        for d in await page.query_selector_all("details.blk-d"):
            await d.evaluate("e => e.open = true")
        await page.wait_for_timeout(500)
        await shot("03-results-open", full=True)

        await tab("How it works")
        await shot("04-how")
        for d in await page.query_selector_all("details.blk-d"):
            await d.evaluate("e => e.open = true")
        await page.wait_for_timeout(500)
        await shot("04-how-open", full=True)

        await tab("Playground")
        for d in await page.query_selector_all("details.blk-more"):
            await d.evaluate("e => e.open = true")
        await page.wait_for_timeout(500)
        await shot("01-playground-raw", full=True)

        await browser.close()

    return report(problems)


def report(problems: list[str]) -> int:
    benign = ("favicon", "queue/data")  # the live event stream aborts when the browser closes
    noisy = [p for p in problems if not any(b in p.lower() for b in benign)]
    if noisy:
        print(f"\n{len(noisy)} console/page problem(s):")
        for p in sorted(set(noisy))[:40]:
            print(" -", p)
        return 1
    print("\nno console errors")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
