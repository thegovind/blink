"""Browser check of the live blink Space as an anonymous visitor: tab deep links and one free-form Ask.

  uv run --with playwright python lab/space_ui_check.py [--skip-ask] [--shots /tmp/space-ui]

1. https://thegovind-blink.hf.space/?tab=results opens the Results tab.
2. https://huggingface.co/spaces/thegovind/blink?tab=results opens Results inside the embedded app.
3. Clicking How it works puts ?tab=how-it-works in the app's address.
4. Ask "What's 17 x 23?" in the Playground: the drafted question shows 391 among the options, a result card appears
   and the provenance line names the drafter.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

APP = "https://thegovind-blink.hf.space"
PAGE = "https://huggingface.co/spaces/thegovind/blink"


def selected_tab(frame) -> str:
    frame.wait_for_selector('button[role="tab"][aria-selected="true"]', timeout=120_000)
    return frame.eval_on_selector_all(
        'button[role="tab"][aria-selected="true"]', "els => els.map(e => e.textContent.trim())")


def wait_selected(frame, label: str, timeout_s: float = 60) -> list:
    import time
    t = time.time()
    while time.time() - t < timeout_s:
        tabs = selected_tab(frame)
        if any(x.lower() == label.lower() for x in tabs):
            return tabs
        frame.wait_for_timeout(500)
    return selected_tab(frame)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-ask", action="store_true")
    ap.add_argument("--shots", default="/tmp/space-ui")
    ap.add_argument("--ask-input", default="#ask-input textarea, #ask-input input")
    ap.add_argument("--ask-run", default="#ask-run")
    ap.add_argument("--provenance", default="#ask-provenance")
    a = ap.parse_args()
    shots = Path(a.shots)
    shots.mkdir(parents=True, exist_ok=True)
    fails = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={"width": 1400, "height": 1000})
        page = ctx.new_page()

        page.goto(f"{APP}/?tab=results", wait_until="domcontentloaded", timeout=180_000)
        tabs = wait_selected(page, "Results")
        ok = any(x.lower() == "results" for x in tabs)
        print(f"[1] app ?tab=results -> selected {tabs} {'ok' if ok else 'FAIL'}")
        fails += [] if ok else ["app deep link"]
        page.screenshot(path=str(shots / "app-results.png"))

        tab = page.locator('button[role="tab"]', has_text="How it works").first
        tab.click()
        page.wait_for_timeout(1500)
        url = page.evaluate("location.href")
        ok = "tab=how-it-works" in url
        print(f"[3] click How it works -> {url} {'ok' if ok else 'FAIL'}")
        fails += [] if ok else ["url sync"]

        hub = ctx.new_page()
        hub.goto(f"{PAGE}?tab=results", wait_until="domcontentloaded", timeout=180_000)
        # the Hub page may replace its iframe after load, so find the app frame afresh on every poll
        tabs, frame_url = [], None
        for _ in range(240):
            frame = next((f for f in hub.frames if "hf.space" in (f.url or "")), None)
            if frame is not None:
                try:
                    frame_url = frame.url
                    tabs = frame.eval_on_selector_all(
                        'button[role="tab"][aria-selected="true"]', "els => els.map(e => e.textContent.trim())")
                    if any(x.lower() == "results" for x in tabs):
                        break
                except Exception:  # noqa: BLE001 - detached or not rendered yet
                    pass
            hub.wait_for_timeout(500)
        ok = any(x.lower() == "results" for x in tabs)
        print(f"[2] hub page ?tab=results -> frame {str(frame_url)[:80]} selected {tabs} {'ok' if ok else 'FAIL'}")
        fails += [] if ok else ["hub deep link"]
        hub.screenshot(path=str(shots / "hub-results.png"))

        if not a.skip_ask:
            page.goto(f"{APP}/?tab=ask", wait_until="domcontentloaded", timeout=180_000)
            box = page.locator(a.ask_input).first
            box.wait_for(timeout=120_000)
            box.click()
            page.keyboard.press("ControlOrMeta+a")
            page.keyboard.type("What's 17 x 23?")  # typed: fill() skips the input event Gradio listens for
            page.locator(a.ask_run).first.click()
            prov = page.locator(a.provenance).first
            prov.wait_for(timeout=180_000)
            page.wait_for_function(
                "sel => { const e = document.querySelector(sel); return e && /qwen/i.test(e.textContent); }",
                arg=a.provenance, timeout=180_000)
            text = prov.inner_text()
            body = page.inner_text("body")
            ok = "391" in body and "qwen" in text.lower()
            print(f"[4] ask 17 x 23 -> provenance {text.strip()[:120]!r}; 391 on page: {'391' in body} "
                  f"{'ok' if ok else 'FAIL'}")
            fails += [] if ok else ["ask"]
            page.screenshot(path=str(shots / "ask-17x23.png"), full_page=True)
        browser.close()
    if fails:
        sys.exit(f"UI check failed: {fails}")
    print("UI_CHECK_OK")


if __name__ == "__main__":
    main()
