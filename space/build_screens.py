"""Render the Screen click presets from the local browser-test fixture pages.

    uv run --no-project --python 3.12 --with playwright --with pillow python space/build_screens.py

The fixture pages and their tasks are the ones the browser-agent loop was checked on, kept
beside this checkout in ../runs/cua-browser (--fixtures to point elsewhere). Each preset serves
that folder on a loopback port, drives one page to a state with Playwright, reads its candidate
elements' boxes from the DOM, burns numbered boxes into the screenshot with screens.mark and
writes space/screens/<key>.png. space/screens/screens.json records the boxes, the task, what
the page expects and the digest of every file used. No other page is ever loaded.
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import hashlib
import http.server
import io
import json
import os
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import screens  # noqa: E402

DEFAULT_FIXTURES = os.path.join(os.path.dirname(HERE), "..", "runs", "cua-browser", "fixtures")
VIEWPORT = {"width": 960, "height": 576}
# 1440 x 864: sharp on a phone, under blink's 2,088,960-pixel default, and both sides a multiple of the
# vision processor's 32-pixel patch, so blink reads these pixels exactly and never resamples the marks
SCALE = 1.5
PAD = 6  # CSS pixels between an element and its box
MAX_PNG_BYTES = 300 * 1024

# The goals are the browser-agent test's own (run_fixtures.py TASKS), word for word.
CATALOG = "Search the Maps category, then open Atlas Field Guide from the results."
DIRECTORY = "Filter workshops to Remote and Morning only, then open Blue Workshop."
LOOKUP = "Select Transit, submit Find articles, then open Blue Line timetable."

# Each preset: the page, the steps that bring it to the state shown, and its candidates in
# reading order as (role, selector). `expect` is what a correct agent would answer; only the
# mock engine and the tests read it.
PRESETS = (
    {
        "key": "catalog", "label": "Open the result", "page": "catalog.html", "task": CATALOG,
        "steps": (("select", "#category", "maps"), ("click", "#search button")),
        "targets": (("select", "#category"), ("button", "#search button"), ("link", "#open-atlas")),
        "expect": {"element": "3", "done": "no", "risky": "no"},
    },
    {
        "key": "directory", "label": "Filter first", "page": "directory.html", "task": DIRECTORY,
        "steps": (("select", "#format", "remote"),),
        "targets": (("select", "#format"), ("checkbox", "label:has(#morning)"),
                    ("link", '#list a[data-name="Blue Workshop"]'), ("link", '#list a[data-name="Green Workshop"]')),
        "expect": {"element": "2", "done": "no", "risky": "no"},
    },
    {
        "key": "lookup", "label": "Pick a topic", "page": "lookup.html", "task": LOOKUP,
        "steps": (),
        "targets": (("radio", 'label:has(input[value="parks"])'), ("radio", 'label:has(input[value="transit"])'),
                    ("button", "#search button")),
        "expect": {"element": "2", "done": "no", "risky": "no"},
    },
    {
        # the catalog task one click later: nothing on the page is left to act on
        "key": "done", "label": "Already done", "page": "catalog.html", "task": CATALOG,
        "steps": (("select", "#category", "maps"), ("click", "#search button"), ("click", "#open-atlas")),
        "targets": (("heading", "header"), ("article", "#verification")),
        "expect": {"done": "yes", "risky": "no"},
    },
)

NAME = """(el) => {
  const text = (e) => (e ? e.textContent : '').replace(/\\s+/g, ' ').trim();
  if (el.tagName === 'SELECT') {
    const picked = el.options[el.selectedIndex];
    return `${el.getAttribute('aria-label') || text(el.labels && el.labels[0])}: ${text(picked)}`;
  }
  const head = el.querySelector('h1, h2');
  return head ? text(head) : text(el);
}"""


def digest(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_args):
        pass


def serve(folder: str):
    handler = functools.partial(_Quiet, directory=folder)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


async def capture(fixtures: str) -> list[dict]:
    from playwright.async_api import async_playwright

    server = serve(fixtures)
    base = f"http://127.0.0.1:{server.server_address[1]}/"
    shots = []
    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            ctx = await browser.new_context(viewport=VIEWPORT, device_scale_factor=SCALE,
                                            reduced_motion="reduce", color_scheme="light")
            # the pages are served from a folder on this machine; nothing else may load
            async def local_only(route):
                if route.request.url.startswith(base):
                    await route.continue_()
                else:
                    await route.abort()

            await ctx.route("**/*", local_only)
            page = await ctx.new_page()
            for p in PRESETS:
                await page.goto(base + p["page"], wait_until="load")
                for step in p["steps"]:
                    if step[0] == "select":
                        await page.select_option(step[1], step[2])
                    else:
                        await page.click(step[1])
                await page.evaluate("document.fonts.ready")
                await page.wait_for_timeout(150)
                boxes, meta = [], []
                for role, selector in p["targets"]:
                    el = page.locator(selector)
                    if await el.count() != 1 or not await el.is_visible():
                        raise SystemExit(f"{p['key']}: {selector} is not one visible element")
                    r = await el.bounding_box()
                    x1, y1 = max(0.0, r["x"] - PAD), max(0.0, r["y"] - PAD)
                    x2 = min(VIEWPORT["width"], r["x"] + r["width"] + PAD)
                    y2 = min(VIEWPORT["height"], r["y"] + r["height"] + PAD)
                    boxes.append(tuple(round(v * SCALE) for v in (x1, y1, x2, y2)))
                    meta.append({"role": role, "name": await el.evaluate(NAME)})
                png = await page.screenshot(type="png", animations="disabled")
                shots.append({**p, "png": png, "boxes": boxes, "meta": meta})
            await browser.close()
    finally:
        server.shutdown()
    return shots


def build(fixtures: str, out: str = screens.FOLDER) -> dict:
    from PIL import Image

    fixtures = os.path.abspath(fixtures)
    pages = sorted({p["page"] for p in PRESETS}) + ["style.css"]
    for name in pages:
        if not os.path.isfile(os.path.join(fixtures, name)):
            raise SystemExit(f"{name} is not in {fixtures}")
    os.makedirs(out, exist_ok=True)
    entries = []
    for shot in asyncio.run(capture(fixtures)):
        with Image.open(io.BytesIO(shot["png"])) as raw:
            if raw.width % screens.FACTOR or raw.height % screens.FACTOR:
                raise SystemExit(f"{shot['key']}: {raw.size} is not a multiple of {screens.FACTOR} pixels")
            marked, placed = screens.mark(raw.convert("RGB"), shot["boxes"], meta=shot["meta"])
        buf = io.BytesIO()
        marked.save(buf, format="PNG", optimize=True)
        png = buf.getvalue()
        if len(png) > MAX_PNG_BYTES:
            raise SystemExit(f"{shot['key']}.png is {len(png):,} bytes, over {MAX_PNG_BYTES:,}")
        name = f"{shot['key']}.png"
        with open(os.path.join(out, name), "wb") as fh:
            fh.write(png)
        entries.append({
            "key": shot["key"], "label": shot["label"], "page": shot["page"], "task": shot["task"],
            "image": name, "size": list(marked.size), "bytes": len(png), "sha256": hashlib.sha256(png).hexdigest(),
            "boxes": [{"n": b.n, "box": list(b.box), "tag": list(b.tag), "at": b.at, "role": b.role, "name": b.name}
                      for b in placed],
            "expect": shot["expect"],
        })
    meta = {
        "about": "Screen click presets: local browser-test fixture pages, marked by build_screens.py.",
        "viewport": [VIEWPORT["width"], VIEWPORT["height"]], "scale": SCALE, "pad_css_px": PAD,
        "style": screens.style(entries[0]["size"][0]),
        "fixtures": {name: digest(os.path.join(fixtures, name)) for name in pages},
        "presets": entries,
    }
    with open(os.path.join(out, "screens.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixtures", default=DEFAULT_FIXTURES, help="the browser-test fixture folder")
    ap.add_argument("--out", default=screens.FOLDER)
    a = ap.parse_args(argv)
    meta = build(a.fixtures, a.out)
    for p in meta["presets"]:
        size = f"{p['size'][0]}x{p['size'][1]}"
        print(f"{p['image']:14s} {size} {p['bytes']:7,d} B  {len(p['boxes'])} boxes  {p['label']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
