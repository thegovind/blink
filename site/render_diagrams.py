#!/usr/bin/env python3
"""Draw every Mermaid block in the site's pages as SVG, ahead of time.

    uv run --no-project --with-requirements site/requirements.txt --with playwright \
        python site/render_diagrams.py

Mermaid, at the version config.json pins, runs in headless Chromium on a page styled by
static/site.css with Geist loaded, so each label is measured in the face it is shown in.
Each drawing is saved as diagrams/<key>.svg; the key covers the block's source, Mermaid's
version and settings, and site.css, so build.py only uses a drawing made for what it shows.
Drawings no page uses any more are removed.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import re
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import build  # noqa: E402

IP_LIKE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|[A-Za-z]")


def blocks(root: Path) -> list[str]:
    site = build.Site(root, HERE / ".unused")
    found = []
    for page in build.PAGES:
        for tok in site.md.parse(site.page_text(page), build.new_env()):
            if tok.type == "fence" and tok.info.strip().split()[:1] == ["mermaid"]:
                found.append(tok.content)
    return found


def mermaid_js() -> str:
    pin = build.CONFIG["mermaid"]["js"]
    with urllib.request.urlopen(pin["url"], timeout=60) as resp:
        data = resp.read()
    algo, _, want = pin["integrity"].partition("-")
    got = base64.b64encode(hashlib.new(algo, data).digest()).decode()
    if got != want:
        raise SystemExit(f"{pin['url']} does not match its pinned integrity")
    return data.decode("utf-8")


def tidy_path(match: re.Match) -> str:
    """Path data with a space between numbers: '.5.25' reads as two numbers, not an address."""
    return f'{match.group(1)}="{" ".join(NUMBER.findall(match.group(2)))}"'


def finish(svg: str) -> str:
    svg = re.sub(r'\b(d)="([^"]*)"', tidy_path, svg)
    # node boxes take the Space's nested radius
    svg = re.sub(r'<rect class="basic label-container"(?![^>]*\brx=)', '<rect class="basic label-container" rx="10" ry="10"', svg)
    head = re.match(r"<svg\b[^>]*>", svg)
    if not head:
        raise ValueError("not an svg")
    tag = head.group(0)
    box = re.search(r'viewBox="([-\d.]+)[ ,]+([-\d.]+)[ ,]+([\d.]+)[ ,]+([\d.]+)"', tag)
    if not box:
        raise ValueError("svg without a viewBox")
    w, h = float(box.group(3)), float(box.group(4))
    new = re.sub(r'\swidth="[^"]*"', "", tag)
    new = re.sub(r'\sheight="[^"]*"', "", new)
    new = re.sub(r'\sstyle="max-width:[^"]*"', "", new)
    new = new.replace("<svg", f'<svg width="{w:g}" height="{h:g}"', 1)
    svg = new + svg[len(tag):]
    if IP_LIKE.search(svg):
        raise ValueError(f"svg still holds an address-like number: {IP_LIKE.search(svg).group(0)}")
    return svg


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="{fonts}"><style>{css}</style></head>
<body class="docs-page"><div class="docs"><main class="doc"><article class="prose">
<figure class="diagram" id="host"></figure></article></main></div></body></html>"""

RENDER = """async ([id, code]) => {
  try {
    await mermaid.parse(code);
    const host = document.getElementById('host');
    const out = await mermaid.render(id, code, host);
    host.replaceChildren();
    return {ok: true, svg: out.svg};
  } catch (e) {
    return {ok: false, err: String((e && e.message) || e)};
  }
}"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=build.DEFAULT_ROOT)
    ap.add_argument("--out", type=Path, default=HERE / "diagrams")
    ap.add_argument("--force", action="store_true", help="redraw drawings that already exist")
    a = ap.parse_args(argv)

    env_hash = build.diagram_env_hash(HERE)
    codes = blocks(a.root)
    keys = {build.diagram_key(code, env_hash): code for code in codes}
    a.out.mkdir(parents=True, exist_ok=True)
    todo = {k: c for k, c in keys.items() if a.force or not (a.out / f"{k}.svg").exists()}
    failed = []
    if todo:
        from playwright.sync_api import sync_playwright

        css = (HERE / "static" / "site.css").read_text(encoding="utf-8")
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1440, "height": 900})
            page.set_content(PAGE.format(fonts=build.CONFIG["fonts_css"], css=css), wait_until="networkidle")
            ready = page.evaluate("""async () => {
                await Promise.all(['400 14px Geist', '500 14px Geist', '600 14px Geist'].map((f) => document.fonts.load(f)));
                await document.fonts.ready;
                return document.fonts.check('14px Geist');
            }""")
            if not ready:
                raise SystemExit("Geist did not load; the drawings would be measured in another face")
            page.add_script_tag(content=mermaid_js())
            page.evaluate("(cfg) => mermaid.initialize(cfg)", build.CONFIG["mermaid"]["config"])
            for key, code in todo.items():
                res = page.evaluate(RENDER, [f"d-{key}", code])
                if not res["ok"]:
                    failed.append(f"{key}: {res['err'][:300]}")
                    continue
                try:
                    (a.out / f"{key}.svg").write_text(finish(res["svg"]) + "\n", encoding="utf-8")
                except ValueError as exc:
                    failed.append(f"{key}: {exc}")
            browser.close()
    stale = [p for p in a.out.glob("*.svg") if p.stem not in keys]
    for p in stale:
        p.unlink()
    print(f"{len(codes)} mermaid blocks, {len(keys)} distinct, {len(todo) - len(failed)} drawn, "
          f"{len(keys) - len(todo)} kept, {len(stale)} removed, {len(failed)} failed")
    for line in failed:
        print("  ", line)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
