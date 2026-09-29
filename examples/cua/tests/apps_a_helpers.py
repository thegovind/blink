"""Helpers for the apps-a scenario tests and review screenshots (shop, mail, settings, calendar, files).

The candidate rule mirrors SPEC.md: elements matching the actionable selector list that are visible,
enabled, inside the viewport, de-duplicated so the outermost actionable wins, in reading order.
"""

from __future__ import annotations

import contextlib
import functools
import http.server
import json
import os
import threading
from urllib.parse import urlsplit

HERE = os.path.dirname(os.path.abspath(__file__))
APPS_DIR = os.path.normpath(os.path.join(HERE, "..", "apps"))
RUNS_DIR = os.path.normpath(os.path.join(HERE, "..", "runs"))
APPS = ("shop", "mail", "settings", "calendar", "files")
VIEWPORT = {"width": 960, "height": 576}
SCALE = 1.5
MAX_CANDIDATES = 16

CANDIDATES_JS = r"""
(expectedSel) => {
  const SEL = 'a[href], button, input:not([type="hidden"]), select, textarea, [role="button"], [role="link"], '
    + '[role="tab"], [role="checkbox"], [role="radio"], [role="switch"], [role="menuitem"], [role="option"]';
  const W = window.innerWidth, H = window.innerHeight;
  const visible = (el) => {
    const r = el.getBoundingClientRect();
    if (r.width < 2 || r.height < 2) return false;
    if (r.right <= 0 || r.bottom <= 0 || r.left >= W || r.top >= H) return false;
    if (el.checkVisibility && !el.checkVisibility({checkOpacity: true, checkVisibilityCSS: true,
        opacityProperty: true, visibilityProperty: true})) return false;
    return true;
  };
  const enabled = (el) => !el.disabled && el.getAttribute('aria-disabled') !== 'true';
  const ok = Array.from(document.querySelectorAll(SEL)).filter((el) => visible(el) && enabled(el));
  const outer = ok.filter((el) => !ok.some((o) => o !== el && o.contains(el)));
  const name = (el) => {
    const al = (el.getAttribute('aria-label') || '').trim();
    if (al) return al;
    if (el.labels && el.labels.length) { const t = el.labels[0].innerText.trim(); if (t) return t; }
    const t = (el.innerText || '').trim().replace(/\s+/g, ' ');
    if (t && el.tagName !== 'SELECT') return t;
    return (el.getAttribute('placeholder') || el.getAttribute('title') || '').trim();
  };
  const rect = (el) => el.getBoundingClientRect();
  outer.sort((a, b) => {
    const ra = rect(a), rb = rect(b);
    return Math.abs(ra.top - rb.top) > 8 ? ra.top - rb.top : ra.left - rb.left;
  });
  const cands = outer.map((el) => {
    const r = rect(el);
    return {name: name(el), role: el.getAttribute('role') || el.tagName.toLowerCase(), id: el.id || '',
            box: [Math.round(r.x), Math.round(r.y), Math.round(r.width), Math.round(r.height)]};
  });
  let expectedIndex = -1, occluded = false, expectedFound = false;
  if (expectedSel) {
    const matches = document.querySelectorAll(expectedSel);
    expectedFound = matches.length === 1;
    const el = matches[0];
    if (el) {
      expectedIndex = outer.indexOf(el);
      const r = rect(el);
      const hit = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
      occluded = !(hit && (el === hit || el.contains(hit)));
    }
  }
  return {cands, expectedIndex, occluded, expectedFound};
}
"""


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):  # keep test output clean
        pass


@contextlib.contextmanager
def serve(directory: str = APPS_DIR):
    handler = functools.partial(_Quiet, directory=directory)
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()


class Guard:
    """Collects non-loopback requests (aborted) and console/page errors for one browser context."""

    def __init__(self):
        self.blocked: list[str] = []
        self.errors: list[str] = []

    def attach(self, context):
        def route(route):
            host = urlsplit(route.request.url).hostname or ""
            if route.request.url.startswith("data:") or host in ("127.0.0.1", "localhost"):
                route.continue_()
            else:
                self.blocked.append(route.request.url)
                route.abort()

        context.route("**/*", route)
        context.on("page", self.watch)
        return self

    def watch(self, page):
        page.on("console", lambda m: m.type == "error" and self.errors.append(f"console: {m.text}"))
        page.on("pageerror", lambda e: self.errors.append(f"pageerror: {e}"))

    def reset(self):
        out = self.blocked + self.errors
        self.blocked, self.errors = [], []
        return out


def scenario(app: str) -> dict:
    with open(os.path.join(APPS_DIR, app, "scenario.json"), encoding="utf-8") as fh:
        return json.load(fh)


def url(base: str, app: str, **params) -> str:
    q = "&".join(f"{k}={v}" for k, v in params.items() if v is not None)
    return f"{base}/{app}/index.html" + (f"?{q}" if q else "")


def load(page, address: str):
    page.goto(address)
    page.wait_for_function("window.blinkScenario && typeof window.blinkScenario.expected === 'function'")


def candidates(page, selector=None) -> dict:
    return page.evaluate(CANDIDATES_JS, selector)


BOX_JS = """(sel) => { const el = document.querySelector(sel); if (!el) return null;
  const r = el.getBoundingClientRect(); return [r.x, r.y, r.width, r.height]; }"""


def click_center(page, sel: str):
    """Click the centre of the element's box with the mouse, the way the harness acts on a numbered box."""
    box = page.evaluate(BOX_JS, sel)
    if not box:
        raise AssertionError(f"nothing matches {sel!r}")
    x, y, w, h = box
    page.mouse.click(x + w / 2, y + h / 2)


def act(page, exp: dict):
    sel, action, value = exp["selector"], exp["action"], exp.get("value")
    if action == "click":
        click_center(page, sel)
    elif action == "type":
        page.fill(sel, value, timeout=3000)
    elif action == "select":
        page.select_option(sel, label=value, timeout=3000)
    else:
        raise AssertionError(f"unknown action {action!r}")


def follow_oracle(page, limit: int) -> dict:
    """Act on expected() until it reports done. Returns steps, per-step candidate info and problems."""
    steps, trace, problems = 0, [], []
    for _ in range(limit + 1):
        exp = page.evaluate("window.blinkScenario.expected()")
        info = candidates(page, exp.get("selector"))
        names = [c["name"].strip().lower() for c in info["cands"]]
        if len(info["cands"]) > MAX_CANDIDATES:
            problems.append(f"step {steps}: {len(info['cands'])} candidates > {MAX_CANDIDATES}: {[c['name'] for c in info['cands']]}")
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            problems.append(f"step {steps}: duplicate labels {dupes}")
        if any(not n for n in names):
            problems.append(f"step {steps}: empty label among {[c['id'] for c in info['cands']]}")
        trace.append({"step": steps, "expected": exp, "state": page.evaluate("window.blinkScenario.state()"),
                      "n": len(info["cands"])})
        if exp.get("done") or exp.get("action") == "none":
            break
        if not info["expectedFound"]:
            problems.append(f"step {steps}: expected selector {exp.get('selector')!r} does not match exactly one element")
            break
        if info["expectedIndex"] < 0:
            problems.append(f"step {steps}: expected {exp.get('selector')!r} is not a candidate")
            break
        if info["occluded"]:
            problems.append(f"step {steps}: expected {exp.get('selector')!r} is covered by another element")
        act(page, exp)
        steps += 1
    return {"steps": steps, "trace": trace, "problems": problems}


def new_context(browser, theme: str = "light"):
    return browser.new_context(viewport=VIEWPORT, device_scale_factor=SCALE,
                               color_scheme="dark" if theme == "dark" else "light")
