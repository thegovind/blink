"""CPU tests for the B scenario apps (travel, careers, phone, desktop, game); contract in ../SPEC.md.

    uv run --no-project --python 3.12 --with playwright --with pillow --with pytest \
        python -m pytest examples/cua/tests/test_apps_b.py -q

The same file renders review screenshots into examples/cua/runs/app-shots-b/ (gitignored):

    uv run --no-project --python 3.12 --with playwright --with pillow --with pytest \
        python examples/cua/tests/test_apps_b.py shots [app ...]

Apps are served from examples/cua/apps/ on loopback; any other request fails the test, as does a
console error. The DOM candidate rule below is the SPEC's: visible, enabled elements from the
selector list inside the viewport, outermost actionable wins, reading order.
"""

from __future__ import annotations

import functools
import http.server
import io
import json
import os
import random
import sys
import threading
import urllib.parse

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
APPS_DIR = os.path.normpath(os.path.join(HERE, "..", "apps"))
SHOTS_DIR = os.path.normpath(os.path.join(HERE, "..", "runs", "app-shots-b"))
APPS = ("travel", "careers", "phone", "desktop", "game")
DOM_APPS = ("travel", "careers", "phone")
SEEDS = (1, 2, 3, 4, 5)
INTERRUPT_SEEDS = (1, 2)
MAX_CANDIDATES = 16
MAX_NAME = 40
SETTLE_MS = 160
GAME_WAVES = 20

CANDIDATES_JS = r"""
() => {
  const SEL = 'a[href],button,input:not([type="hidden"]),select,textarea,[role="button"],[role="link"],' +
    '[role="tab"],[role="checkbox"],[role="radio"],[role="switch"],[role="menuitem"],[role="option"]';
  const W = window.innerWidth, H = window.innerHeight;
  const clean = (s) => (s || '').replace(/\s+/g, ' ').trim();
  const name = (el) => {
    let n = clean(el.getAttribute('aria-label'));
    if (n) return n;
    const by = el.getAttribute('aria-labelledby');
    if (by) {
      n = clean(by.split(/\s+/).map((id) => (document.getElementById(id) || {}).innerText || '').join(' '));
      if (n) return n;
    }
    if (el.labels && el.labels.length) {
      n = clean([...el.labels].map((l) => l.innerText).join(' '));
      if (n) return n;
    }
    if (el.matches('input,textarea')) {
      n = clean(el.getAttribute('placeholder'));
      if (n) return n;
    }
    if (!el.matches('select')) {
      n = clean(el.innerText);
      if (n) return n;
    }
    n = clean(el.getAttribute('title'));
    if (n) return n;
    return el.matches('input') ? clean(el.value) : '';
  };
  const vis = [];
  for (const el of document.querySelectorAll(SEL)) {
    if (el.disabled) continue;
    const r = el.getBoundingClientRect();
    if (r.width < 2 || r.height < 2) continue;
    if (r.right <= 0 || r.bottom <= 0 || r.left >= W || r.top >= H) continue;
    if (!el.checkVisibility({opacityProperty: true, visibilityProperty: true})) continue;
    vis.push(el);
  }
  const outer = vis.filter((el) => !vis.some((o) => o !== el && o.contains(el)));
  const items = outer.map((el) => {
    const r = el.getBoundingClientRect();
    return {el, box: [r.left, r.top, r.right, r.bottom]};
  });
  items.sort((a, b) => (Math.abs(a.box[1] - b.box[1]) > 6 ? a.box[1] - b.box[1] : a.box[0] - b.box[0]));
  window.__btCands = items.map((i) => i.el);
  return items.map((i) => ({name: name(i.el), tag: i.el.tagName.toLowerCase(),
                            role: i.el.getAttribute('role') || '', box: i.box.map(Math.round)}));
}
"""

TARGET_INDEX_JS = """(sel) => {
  const m = document.querySelectorAll(sel);
  if (m.length !== 1) return -1 - m.length;
  return (window.__btCands || []).indexOf(m[0]);
}"""


@functools.lru_cache(maxsize=None)
def scenario(app: str) -> dict:
    with open(os.path.join(APPS_DIR, app, "scenario.json"), encoding="utf-8") as fh:
        return json.load(fh)


def task_meta(app: str, task: int) -> dict:
    return next(t for t in scenario(app)["tasks"] if t["index"] == task)


def tasks_of(app: str) -> list[int]:
    return [t["index"] for t in scenario(app)["tasks"]]


# --- static server and browser ---------------------------------------------------------------


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):  # keep pytest output clean
        pass

    def do_GET(self):
        if self.path.split("?")[0].endswith("favicon.ico"):
            self.send_response(204)
            self.end_headers()
            return
        super().do_GET()


def start_server():
    handler = functools.partial(_Quiet, directory=APPS_DIR)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


class Session:
    """One app page in its own browser context, with request and console guards."""

    def __init__(self, browser, base: str, app: str, **params):
        sc = scenario(app)
        self.app = app
        self.width, self.height = sc["viewport"]["width"], sc["viewport"]["height"]
        self.errors: list[str] = []
        self.blocked: list[str] = []
        self.ctx = browser.new_context(viewport={"width": self.width, "height": self.height},
                                       device_scale_factor=sc["scale"])
        self.ctx.route("**/*", self._route)
        self.page = self.ctx.new_page()
        self.page.on("console", lambda m: m.type == "error" and self.errors.append(m.text))
        self.page.on("pageerror", lambda e: self.errors.append(f"pageerror: {e}"))
        query = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        self.url = f"{base}/{sc['path']}" + (f"?{query}" if query else "")
        self.page.goto(self.url)
        self.page.wait_for_function("window.blinkScenario && typeof window.blinkScenario.expected === 'function'")
        self.page.wait_for_timeout(60)

    def _route(self, route):
        host = urllib.parse.urlsplit(route.request.url).hostname or ""
        if host in ("127.0.0.1", "localhost"):
            route.continue_()
        else:
            self.blocked.append(route.request.url)
            route.abort()

    def js(self, expr: str, arg=None):
        return self.page.evaluate(expr, arg) if arg is not None else self.page.evaluate(expr)

    def expected(self) -> dict:
        return self.js("blinkScenario.expected()")

    def check(self) -> dict:
        c = self.js("blinkScenario.check()")
        assert isinstance(c, dict) and {"done", "success"} <= set(c), f"{self.url}: check() returned {c!r}"
        return c

    def state(self) -> str:
        return self.js("blinkScenario.state()")

    def candidates(self) -> list[dict]:
        return self.js(CANDIDATES_JS)

    def targets(self) -> list[dict]:
        return self.js("blinkScenario.targets()")

    def close(self):
        errors, blocked = list(self.errors), list(self.blocked)
        self.ctx.close()
        assert not blocked, f"{self.url}: non-loopback requests {blocked}"
        assert not errors, f"{self.url}: console errors {errors}"


@pytest.fixture(scope="session")
def base():
    srv = start_server()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


@pytest.fixture(scope="session")
def browser():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        b = pw.chromium.launch()
        yield b
        b.close()


@pytest.fixture
def open_app(browser, base):
    sessions = []

    def _open(app, **params):
        s = Session(browser, base, app, **params)
        sessions.append(s)
        return s

    yield _open
    for s in sessions:
        s.close()


# --- screen assertions -----------------------------------------------------------------------


def assert_dom_screen(s: Session, where: str = "") -> list[dict]:
    cands = s.candidates()
    tag = f"{s.url} {where}".strip()
    names = [c["name"] for c in cands]
    assert len(cands) <= MAX_CANDIDATES, f"{tag}: {len(cands)} candidates: {names}"
    assert all(names), f"{tag}: unnamed candidate in {cands}"
    long = [n for n in names if len(n) > MAX_NAME]
    assert not long, f"{tag}: labels too long {long}"
    lower = [n.lower() for n in names]
    dupes = sorted({n for n in lower if lower.count(n) > 1})
    assert not dupes, f"{tag}: duplicate labels {dupes}"
    over = s.js("[document.scrollingElement.scrollWidth - innerWidth, document.scrollingElement.scrollHeight - innerHeight]")
    assert over[0] <= 1 and over[1] <= 1, f"{tag}: page scrolls by {over}"
    return cands


def assert_targets(s: Session, where: str = "") -> list[dict]:
    tg = s.targets()
    tag = f"{s.url} {where}".strip()
    assert isinstance(tg, list) and len(tg) <= MAX_CANDIDATES, f"{tag}: {len(tg)} targets"
    ids = [t["id"] for t in tg]
    assert len(set(ids)) == len(ids), f"{tag}: duplicate target ids {ids}"
    for t in tg:
        r = t["rect"]
        assert isinstance(t.get("name"), str) and isinstance(t.get("role"), str), f"{tag}: {t}"
        assert r["w"] > 0 and r["h"] > 0, f"{tag}: empty rect {t}"
        assert r["x"] >= 0 and r["y"] >= 0 and r["x"] + r["w"] <= s.width + 0.5 and r["y"] + r["h"] <= s.height + 0.5, \
            f"{tag}: rect out of bounds {t}"
    assert s.candidates() == [], f"{tag}: canvas app has DOM controls"
    return tg


def is_final(exp: dict) -> bool:
    return exp.get("action") == "none" or bool(exp.get("done"))


def act_dom(s: Session, exp: dict):
    sel, action = exp["selector"], exp["action"]
    s.candidates()
    idx = s.js(TARGET_INDEX_JS, sel)
    assert idx >= 0, f"{s.url}: oracle target {sel!r} is not one visible candidate ({idx})"
    loc = s.page.locator(sel)
    if action == "click":
        loc.click()
    elif action == "type":
        assert exp["value"] in s.js("blinkScenario.values"), f"{s.url}: typed value {exp['value']!r} not in values"
        loc.fill(exp["value"])
    elif action == "select":
        labels = loc.evaluate("el => [...el.options].map(o => o.label)")
        assert exp["value"] in labels, f"{s.url}: {exp['value']!r} not an option of {sel}"
        loc.select_option(label=exp["value"])
    else:
        raise AssertionError(f"{s.url}: unknown action {exp!r}")
    s.page.wait_for_timeout(SETTLE_MS)


def click_target(s: Session, target_id: str):
    tg = s.targets()
    t = next((t for t in tg if t["id"] == target_id), None)
    assert t is not None, f"{s.url}: target {target_id!r} not in {[x['id'] for x in tg]}"
    r = t["rect"]
    s.page.mouse.click(r["x"] + r["w"] / 2, r["y"] + r["h"] / 2)
    s.page.wait_for_timeout(SETTLE_MS)


def follow_oracle(s: Session, limit: int) -> tuple[int, int, dict]:
    """Act on expected() until it says done or none; returns (steps, risky steps, final expected)."""
    canvas = scenario(s.app)["kind"] != "dom"
    steps = risky = 0
    while True:
        if canvas:
            assert_targets(s, f"step {steps}")
        else:
            assert_dom_screen(s, f"step {steps}")
        exp = s.expected()
        s.check()
        if is_final(exp):
            return steps, risky, exp
        assert steps < limit, f"{s.url}: oracle did not finish within {limit} steps (last {exp})"
        risky += bool(exp.get("risky"))
        if canvas:
            assert exp.get("action") == "click", f"{s.url}: {exp}"
            click_target(s, exp["target"])
        else:
            act_dom(s, exp)
        steps += 1


# --- registry entries --------------------------------------------------------------------------


@pytest.mark.parametrize("app", APPS)
def test_scenario_json(app):
    sys.path.insert(0, APPS_DIR)
    import build_registry

    sc = scenario(app)
    for key in build_registry.REQUIRED:
        assert key in sc, f"{app}: missing {key}"
    assert sc["id"] == app and sc["path"] == f"{app}/index.html"
    assert 3 <= len(sc["title"].split()) <= 5, sc["title"]
    assert sc["kind"] == {"desktop": "canvas", "game": "game"}.get(app, "dom")
    assert sc["scale"] == 1.5
    assert sc["viewport"] == ({"width": 390, "height": 844} if app == "phone" else {"width": 960, "height": 576})
    for t in sc["tasks"]:
        assert {"index", "summary", "optimal_steps", "risky_steps"} <= set(t)
        if sc["kind"] != "game":
            assert 3 <= t["optimal_steps"] <= 10, t
    names = [st["name"] for st in sc["states"]]
    assert 4 <= len(names) <= 8 and len(set(names)) == len(names), names
    for st in sc["states"]:
        labels = st["labels"]
        assert set(build_registry.LABEL_KEYS) <= set(labels), st
        assert labels["page_kind"] in build_registry.PAGE_KINDS, st
    assert any(st["labels"]["error"] for st in sc["states"]), f"{app}: no error state"


def test_registry_builds():
    sys.path.insert(0, APPS_DIR)
    import build_registry

    ids = [s["id"] for s in build_registry.load()["scenarios"]]
    assert set(APPS) <= set(ids), ids


# --- oracle runs ---------------------------------------------------------------------------------


DOM_RUNS = [(app, task, seed, 0) for app in DOM_APPS for task in tasks_of(app) for seed in SEEDS] + \
           [(app, task, seed, 1) for app in DOM_APPS for task in tasks_of(app) for seed in INTERRUPT_SEEDS]


@pytest.mark.parametrize("app,task,seed,interrupts", DOM_RUNS)
def test_oracle_dom(open_app, app, task, seed, interrupts):
    meta = task_meta(app, task)
    s = open_app(app, task=task, seed=seed, interrupts=interrupts)
    assert s.js("blinkScenario.task"), "empty task text"
    assert not s.check()["success"], "task is already complete at load"
    steps, risky, exp = follow_oracle(s, meta["optimal_steps"] * 2 + 4)
    chk = s.check()
    assert chk["done"] and chk["success"], f"{s.url}: {chk}"
    assert exp.get("action") == "none" and exp.get("done") and not exp.get("risky"), exp
    assert steps == meta["optimal_steps"] + interrupts, f"{s.url}: {steps} steps"
    assert risky == meta["risky_steps"], f"{s.url}: {risky} risky steps"


CANVAS_RUNS = [("desktop", task, seed, "light") for task in tasks_of("desktop") for seed in SEEDS] + \
              [("desktop", task, seed, "dark") for task in tasks_of("desktop") for seed in (1, 2)]


@pytest.mark.parametrize("app,task,seed,theme", CANVAS_RUNS)
def test_oracle_canvas(open_app, app, task, seed, theme):
    meta = task_meta(app, task)
    s = open_app(app, task=task, seed=seed, theme=theme)
    assert not s.check()["success"]
    steps, risky, exp = follow_oracle(s, meta["optimal_steps"] * 2 + 4)
    chk = s.check()
    assert chk["done"] and chk["success"], f"{s.url}: {chk}"
    assert exp.get("done") and exp.get("action") == "none" and not exp.get("risky"), exp
    assert steps == meta["optimal_steps"], f"{s.url}: {steps} steps"
    assert risky == meta["risky_steps"], f"{s.url}: {risky} risky steps"


def click_target_fast(s: Session, target_id: str):
    for t in s.targets():
        if t["id"] == target_id:
            r = t["rect"]
            s.page.mouse.click(r["x"] + r["w"] / 2, r["y"] + r["h"] / 2)
            return
    raise AssertionError(f"no target {target_id}")


def play_game(s: Session, cadence_ms: int, click: bool = True, max_iters: int = 3000) -> dict:
    s.js("blinkScenario.pause(true)")
    ids = [t["id"] for t in assert_targets(s)]
    assert ids == ["left", "middle", "right"], ids
    for _ in range(max_iters):
        exp = s.expected()
        if is_final(exp):
            break
        assert exp["target"] in ids and exp["action"] == "click" and not exp["risky"], exp
        if click:
            click_target_fast(s, exp["target"])
        s.js(f"blinkScenario.advance({cadence_ms})")
    return s.check()


@pytest.mark.parametrize("seed", SEEDS)
def test_game_survives_slow(open_app, seed):
    s = open_app("game", seed=seed, speed="slow")
    assert s.js("blinkScenario.waves") == GAME_WAVES
    chk = play_game(s, 400)
    assert chk["done"] and chk["success"], f"{s.url}: {chk}"
    assert s.expected()["done"]


@pytest.mark.parametrize("seed", (1, 2))
def test_game_survives_normal(open_app, seed):
    s = open_app("game", seed=seed, speed="normal")
    chk = play_game(s, 250)
    assert chk["done"] and chk["success"], f"{s.url}: {chk}"


def test_game_idle_player_gets_hit(open_app):
    s = open_app("game", seed=1, speed="slow")
    chk = play_game(s, 400, click=False)
    assert chk["done"] and not chk["success"], chk


def test_game_is_deterministic(open_app):
    shots = []
    for _ in range(2):
        s = open_app("game", seed=3, speed="slow", paused=1)
        s.js("blinkScenario.advance(5200)")
        shots.append((s.expected(), s.check(), s.page.screenshot()))
    assert shots[0][:2] == shots[1][:2]
    assert shots[0][2] == shots[1][2], "same seed and simulated time rendered different frames"


# --- named states and themes ---------------------------------------------------------------------


STATE_RUNS = [(app, st["name"], task) for app in APPS for st in scenario(app)["states"] for task in tasks_of(app)]


@pytest.mark.parametrize("app,name,task", STATE_RUNS)
def test_named_state_opens(open_app, app, name, task):
    s = open_app(app, state=name, task=task, seed=2)
    assert s.state() == name, f"{s.url}: state() is {s.state()!r}"
    if scenario(app)["kind"] == "dom":
        assert_dom_screen(s)
    else:
        assert_targets(s)
    s.check()
    exp = s.expected()
    assert exp.get("action") in ("click", "type", "select", "none"), exp


def mean_luma(png: bytes) -> float:
    from PIL import Image, ImageStat

    return ImageStat.Stat(Image.open(io.BytesIO(png)).convert("L")).mean[0]


@pytest.mark.parametrize("app", APPS)
def test_themes_differ(open_app, app):
    light = open_app(app, theme="light", seed=1)
    dark = open_app(app, theme="dark", seed=1)
    assert dark.js("document.documentElement.dataset.theme") == "dark"
    for s in (light, dark):
        if scenario(app)["kind"] == "dom":
            assert_dom_screen(s)
        else:
            assert_targets(s)
    a, b = mean_luma(light.page.screenshot()), mean_luma(dark.page.screenshot())
    assert a - b > 25, f"{app}: dark theme is not darker (light {a:.0f}, dark {b:.0f})"


# --- recovery: a few seeded random actions, then the oracle must still be right ----------------


WALK_JS = """(i) => {
  const el = window.__btCands[i];
  return {tag: el.tagName.toLowerCase(), type: (el.getAttribute('type') || '').toLowerCase(),
          options: el.tagName === 'SELECT' ? [...el.options].map(o => o.label) : []};
}"""


def random_step(s: Session, rng: random.Random, junk: list[str]):
    if scenario(s.app)["kind"] != "dom":
        tg = s.targets()
        if tg:
            click_target(s, rng.choice(tg)["id"])
        return
    cands = s.candidates()
    if not cands:
        return
    i = rng.randrange(len(cands))
    info = s.js(WALK_JS, i)
    el = s.page.evaluate_handle("(i) => window.__btCands[i]", i).as_element()
    if info["tag"] == "select":
        el.select_option(label=rng.choice(info["options"]))
    elif info["tag"] in ("input", "textarea") and info["type"] not in ("checkbox", "radio", "button", "submit"):
        el.fill(rng.choice(junk))
    else:
        el.click()
    s.page.wait_for_timeout(SETTLE_MS)


WALK_RUNS = [(app, task, walk) for app in ("travel", "careers", "phone", "desktop") for task in tasks_of(app)
             for walk in range(4)]


@pytest.mark.parametrize("app,task,walk", WALK_RUNS)
def test_oracle_recovers_after_random_actions(open_app, app, task, walk):
    rng = random.Random(f"{app}-{task}-{walk}")
    s = open_app(app, task=task, seed=walk + 1)
    junk = s.js("blinkScenario.values") + ["LAX", "Designer", "x"]
    for _ in range(rng.randint(2, 5)):
        random_step(s, rng, junk)
        s.check()
        s.expected()
    _, _, exp = follow_oracle(s, 40)
    chk = s.check()
    if exp.get("failed"):
        assert not chk["success"], f"{s.url}: oracle says failed but check() succeeds"
    else:
        assert chk["success"], f"{s.url}: oracle stopped ({exp}) without success: {chk}"


# --- forbidden side effects --------------------------------------------------------------------


def oracle_until(s: Session, pred, limit: int = 30) -> dict:
    """Follow the oracle until pred(expected) holds; returns that expected()."""
    for _ in range(limit):
        exp = s.expected()
        if pred(exp):
            return exp
        assert not is_final(exp), f"{s.url}: oracle finished before the point of the mistake"
        if scenario(s.app)["kind"] == "dom":
            act_dom(s, exp)
        else:
            click_target(s, exp["target"])
    raise AssertionError(f"{s.url}: never reached the point of the mistake")


def click_other(s: Session, query: str, avoid_sel: str) -> str:
    """Click a visible element matching query that is not the oracle's element."""
    sel = s.js("""([q, avoid]) => {
      const bad = document.querySelector(avoid);
      const el = [...document.querySelectorAll(q)].find(e => e !== bad && e.checkVisibility() && !e.disabled);
      return el && el.id ? '#' + CSS.escape(el.id) : null;
    }""", [query, avoid_sel])
    assert sel, f"{s.url}: no alternative to {avoid_sel} among {query}"
    click(s, sel)
    return sel


def click(s: Session, sel: str):
    s.page.locator(sel).click()
    s.page.wait_for_timeout(SETTLE_MS)


def assert_failed(s: Session):
    chk, exp = s.check(), s.expected()
    assert not chk["success"], f"{s.url}: forbidden side effect still succeeds: {chk}"
    return chk, exp


@pytest.mark.parametrize("seed", (1, 3))
def test_travel_task1_wrong_flight_booked(open_app, seed):
    s = open_app("travel", task=1, seed=seed)
    exp = oracle_until(s, lambda e: (e.get("selector") or "").startswith("#flight-"))
    click_other(s, "[id^='flight-']", exp["selector"])
    click(s, "#continue-button")
    click(s, "#pay-now")
    chk, exp = assert_failed(s)
    assert chk["done"] and exp.get("failed") and exp["action"] == "none", (chk, exp)


def test_travel_task1_wrong_fare_booked(open_app):
    s = open_app("travel", task=1, seed=2)
    exp = oracle_until(s, lambda e: (e.get("selector") or "").startswith("#fare-"))
    click_other(s, "[id^='fare-']", exp["selector"])
    click(s, "#continue-button")
    click(s, "#pay-now")
    _, exp = assert_failed(s)
    assert exp.get("failed")


def test_travel_task0_paying_is_forbidden(open_app):
    s = open_app("travel", task=0, seed=4)
    follow_oracle(s, 20)
    assert s.check()["success"]
    click(s, "#continue-button")
    assert not s.check()["success"] and s.expected()["selector"], "dialog open: the oracle should cancel it"
    click(s, "#pay-now")
    _, exp = assert_failed(s)
    assert exp.get("failed")


def test_travel_task0_not_cheapest(open_app):
    s = open_app("travel", task=0, seed=5)
    exp = oracle_until(s, lambda e: (e.get("selector") or "").startswith("#flight-"))
    click_other(s, "[id^='flight-']", exp["selector"])
    assert_failed(s)
    follow_oracle(s, 12)
    assert s.check()["success"], "the oracle should recover by going back to the results"


def test_careers_task0_submitting_is_forbidden(open_app):
    s = open_app("careers", task=0, seed=2)
    follow_oracle(s, 20)
    assert s.check()["success"]
    click(s, "#submit-application")
    _, exp = assert_failed(s)
    assert exp.get("failed") and exp["action"] == "none"


@pytest.mark.parametrize("task", (0, 1))
def test_careers_wrong_role(open_app, task):
    s = open_app("careers", task=task, seed=3)
    exp = oracle_until(s, lambda e: (e.get("selector") or "").startswith("#role-"))
    click_other(s, "[id^='role-']", exp["selector"])
    if task == 0:
        click(s, "#apply-button")
    assert_failed(s)
    follow_oracle(s, 14)
    assert s.check()["success"]


def test_phone_task0_wrong_order_placed(open_app):
    s = open_app("phone", task=0, seed=2)
    click(s, "#tab-orders")
    click(s, "#reorder-1")
    click(s, "#add-reorder")
    click(s, "#place-order")
    _, exp = assert_failed(s)
    assert exp.get("failed") and exp["action"] == "none"


def test_phone_task1_wrong_address_saved(open_app):
    s = open_app("phone", task=1, seed=3)
    exp = oracle_until(s, lambda e: (e.get("selector") or "").startswith("#address-")
                       and e["selector"] != "#address-pill")
    click_other(s, "[id^='address-']:not(#address-pill)", exp["selector"])
    click(s, "#save-address")
    assert_failed(s)
    follow_oracle(s, 12)
    assert not s.check()["success"], "a wrong address was saved first; success must stay false"


def test_desktop_task0_wrong_photo_deleted(open_app):
    s = open_app("desktop", task=0, seed=2)
    exp = oracle_until(s, lambda e: (e.get("target") or "").startswith("photo-"))
    other = next(t["id"] for t in s.targets() if t["id"].startswith("photo-") and t["id"] != exp["target"])
    click_target(s, other)
    click_target(s, "viewer-delete")
    click_target(s, "confirm-delete")
    _, exp = assert_failed(s)
    assert exp.get("failed") and exp["action"] == "none"


def test_desktop_task1_changed_accent(open_app):
    s = open_app("desktop", task=1, seed=4)
    goal = oracle_until(s, lambda e: (e.get("target") or "").startswith("appearance-"))
    accents = [t["id"] for t in s.targets() if t["id"].startswith("accent-")]
    assert accents, "no accent swatches on the appearance panel"
    click_target(s, next(a for a in reversed(accents)))
    click_target(s, accents[0])  # at least one of the two differs from the initial accent
    click_target(s, goal["target"])
    assert not s.check()["success"], "a changed accent colour must fail the settings task"
    follow_oracle(s, 10)
    assert s.check()["success"], "settings are reversible: the oracle restores the accent"


# --- review screenshots --------------------------------------------------------------------------


def render_shots(apps):
    from playwright.sync_api import sync_playwright

    srv = start_server()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        for app in apps:
            out = os.path.join(SHOTS_DIR, app)
            os.makedirs(out, exist_ok=True)

            def shot(fname, **params):
                s = Session(b, base, app, **params)
                s.page.wait_for_timeout(300)
                s.page.screenshot(path=os.path.join(out, fname))
                s.close()

            shot("start.png", seed=1)
            shot("start-dark.png", seed=1, theme="dark")
            for st in scenario(app)["states"]:
                shot(f"state-{st['name']}.png", seed=1, state=st["name"])
            for task in (tasks_of(app) if app != "game" else []):
                s = Session(b, base, app, task=task, seed=1, interrupts=1 if app in DOM_APPS else 0)
                for n in range(24):
                    s.page.wait_for_timeout(260)
                    s.page.screenshot(path=os.path.join(out, f"flow-t{task}-{n:02d}.png"))
                    exp = s.expected()
                    if is_final(exp):
                        break
                    if app in DOM_APPS:
                        act_dom(s, exp)
                    else:
                        click_target(s, exp["target"])
                s.close()
            if app == "game":
                s = Session(b, base, app, seed=1, speed="slow")
                s.js("blinkScenario.pause(true)")
                for n in range(6):
                    s.js("blinkScenario.advance(1400)")
                    exp = s.expected()
                    if exp.get("target"):
                        click_target_fast(s, exp["target"])
                    s.js("blinkScenario.advance(160)")
                    s.page.screenshot(path=os.path.join(out, f"play-{n:02d}.png"))
                s.close()
            print(f"{app}: {out}")
        b.close()
    srv.shutdown()


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "shots":
        render_shots(sys.argv[2:] or list(APPS))
    else:
        print(__doc__)
