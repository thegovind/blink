"""Tests for the apps-a scenario apps: shop, mail, settings, calendar, files (CPU only, no model).

    uv run --no-project --python 3.12 --with playwright --with pillow --with pytest \
        python -m pytest examples/cua/tests/test_apps_a.py -q

The apps are served from examples/cua/apps/ on loopback; any other request fails the test, and so
does any console error. Each app's own oracle (blinkScenario.expected()) drives it step by step with
the SPEC candidate rule checked at every step.
"""

from __future__ import annotations

import json
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import apps_a_check as C  # noqa: E402
import apps_a_helpers as H  # noqa: E402

pw_api = pytest.importorskip("playwright.sync_api")

SCENARIOS = {app: H.scenario(app) for app in H.APPS}
# a wrong click on one of these would commit something (or sign out); the recovery test avoids them
RISKY_WORDS = re.compile(r"\b(send|delete|share|place order|pay|cancel meeting|sign out|log out|book room)\b", re.I)


def oracle_cases():
    for app, sc in SCENARIOS.items():
        for t in sc["tasks"]:
            for seed in range(1, 6):
                yield pytest.param(app, t, seed, 0, id=f"{app}-t{t['index']}-s{seed}")
            for seed in (1, 2):
                yield pytest.param(app, t, seed, 1, id=f"{app}-t{t['index']}-s{seed}-interrupt")


def state_cases():
    for app, sc in SCENARIOS.items():
        for st in sc["states"]:
            yield pytest.param(app, st["name"], len(sc["tasks"]), id=f"{app}-{st['name']}")


def negative_cases():
    for app in H.APPS:
        cases = C.negatives(app)
        assert cases, f"{app} has no negative cases"
        for i, case in enumerate(cases):
            yield pytest.param(app, case, id=f"{app}-neg{i}")


@pytest.fixture(scope="module")
def base():
    with H.serve() as address:
        yield address


@pytest.fixture(scope="module")
def browser():
    with pw_api.sync_playwright() as pw:
        b = pw.chromium.launch()
        yield b
        b.close()


@pytest.fixture()
def page(browser):
    ctx = H.new_context(browser)
    guard = H.Guard().attach(ctx)
    pg = ctx.new_page()
    yield pg
    problems = guard.reset()
    ctx.close()
    assert not problems, f"network or console problems: {problems}"


# ---------- static checks ----------

@pytest.mark.parametrize("app", H.APPS)
def test_scenario_json(app):
    sc = SCENARIOS[app]
    for key in ("id", "title", "path", "viewport", "scale", "kind", "tasks", "states"):
        assert key in sc, key
    assert sc["id"] == app and sc["path"] == f"{app}/index.html" and sc["kind"] == "dom"
    assert sc["viewport"] == H.VIEWPORT and sc["scale"] == H.SCALE
    assert 3 <= len(sc["title"].split()) <= 5
    assert [t["index"] for t in sc["tasks"]] == list(range(len(sc["tasks"])))
    for t in sc["tasks"]:
        assert 3 <= t["optimal_steps"] <= 10 and 0 <= t["risky_steps"] <= t["optimal_steps"] and t["summary"]
    assert 4 <= len(sc["states"]) <= 8
    names = [s["name"] for s in sc["states"]]
    assert len(set(names)) == len(names)
    for st in sc["states"]:
        assert set(st["labels"]) == {"page_kind", "modal", "error", "signed_in", "loading"}
    assert any(s["labels"]["error"] for s in sc["states"]), "needs an error state"
    assert any(s["labels"]["loading"] or s["labels"]["page_kind"] == "login" for s in sc["states"]), "needs login or loading"


def test_registry_matches_scenarios():
    with open(os.path.join(H.APPS_DIR, "registry.json"), encoding="utf-8") as fh:
        reg = {s["id"]: s for s in json.load(fh)["scenarios"]}
    for app in H.APPS:
        assert reg.get(app) == SCENARIOS[app], f"registry.json is stale for {app}; run build_registry.py"


@pytest.mark.parametrize("app", H.APPS)
def test_self_contained(app):
    folder = os.path.join(H.APPS_DIR, app)
    for name in os.listdir(folder):
        if name.endswith((".html", ".js", ".css")):
            text = open(os.path.join(folder, name), encoding="utf-8").read()
            assert not re.search(r"(src|href)\s*=\s*[\"']?(https?:)?//", text), f"{name} references a remote URL"
            assert "@import" not in text and "url(http" not in text and "Math.random" not in text
            assert not re.search(r"lorem ipsum", text, re.I)


# ---------- oracle runs ----------

@pytest.mark.parametrize("app,t,seed,intr", list(oracle_cases()))
def test_oracle_reaches_goal(page, base, app, t, seed, intr):
    H.load(page, H.url(base, app, seed=seed, task=t["index"], interrupts=intr))
    task = page.evaluate("window.blinkScenario.task")
    values = page.evaluate("window.blinkScenario.values")
    assert task and set(re.findall(r'"([^"]+)"', task)) == set(values), (task, values)
    r = H.follow_oracle(page, t["optimal_steps"] + intr + 2)
    assert not r["problems"], r["problems"]
    chk = page.evaluate("window.blinkScenario.check()")
    assert chk["success"] and chk["done"], chk
    budget = t["optimal_steps"] + intr
    if t.get("optimal_steps_exact") is False:
        assert r["steps"] <= budget
    else:
        assert r["steps"] == budget, r["trace"]
    risky = [x for x in r["trace"] if x["expected"].get("risky")]
    assert len(risky) == t["risky_steps"]
    assert all(x["expected"]["action"] == "click" for x in risky)
    if intr:
        assert any(x["expected"].get("selector") == "#a-dismiss" for x in r["trace"]), "interrupt never appeared"
    final = r["trace"][-1]["expected"]
    assert final["action"] == "none" and final["done"] is True


@pytest.mark.parametrize("app", H.APPS)
def test_seeds_vary_and_repeat(page, base, app):
    seen = set()
    for seed in (1, 2, 3):
        H.load(page, H.url(base, app, seed=seed))
        first = (page.evaluate("window.blinkScenario.task"), json.dumps(H.candidates(page)["cands"]))
        H.load(page, H.url(base, app, seed=seed))
        again = (page.evaluate("window.blinkScenario.task"), json.dumps(H.candidates(page)["cands"]))
        assert first == again, f"seed {seed} is not deterministic"
        seen.add(first)
    assert len(seen) == 3, "seeds should change the content"


@pytest.mark.parametrize("app", H.APPS)
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_recovers_from_a_wrong_click(page, base, app, seed):
    """Click one wrong (non-committing) candidate mid-task; the oracle must still reach the goal."""
    for t in SCENARIOS[app]["tasks"]:
        for at in (0, 1, 2):
            H.load(page, H.url(base, app, seed=seed, task=t["index"]))
            for _ in range(at):
                H.act(page, page.evaluate("window.blinkScenario.expected()"))
            exp = page.evaluate("window.blinkScenario.expected()")
            info = H.candidates(page, exp.get("selector"))
            others = [c for i, c in enumerate(info["cands"])
                      if i != info["expectedIndex"] and not RISKY_WORDS.search(c["name"]) and c["role"] != "select"]
            if not others:
                continue
            c = others[(seed * 7 + at * 3 + t["index"]) % len(others)]
            x, y, w, h = c["box"]
            page.mouse.click(x + w / 2, y + h / 2)
            r = H.follow_oracle(page, t["optimal_steps"] * 2 + 6)
            where = f"{app} task {t['index']} seed {seed}: wrong click on {c['name']!r} at step {at}"
            assert not r["problems"], (where, r["problems"])
            chk = page.evaluate("window.blinkScenario.check()")
            assert chk["success"], (where, chk, r["trace"][-3:])


# ---------- named states, theme, negatives ----------

@pytest.mark.parametrize("app,name,ntasks", list(state_cases()))
def test_named_state_opens(page, base, app, name, ntasks):
    for task in range(ntasks):
        H.load(page, H.url(base, app, state=name, task=task, seed=2))
        assert page.evaluate("window.blinkScenario.state()") == name
        names = [c["name"].strip().lower() for c in H.candidates(page)["cands"]]
        assert len(names) <= H.MAX_CANDIDATES and all(names) and len(set(names)) == len(names), names
        page.evaluate("window.blinkScenario.check()")
        page.evaluate("window.blinkScenario.expected()")


@pytest.mark.parametrize("app", H.APPS)
def test_dark_theme(browser, base, app):
    ctx = H.new_context(browser, "dark")
    guard = H.Guard().attach(ctx)
    pg = ctx.new_page()
    H.load(pg, H.url(base, app, theme="dark", seed=3))
    assert pg.evaluate("document.documentElement.dataset.theme") == "dark"
    lum = pg.evaluate("""() => {
      const el = document.elementFromPoint(window.innerWidth - 20, window.innerHeight - 20);
      let n = el, c = 'rgba(0, 0, 0, 0)';
      while (n && (c === 'rgba(0, 0, 0, 0)' || c === 'transparent')) { c = getComputedStyle(n).backgroundColor; n = n.parentElement; }
      if (c === 'rgba(0, 0, 0, 0)') c = getComputedStyle(document.body).backgroundColor;
      const [r, g, b] = c.match(/\\d+/g).map(Number);
      return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;
    }""")
    assert lum < 0.35, f"dark theme background looks light ({lum:.2f})"
    problems = guard.reset()
    ctx.close()
    assert not problems, problems


@pytest.mark.parametrize("app,case", list(negative_cases()))
def test_forbidden_side_effect_fails(page, base, app, case):
    H.load(page, H.url(base, app, seed=case.get("seed", 1), task=case.get("task", 0)))
    C.run_steps(page, case["steps"])
    chk = page.evaluate("window.blinkScenario.check()")
    for key, want in case.get("expect", {"success": False}).items():
        assert chk.get(key) == want, (case["name"], chk)
    assert chk["success"] is False
