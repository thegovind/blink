"""Drive one apps-a scenario app with its own oracle, run its negative cases and render review shots.

    uv run --no-project --python 3.12 --with playwright --with pillow python examples/cua/tests/apps_a_check.py shop [--shots] [--seeds 5]

Prints one line per (task, seed, interrupts) run. Negative cases live in apps_a_neg_<app>.json:
[{"name": "...", "task": 0, "seed": 1, "steps": [["oracle", 3], ["click", "#sel"], ["fill", "#sel", "text"],
  ["select", "#sel", "label"]], "expect": {"success": false}}]
A selector that starts with "js:" is evaluated in the page and must return a CSS selector string.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import apps_a_helpers as H  # noqa: E402


def resolve(page, sel: str) -> str:
    return page.evaluate(sel[3:]) if sel.startswith("js:") else sel


def run_steps(page, steps) -> None:
    for step in steps:
        kind = step[0]
        if kind == "oracle":
            for _ in range(int(step[1])):
                H.act(page, page.evaluate("window.blinkScenario.expected()"))
            continue
        sel = resolve(page, step[1])
        if kind == "click":
            H.click_center(page, sel)
        elif kind == "fill":
            page.fill(sel, step[2], timeout=3000)
        elif kind == "select":
            page.select_option(sel, label=step[2], timeout=3000)
        else:
            raise ValueError(f"unknown step {step!r}")


def negatives(app: str) -> list:
    path = os.path.join(H.HERE, f"apps_a_neg_{app}.json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("app")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--shots", action="store_true")
    args = ap.parse_args(argv)
    from playwright.sync_api import sync_playwright

    sc = H.scenario(args.app)
    bad = 0
    with H.serve() as base, sync_playwright() as pw:
        browser = pw.chromium.launch()
        ctx = H.new_context(browser)
        guard = H.Guard().attach(ctx)
        page = ctx.new_page()
        for t in sc["tasks"]:
            for seed in range(1, args.seeds + 1):
                for intr in ([0, 1] if seed <= 2 else [0]):
                    H.load(page, H.url(base, args.app, seed=seed, task=t["index"], interrupts=intr))
                    task = page.evaluate("window.blinkScenario.task")
                    r = H.follow_oracle(page, t["optimal_steps"] + 3)
                    chk = page.evaluate("window.blinkScenario.check()")
                    risky = sum(1 for x in r["trace"] if x["expected"].get("risky"))
                    # a task whose step count depends on the seed declares "optimal_steps_exact": false and
                    # lists its maximum; every other task must take exactly optimal_steps
                    budget = t["optimal_steps"] + intr
                    steps_ok = r["steps"] <= budget if t.get("optimal_steps_exact") is False else r["steps"] == budget
                    ok = chk["success"] and steps_ok and not r["problems"] and risky == t["risky_steps"]
                    errs = guard.reset()
                    bad += (not ok) or bool(errs)
                    print("OK " if ok and not errs else "BAD", f"t{t['index']} s{seed} i{intr} steps={r['steps']}/{t['optimal_steps']}"
                          f" risky={risky} n={[x['n'] for x in r['trace']]} {chk['detail']} | {task}")
                    for p in r["problems"] + errs:
                        print("    ", p)
                    if not ok:
                        print("     trace:", [(x["state"], x["expected"].get("selector"), x["expected"].get("action"))
                                              for x in r["trace"]])
        for case in negatives(args.app):
            H.load(page, H.url(base, args.app, seed=case.get("seed", 1), task=case.get("task", 0)))
            try:
                run_steps(page, case["steps"])
                chk = page.evaluate("window.blinkScenario.check()")
                ok = all(chk.get(k) == v for k, v in case.get("expect", {"success": False}).items())
            except Exception as e:  # noqa: BLE001
                chk, ok = {"detail": f"error {e}"}, False
            errs = guard.reset()
            bad += (not ok) or bool(errs)
            print("OK " if ok and not errs else "BAD", f"negative: {case['name']} -> {chk.get('detail')}", *errs)
        for st in sc["states"]:
            for task in range(len(sc["tasks"])):
                H.load(page, H.url(base, args.app, state=st["name"], task=task, seed=1))
                got = page.evaluate("window.blinkScenario.state()")
                info = H.candidates(page)
                names = [c["name"].lower() for c in info["cands"]]
                ok = got == st["name"] and len(names) <= H.MAX_CANDIDATES and len(set(names)) == len(names) and all(names)
                errs = guard.reset()
                if not ok or errs:
                    bad += 1
                    print("BAD", f"state {st['name']} task {task}: state()={got} n={len(names)} names={names}", *errs)
        print(f"states checked: {len(sc['states'])}")
        if args.shots:
            out = os.path.join(H.RUNS_DIR, "app-shots")
            os.makedirs(out, exist_ok=True)
            for st in sc["states"]:
                H.load(page, H.url(base, args.app, state=st["name"], seed=1))
                page.screenshot(path=os.path.join(out, f"{args.app}-{st['name']}.png"))
            H.load(page, H.url(base, args.app, seed=1))
            page.screenshot(path=os.path.join(out, f"{args.app}-start.png"))
            dark = H.new_context(browser, "dark")
            dpage = dark.new_page()
            H.load(dpage, H.url(base, args.app, theme="dark", seed=2))
            dpage.screenshot(path=os.path.join(out, f"{args.app}-dark.png"))
            dark.close()
            print(f"shots in {out}")
        browser.close()
    print("bad", bad)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
