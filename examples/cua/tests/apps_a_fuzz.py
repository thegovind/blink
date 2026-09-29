"""Probe the apps-a oracles: at every oracle step, make one wrong click, then follow expected() to the goal.

    uv run --no-project --python 3.12 --with playwright python examples/cua/tests/apps_a_fuzz.py shop [--seeds 2] [--per-step 5]

A wrong click that commits something irreversible (check().done and not success right after it) is
counted, not failed. Everything else must end in check().success within optimal_steps * 2 + 6 actions.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import apps_a_helpers as H  # noqa: E402

COMMIT = re.compile(r"\b(send|delete|share|place order|pay|cancel meeting|sign out|log out|book room)\b", re.I)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("app")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--per-step", type=int, default=5)
    args = ap.parse_args(argv)
    from playwright.sync_api import sync_playwright

    sc = H.scenario(args.app)
    runs = fails = irreversible = 0
    with H.serve() as base, sync_playwright() as pw:
        browser = pw.chromium.launch()
        ctx = H.new_context(browser)
        guard = H.Guard().attach(ctx)
        page = ctx.new_page()
        for t in sc["tasks"]:
            for seed in range(1, args.seeds + 1):
                address = H.url(base, args.app, seed=seed, task=t["index"])
                for k in range(t["optimal_steps"]):
                    H.load(page, address)
                    for _ in range(k):
                        H.act(page, page.evaluate("window.blinkScenario.expected()"))
                    exp = page.evaluate("window.blinkScenario.expected()")
                    if exp.get("done"):
                        break
                    info = H.candidates(page, exp.get("selector"))
                    wrong = [c for i, c in enumerate(info["cands"]) if i != info["expectedIndex"]
                             and c["role"] != "select" and not COMMIT.search(c["name"])]
                    if len(wrong) > args.per_step:
                        step = len(wrong) / args.per_step
                        wrong = [wrong[int(i * step)] for i in range(args.per_step)]
                    for c in wrong:
                        H.load(page, address)
                        for _ in range(k):
                            H.act(page, page.evaluate("window.blinkScenario.expected()"))
                        x, y, w, h = c["box"]
                        page.mouse.click(x + w / 2, y + h / 2)
                        runs += 1
                        after = page.evaluate("window.blinkScenario.check()")
                        if after["done"] and not after["success"]:
                            irreversible += 1
                            continue
                        r = H.follow_oracle(page, t["optimal_steps"] * 2 + 6)
                        chk = page.evaluate("window.blinkScenario.check()")
                        errs = guard.reset()
                        if r["problems"] or not chk["success"] or errs:
                            fails += 1
                            print(f"FAIL t{t['index']} s{seed} step {k}: wrong click {c['name']!r} -> {chk['detail']}",
                                  r["problems"][:2], errs[:2],
                                  [(x["state"], x["expected"].get("selector")) for x in r["trace"][-4:]])
        browser.close()
    print(f"{args.app}: runs {runs}, failures {fails}, irreversible wrong clicks {irreversible}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
