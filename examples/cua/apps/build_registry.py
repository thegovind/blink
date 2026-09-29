"""Merge every apps/<id>/scenario.json into apps/registry.json (see ../SPEC.md).

    python examples/cua/apps/build_registry.py [--check]

Each app owns its scenario.json; this file is the only writer of registry.json, so parallel
builders never edit the same file. --check exits 1 when registry.json is stale.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "registry.json")
REQUIRED = ("id", "title", "path", "viewport", "scale", "kind", "tasks", "states")
LABEL_KEYS = ("page_kind", "modal", "error", "signed_in", "loading")
PAGE_KINDS = {"home", "list", "detail", "form", "cart", "checkout", "confirm", "inbox", "message", "settings",
              "calendar", "files", "search_results", "error", "login", "dialog", "game"}


def load() -> dict:
    scenarios = []
    problems = []
    for name in sorted(os.listdir(HERE)):
        path = os.path.join(HERE, name, "scenario.json")
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as fh:
            s = json.load(fh)
        for key in REQUIRED:
            if key not in s:
                problems.append(f"{name}: missing {key}")
        if s.get("id") != name:
            problems.append(f"{name}: id {s.get('id')!r} must equal its folder name")
        if not os.path.isfile(os.path.join(HERE, s.get("path", ""))):
            problems.append(f"{name}: path {s.get('path')!r} not found")
        for st in s.get("states", []):
            labels = st.get("labels", {})
            missing = [k for k in LABEL_KEYS if k not in labels]
            if missing:
                problems.append(f"{name}: state {st.get('name')!r} lacks labels {missing}")
            if labels.get("page_kind") not in PAGE_KINDS:
                problems.append(f"{name}: state {st.get('name')!r} page_kind {labels.get('page_kind')!r} not allowed")
        scenarios.append(s)
    if problems:
        raise SystemExit("registry problems:\n  " + "\n  ".join(problems))
    return {"version": 1, "scenarios": scenarios}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    text = json.dumps(load(), indent=2, ensure_ascii=False) + "\n"
    if args.check:
        current = open(OUT, encoding="utf-8").read() if os.path.exists(OUT) else ""
        if current != text:
            print("registry.json is stale; run build_registry.py", file=sys.stderr)
            return 1
        return 0
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
