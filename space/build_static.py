"""Build the static Space from the same renderers the Gradio app uses.

    python build_static.py                      # needs a complete replay.json
    python build_static.py --allow-synthetic /tmp/replay-dev.json   # marked preview

Every bundled request — every use-case example and every playground preset — is run
through blink.decide against the replay engine and written into the page. The result is
plain HTML, CSS and one small script: no server, no API, no network.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import blink
import examples
import record_replay
import results
import ui

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "static")
REPLAY = os.path.join(HERE, "replay.json")

SYNTHETIC_NOTE = (
    "Synthetic preview — the numbers on this page came from the stand-in engine, not "
    "from the model. Do not publish this build."
)


class BuildError(RuntimeError):
    """The inputs cannot produce a page worth shipping."""


# --- inputs -------------------------------------------------------------------------


def bundled() -> list[tuple[str, object, dict]]:
    return record_replay.bundled_requests()


def check_replay(path: str, synthetic_ok: bool) -> dict:
    if not os.path.exists(path):
        raise BuildError(
            f"{os.path.relpath(path, HERE)} is missing. Record it on a GPU with "
            "record_replay.py, or pass --allow-synthetic PATH for a marked preview."
        )
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    cache = data.get("requests") or {}
    if not synthetic_ok and "synthetic" in str(data.get("engine", "")).lower():
        raise BuildError(
            f"{os.path.relpath(path, HERE)} is a synthetic recording. Pass "
            "--allow-synthetic to build a marked preview from it."
        )
    missing = [name for name, state, qs in bundled() if blink.request_key(state, qs) not in cache]
    if missing:
        raise BuildError(
            f"{os.path.relpath(path, HERE)} covers {len(cache)} requests but is missing "
            f"{len(missing)}: {', '.join(missing[:4])}"
            + (" …" if len(missing) > 4 else "")
            + ". Re-record it before building."
        )
    return data


def miss_notice(engine: blink.ReplayEngine) -> str:
    """The exact notice the engine raises, so the page and the app say the same thing."""
    try:
        engine.logits("\u0000 not a recorded request", {"q": {"type": "noul"}})
    except blink.ReplayMiss as exc:
        return ui.notice_html(str(exc))
    raise BuildError("the replay engine answered a request that was never recorded")


# --- page pieces --------------------------------------------------------------------


def _out(key: str, html: str, first: bool, extra: str = "") -> str:
    cls = "blk-out" + (" on" if first else "") + (f" {extra}" if extra else "")
    return f'<div class="{cls}" data-key="{ui.esc(key)}">{html}</div>'


def _chips(group: str, chips: list[dict]) -> str:
    btns = "".join(
        f'<button type="button" class="blk-chip{" on" if i == 0 else ""}" '
        f'data-key="{ui.esc(c["key"])}">{ui.esc(c["label"])}</button>'
        for i, c in enumerate(chips)
    )
    return f'<div class="blk-chiprow" data-chips="{ui.esc(group)}">{btns}</div>'


def _field(fid: str, group: str, role: str, label: str, value: str, rows: int, mono=False) -> str:
    cls = ' class="blk-mono"' if mono else ""
    return (
        f'<div class="blk-field"><label class="blk-label" for="{fid}">{ui.esc(label)}</label>'
        f'<textarea id="{fid}" rows="{rows}"{cls} spellcheck="false" '
        f'data-group="{ui.esc(group)}" data-role="{role}">{ui.esc(value)}</textarea></div>'
    )


def playground_pane(notice: str) -> tuple[str, dict]:
    group = "playground"
    chips, outs = [], []
    for i, (label, state, qs_text) in enumerate(ui.PLAYGROUND_PRESETS):
        key = blink.request_key(blink.as_state(state), json.loads(qs_text))
        chips.append({"key": key, "label": label, "state": state, "questions": qs_text})
        outs.append(_out(key, ui.run_playground(state, qs_text), i == 0))
    outs.append(_out("__miss__", notice, False, "blk-miss"))
    first = chips[0]
    html = (
        f'<section class="blk-pane" id="pane-playground" role="tabpanel" '
        f'aria-labelledby="tab-playground">{ui.playground_note()}'
        '<div class="blk-cols"><div>'
        + ui.demo_label()
        + _chips(group, chips)
        + _field("pg-state", group, "state", "State", first["state"], 11)
        + _field("pg-questions", group, "questions", "Questions", first["questions"], 20, True)
        + f'<button type="button" class="blk-go" data-go="{group}">Decide</button>'
        "</div>"
        f'<div class="blk-slot" data-slot="{group}">{"".join(outs)}</div>'
        "</div></section>"
    )
    return html, {group: chips}


def case_pane(case: examples.UseCase, notice: str) -> tuple[str, dict]:
    group = f"case-{case.key}"
    chips, outs = [], []
    for i, ex in enumerate(case.examples):
        key = blink.request_key(ex.state, case.questions)
        chips.append({"key": key, "label": ex.label, "state": ex.state})
        outs.append(_out(key, ui.run_use_case(case, ex.state), i == 0))
    outs.append(_out("__miss__", notice, False, "blk-miss"))
    html = (
        f'<section class="blk-pane blk-case" id="pane-{group}" role="tabpanel" '
        f'aria-labelledby="tab-{group}">{ui.use_case_note(case)}'
        '<div class="blk-cols"><div>'
        + ui.demo_label()
        + _chips(group, chips)
        + _field(
            f"{group}-state", group, "state", case.state_label, chips[0]["state"],
            max(8, case.state_lines),
        )
        + f'<button type="button" class="blk-go" data-go="{group}">Decide</button>'
        + ui.use_case_details(case)
        + "</div>"
        f'<div class="blk-slot" data-slot="{group}">{"".join(outs)}</div>'
        "</div></section>"
    )
    return html, {group: chips}


def tabs(items: list[tuple[str, str]], sub: bool = False) -> str:
    btns = "".join(
        f'<button type="button" role="tab" id="tab-{ui.esc(tid)}" '
        f'aria-controls="pane-{ui.esc(tid)}" aria-selected="{"true" if i == 0 else "false"}" '
        f'data-target="pane-{ui.esc(tid)}">{ui.esc(label)}</button>'
        for i, (tid, label) in enumerate(items)
    )
    return f'<div class="blk-tabs{" sub" if sub else ""}" role="tablist">{btns}</div>'


def page(synthetic: bool) -> str:
    engine = blink.engine()
    notice = miss_notice(engine)

    pg_html, chip_data = playground_pane(notice)
    case_html = []
    for case in examples.USE_CASES:
        html, data = case_pane(case, notice)
        case_html.append(html)
        chip_data.update(data)

    cases_pane = (
        '<section class="blk-pane" id="pane-usecases" role="tabpanel" '
        'aria-labelledby="tab-usecases">'
        + tabs([(f"case-{c.key}", c.title) for c in examples.USE_CASES], sub=True)
        + "".join(case_html)
        + "</section>"
    )
    results_pane = (
        '<section class="blk-pane" id="pane-results" role="tabpanel" '
        'aria-labelledby="tab-results">' + "".join(ui.results_blocks()) + "</section>"
    )
    how_pane = (
        '<section class="blk-pane" id="pane-how" role="tabpanel" aria-labelledby="tab-how">'
        + "".join(ui.how_blocks())
        + "</section>"
    )

    title = "blink" + (" · synthetic preview" if synthetic else "")
    mark = f'<div class="blk-warn blk-synthetic">{ui.esc(SYNTHETIC_NOTE)}</div>' if synthetic else ""
    body_attr = ' data-synthetic="1"' if synthetic else ""
    payload = json.dumps(
        {"chips": chip_data, "synthetic": synthetic}, ensure_ascii=False
    ).replace("</", "<\\/")

    return (
        "<!doctype html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{ui.esc(title)}</title>\n"
        f'<meta name="description" content="{ui.esc(results.load()["decision_index"]["caption"])}">\n'
        '<link rel="stylesheet" href="style.css">\n</head>\n'
        f"<body{body_attr}>\n"
        '<main class="blk-shell">'
        + mark
        + ui.masthead()
        + tabs(
            [
                ("playground", "Playground"),
                ("usecases", "Use cases"),
                ("results", "Results"),
                ("how", "How it works"),
            ]
        )
        + pg_html
        + cases_pane
        + results_pane
        + how_pane
        + ui.footer()
        + "</main>\n"
        f'<script type="application/json" id="blk-data">{payload}</script>\n'
        '<script src="app.js"></script>\n</body>\n</html>\n'
    )


APP_JS = """/* Tabs, chips, and the one thing a static page can say about an edited input. */
(function () {
  "use strict";
  var DATA = JSON.parse(document.getElementById("blk-data").textContent);
  var all = function (sel, root) {
    return Array.prototype.slice.call((root || document).querySelectorAll(sel));
  };

  all('[role="tablist"]').forEach(function (bar) {
    var tabs = all("button", bar);
    var panes = tabs.map(function (t) { return document.getElementById(t.dataset.target); });
    tabs.forEach(function (tab, i) {
      tab.addEventListener("click", function () {
        tabs.forEach(function (t, j) {
          t.setAttribute("aria-selected", j === i ? "true" : "false");
          if (panes[j]) panes[j].classList.toggle("on", j === i);
        });
      });
    });
    if (panes[0]) panes[0].classList.add("on");
  });

  Object.keys(DATA.chips).forEach(function (group) {
    var chips = DATA.chips[group];
    var row = document.querySelector('[data-chips="' + group + '"]');
    var slot = document.querySelector('[data-slot="' + group + '"]');
    var state = document.querySelector('textarea[data-group="' + group + '"][data-role="state"]');
    var qs = document.querySelector('textarea[data-group="' + group + '"][data-role="questions"]');
    if (!row || !slot) return;
    var current = chips[0];

    var show = function (key) {
      all(".blk-out", slot).forEach(function (el) {
        var on = el.dataset.key === key;
        if (on && el.classList.contains("on")) {
          el.classList.remove("on");
          void el.offsetWidth; /* replay the entrance animation */
        }
        el.classList.toggle("on", on);
      });
    };
    var edited = function () {
      if (state && current.state !== undefined && state.value !== current.state) return true;
      return !!(qs && current.questions !== undefined && qs.value !== current.questions);
    };
    var refresh = function () { show(edited() ? "__miss__" : current.key); };

    all("button", row).forEach(function (btn, i) {
      btn.addEventListener("click", function () {
        current = chips[i];
        all("button", row).forEach(function (b, j) { b.classList.toggle("on", i === j); });
        if (state) state.value = current.state;
        if (qs && current.questions !== undefined) qs.value = current.questions;
        show(current.key);
      });
    });
    [state, qs].forEach(function (box) {
      if (box) box.addEventListener("input", refresh);
    });
    var go = document.querySelector('[data-go="' + group + '"]');
    if (go) go.addEventListener("click", refresh);
  });
})();
"""


# --- writing ------------------------------------------------------------------------


def space_readme(synthetic: bool) -> str:
    """Front matter carried over from the Gradio README, retargeted at a static Space."""
    with open(os.path.join(HERE, "README.md"), encoding="utf-8") as fh:
        src = fh.read().split("---\n")
    drop = ("sdk:", "sdk_version:", "app_file:", "suggested_hardware:", "python_version:")
    keep = [ln for ln in src[1].splitlines() if not ln.startswith(drop)]
    body = (
        "# blink\n\n"
        "A decision model. You send a state — a ticket, an email, a policy document, a row "
        "of JSON — and a set of typed questions. It returns a probability for every option "
        "you offered, in one forward pass, having generated nothing.\n\n"
        "This page is static: saved runs, not live inference. Every example and preset on "
        "the page is a real response from the published weights, saved with the page, and "
        "the latency shown is from that run. Editing an input asks for a response that was "
        "never saved, and the page says so.\n\n"
        "To run the model, `blink.py` is published alongside the weights at "
        "`thegovind/blink-4b`. The weights are for non-commercial research use; see the model cards. "
        "The `license` above covers this Space's code. The Results tab shows what the model "
        "scores and on which edition of the benchmark.\n\n"
        "Built from the app sources with `python build_static.py`.\n"
    )
    if synthetic:
        body = f"> {SYNTHETIC_NOTE}\n\n" + body
    return "---\n" + "\n".join(["sdk: static", "app_file: index.html"] + keep) + "\n---\n\n" + body


def build(out_dir: str = OUT_DIR, replay_path: str | None = None, synthetic: bool = False) -> dict:
    path = replay_path or REPLAY
    check_replay(path, synthetic)
    engine = blink.ReplayEngine(path, blink.TEMPERATURE)
    saved = blink._ENGINE
    blink._ENGINE = engine
    try:
        html = page(synthetic)
    finally:
        blink._ENGINE = saved

    os.makedirs(out_dir, exist_ok=True)
    written = {}
    for name, text in (
        ("index.html", html),
        ("app.js", APP_JS),
        ("README.md", space_readme(synthetic)),
    ):
        dst = os.path.join(out_dir, name)
        with open(dst, "w", encoding="utf-8") as fh:
            fh.write(text)
        written[name] = dst
    shutil.copyfile(os.path.join(HERE, "style.css"), os.path.join(out_dir, "style.css"))
    written["style.css"] = os.path.join(out_dir, "style.css")
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument(
        "--allow-synthetic",
        metavar="PATH",
        help="build a visibly marked preview from a stand-in recording",
    )
    a = ap.parse_args(argv)
    try:
        written = build(a.out, a.allow_synthetic, synthetic=bool(a.allow_synthetic))
    except BuildError as exc:
        print(f"build_static: {exc}", file=sys.stderr)
        return 2
    total = sum(os.path.getsize(p) for p in written.values())
    for name, p in written.items():
        print(f"  {name:12s} {os.path.getsize(p) / 1024:8.1f} KB")
    print(f"wrote {os.path.relpath(a.out, HERE)}/ — {total / 1024:.1f} KB total")
    if a.allow_synthetic:
        print("SYNTHETIC PREVIEW: marked in the page. Do not publish.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
