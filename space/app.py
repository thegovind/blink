"""blink — the Gradio interface for one-pass typed decisions.

    BLINK_MOCK=1 uv run --with gradio python app.py   # no weights, fabricated numbers
    uv run --with gradio python app.py                # runs the model when a GPU is there

The first render of every tab comes from a saved run, because no device can be attached
while the app is starting. Every click after that runs the model.
"""

from __future__ import annotations

import inspect
import json
import os
import time
import traceback

import gradio as gr

import author
import blink
import examples
import ui

HERE = os.path.dirname(os.path.abspath(__file__))
TITLE = "blink"

DATA = ui.DATA

with open(os.path.join(HERE, "style.css"), encoding="utf-8") as fh:
    CSS = fh.read()

# Claim the engines at startup so the first request never pays for loading, and because
# a device is attached at import rather than mid-request.
blink.warm()
author.warm()

def one_palette(theme):
    """Draw the page the same whether the visitor's system is light or dark.

    Gradio marks the page dark from the system setting, or from `?__theme=dark`, which
    the Hub can pass, and its dark values put near-white text on this page's light
    surfaces. Every dark value is given its light counterpart.
    """
    for name in [n for n in vars(theme) if n.endswith("_dark")]:
        light = name[: -len("_dark")]
        if light in vars(theme):
            setattr(theme, name, getattr(theme, light))
    return theme


# blink's cobalt as a gradio hue, so what gradio draws itself (menus, the loader, focus)
# agrees with style.css
BLINK_HUE = gr.themes.Color(
    c50="#eef1fe", c100="#dfe5fd", c200="#c3cefc", c300="#9aabfa", c400="#6a82f8",
    c500="#4a66f6", c600="#2447f5", c700="#1c38d0", c800="#1a2fa6", c900="#15237a",
    c950="#0e1650", name="blink",
)

THEME = one_palette(gr.themes.Base(
    primary_hue=BLINK_HUE,
    secondary_hue=gr.themes.colors.neutral,
    neutral_hue=gr.themes.colors.neutral,
    font=["Geist", "Inter", "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=["Geist Mono", "ui-monospace", "SF Mono", "Menlo", "monospace"],
).set(
    body_background_fill="#f5f5f5",
    body_text_color="#0a0a0a",
    body_text_color_subdued="#666666",
    background_fill_primary="#ffffff",
    background_fill_secondary="#fafafa",
    block_background_fill="#ffffff",
    block_border_color="#e5e5e5",
    block_label_text_color="#666666",
    block_title_text_color="#0a0a0a",
    block_info_text_color="#666666",
    border_color_primary="#e5e5e5",
    border_color_accent="#2447f5",
    border_color_accent_subdued="#c3cefc",
    color_accent="#2447f5",
    color_accent_soft="#eef1fe",
    input_background_fill="#ffffff",
    input_background_fill_focus="#ffffff",
    input_border_color="#e5e5e5",
    input_border_color_hover="#d4d4d4",
    input_border_color_focus="#2447f5",
    input_placeholder_color="#737373",
    input_radius="18px",
    block_radius="24px",
    container_radius="24px",
    button_large_radius="18px",
    button_medium_radius="18px",
    button_small_radius="18px",
    button_primary_background_fill="#2447f5",
    button_primary_background_fill_hover="#1c38d0",
    button_primary_border_color="#2447f5",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="#ffffff",
    button_secondary_background_fill_hover="#fafafa",
    button_secondary_border_color="#e5e5e5",
    button_secondary_text_color="#0a0a0a",
    checkbox_background_color_selected="#2447f5",
    checkbox_border_color_selected="#2447f5",
    code_background_fill="#fafafa",
    error_background_fill="#fef2f2",
    error_border_color="#fecaca",
    error_text_color="#b91c1c",
    error_icon_color="#b91c1c",
    link_text_color="#2447f5",
    link_text_color_hover="#1c38d0",
    link_text_color_active="#1c38d0",
    link_text_color_visited="#1c38d0",
    loader_color="#2447f5",
    slider_color="#2447f5",
    table_border_color="#e5e5e5",
    table_even_background_fill="#ffffff",
    table_odd_background_fill="#fafafa",
))

# Each question's options, drawn as chips from its options box, so they can be read
# without opening anything. It only reads the page and writes into its own list: it
# listens to nothing, sends nothing and marks nothing. The keys follow ui.parse_options.
CHIPS = r"""
(() => {
  if (window.__blinkChips) return;
  window.__blinkChips = true;
  const slug = (s) => s.trim().toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '');
  const items = (text, kind) => {
    const lines = String(text || '').split('\n').map((l) => l.trim()).filter(Boolean);
    if (kind === 'score') return lines.map((l, i) => [String(i), l]);
    const keys = lines.map((l) => {
      const at = l.indexOf(':');
      const k = at < 0 ? '' : l.slice(0, at).trim();
      const d = at < 0 ? '' : l.slice(at + 1).trim();
      return k && d ? (slug(k) || slug(l)) : slug(l);
    });
    const clash = new Set(keys).size !== keys.length || keys.some((k) => !k);
    return lines.map((l, i) => [null, clash ? l : keys[i]]);
  };
  const paint = (row) => {
    const list = row.querySelector('.blk-optchips');
    const box = row.querySelector('.blk-qopts textarea');
    if (!list || !box) return;
    const pick = row.querySelector('.blk-qtype input');
    const kind = pick ? pick.value.trim() : '';
    const sig = kind + '\n' + box.value;
    if (list.dataset.sig === sig) return;
    list.dataset.sig = sig;
    list.replaceChildren(...items(box.value, kind).map(([n, t]) => {
      const li = document.createElement('li');
      if (n !== null) {
        const i = document.createElement('i');
        i.textContent = n;
        li.append(i);
      }
      li.append(document.createTextNode(t));
      return li;
    }));
  };
  // a name is as wide as its text: the field is shrunk to nothing and grown back to
  // what it scrolls, so the face it is actually drawn in decides
  const fit = (row) => {
    const name = row.querySelector('.blk-qname input');
    if (!name || name.offsetParent === null) return;
    const was = name.style.width;
    name.style.width = '0px';
    const want = `${Math.max(name.scrollWidth + 2, 44)}px`;
    name.style.width = was === want ? was : want;
  };
  setInterval(() => {
    document.querySelectorAll('.blk-qrow:not(.blk-qrow .blk-qrow)').forEach((row) => {
      paint(row);
      fit(row);
    });
  }, 250);
})();
"""

# Geist from Google Fonts; until it arrives, and in the static build, Inter or the system face.
HEAD = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
    'family=Geist:wght@400;500;600&family=Geist+Mono:wght@400;500;600&display=swap">'
    f"<script>{CHIPS}</script>"
)

# Gradio 6 moved theme, css and head from the Blocks constructor to launch().
GRADIO_MAJOR = int(str(getattr(gr, "__version__", "5")).split(".")[0])
LOOK = {"theme": THEME, "css": CSS, "head": HEAD}
BLOCKS_LOOK = {} if GRADIO_MAJOR >= 6 else LOOK
LAUNCH_LOOK = LOOK if GRADIO_MAJOR >= 6 else {}

esc = ui.esc
as_state = blink.as_state
run_playground = ui.run_playground
PLAYGROUND_PRESETS = ui.PLAYGROUND_PRESETS

MODELS = blink.models()
MULTI = len(MODELS) > 1


def model_switch():
    """One segmented control per interactive tab. With a single model there is no control."""
    if not MULTI:
        return gr.State(MODELS[0])
    return gr.Radio(
        choices=ui.model_choices(),
        value=MODELS[0],
        show_label=False,
        container=False,
        elem_classes="blk-seg",
    )


# --- playground ---------------------------------------------------------------------


SLOTS = ui.MAX_FORM_QUESTIONS
# seconds to hold a sync open, so a test can click Decide while one is in flight
SLOW_SYNC = float(os.environ.get("BLINK_SLOW_SYNC") or 0)
def _rows_from(flat) -> list[dict]:
    """The flat run of textboxes and dropdowns, back into rows."""
    return [
        {"name": flat[i * 4], "type": flat[i * 4 + 1], "ask": flat[i * 4 + 2],
         "options": flat[i * 4 + 3]}
        for i in range(SLOTS)
    ]


def _fill(rows: list[dict], shown: int | None = None, locked: bool = False) -> list:
    """Updates for every slot: which are shown, what is in them, and whether they edit.

    `shown` is how many rows are live; it is carried in its own state rather than
    guessed from the values, because an empty row is still a row."""
    rows = rows[:SLOTS]
    n = len(rows) if shown is None else max(0, min(int(shown), SLOTS))
    out = [gr.update(visible=i < n) for i in range(SLOTS)]
    for i in range(SLOTS):
        row = rows[i] if i < len(rows) else ui.BLANK_ROW
        out += [gr.update(value=row["name"], interactive=not locked),
                gr.update(value=row["type"], interactive=not locked),
                gr.update(value=row["ask"], interactive=not locked),
                gr.update(value=row["options"], interactive=not locked)]
    return out


def _restore(n, locked: bool) -> list:
    """Visibility and editability only.

    Leaving a tab and coming back remounts its children from the page's own config, so
    the row state has to be put back from where it is actually kept."""
    n = max(0, min(int(n or 0), SLOTS))
    return ([gr.update(visible=i < n) for i in range(SLOTS)]
            + [gr.update(interactive=not locked)] * (SLOTS * 4))


def question_form(rows: list[dict]):
    """One entry per question: what it asks, then its name and type, then its options."""
    groups, fields, removers = [], [], []
    gr.HTML(ui.form_heading(), elem_classes="blk-head")
    with gr.Group(elem_classes="blk-qblock"):
        for i in range(SLOTS):
            row = rows[i] if i < len(rows) else ui.BLANK_ROW
            with gr.Group(visible=i < len(rows), elem_classes="blk-qrow") as group:
                # the question reads first and wraps; its name and type sit quietly under it
                with gr.Column(elem_classes="blk-qhead"):
                    ask = gr.Textbox(value=row["ask"], label=ui.FORM["ask"],
                                     show_label=False, placeholder=ui.FORM["ask"],
                                     lines=1, max_lines=6, elem_classes="blk-qask")
                    with gr.Row(elem_classes="blk-qmeta"):
                        name = gr.Textbox(value=row["name"], label=ui.FORM["name"],
                                          show_label=False, placeholder=ui.FORM["name"],
                                          scale=0, min_width=0, max_lines=1,
                                          elem_classes="blk-qname")
                        qtype = gr.Dropdown(
                            choices=list(ui.Q_TYPES), value=row["type"], min_width=0,
                            label=ui.FORM["type"], show_label=False, scale=0,
                            filterable=False, elem_classes="blk-qtype",
                        )
                # the options at a glance; CHIPS draws them from the options box
                gr.HTML('<ul class="blk-optchips"></ul>', elem_classes="blk-chipslot")
                # the fold is the toggle, and what it opens sits right after it: a closed
                # fold renders nothing, and the chips read the box while it is closed
                with gr.Accordion(ui.FORM["more"], open=False, elem_classes="blk-qmore"):
                    pass
                opts = gr.Textbox(value=row["options"], label=ui.FORM["options"],
                                  show_label=False, lines=3, max_lines=8,
                                  elem_classes="blk-qopts")
                drop = gr.Button(ui.FORM["remove"], size="sm", elem_classes="blk-qdrop")
            groups.append(group)
            fields += [name, qtype, ask, opts]
            removers.append(drop)
    add = gr.Button(ui.FORM["add"], size="sm", elem_classes="blk-qadd")
    return groups, fields, removers, add


def playground_tab():
    gr.HTML(ui.playground_note())
    start_rows = ui.questions_to_rows(json.loads(examples.PLAYGROUND_QUESTIONS))
    with gr.Row(equal_height=False):
        with gr.Column(scale=5, elem_classes="blk-side"):
            model_in = model_switch()
            with gr.Row(elem_classes="blk-chips"):
                preset_btns = [gr.Button(n, size="sm") for n, _, _ in PLAYGROUND_PRESETS]
                blank_btn = gr.Button(ui.FORM["blank"], size="sm")
            gr.HTML(ui.state_heading(), elem_classes="blk-head")
            state = gr.Textbox(
                value=examples.PLAYGROUND_STATE,
                show_label=False,
                lines=9,
                max_lines=22,
                elem_classes="blk-state",
            )
            groups, fields, removers, add_btn = question_form(start_rows)
            trouble = gr.HTML()
            touched = gr.Textbox("", visible=False)
            rev = gr.Textbox("0", visible=False)
            with gr.Accordion(ui.FORM["json"], open=False, elem_classes="blk-acc"):
                questions = gr.Code(
                    value=examples.PLAYGROUND_QUESTIONS,
                    language="json",
                    show_label=False,
                    lines=12,
                    wrap_lines=True,
                )
            run_btn = gr.Button("Decide", variant="primary", elem_classes="blk-decide")
        with gr.Column(scale=7, elem_classes="blk-main"):
            # first render only: a saved run, since no device can be attached at startup
            out = gr.HTML(run_playground(examples.PLAYGROUND_STATE,
                                         examples.PLAYGROUND_QUESTIONS, prefer="saved"))

    slots = groups + fields
    first = ui.request_from_json(examples.PLAYGROUND_QUESTIONS)
    req = gr.State(first)
    shown = gr.State(len(start_rows))
    # the newest revision each surface has already taken, so a late reply can be dropped
    seen = gr.State({})

    def current(touched, text, n, flat) -> dict:
        """Validate what is on screen right now, from the values in this very call.

        A sync event may still be in flight, so nothing stored is trusted: the editor
        and the rows both answer for themselves, and `touched` says which of the two the
        visitor changed last when they still disagree. Only a changed value moves it:
        focus, a click, a key that types nothing and opening the editor all leave it
        exactly where it was."""
        held = ui.request_from_json(text)
        rows = _rows_from(flat)[: max(0, min(int(n or 0), SLOTS))]
        if touched == "rows" and not held["locked"]:
            return ui.request_from_rows(rows)  # the rows hold the newer change
        if held["problems"] or held["locked"] or touched == "json":
            # bad JSON blocks; so does a locked request or a paste the rows lag behind
            return held
        edited = ui.request_from_rows(rows)
        if edited["problems"] or edited["questions"] != held["questions"]:
            return edited  # the rows are the visible surface and were touched last
        return held

    def decide(state_text, model, touched, rev, seen, text, n, *flat):
        if not ui.newest(seen, "answers", rev):
            return gr.update()  # an older run must not paint over a newer one
        held = current(touched, text, n, flat)
        if held["problems"]:
            return ui.problems_html(held["problems"])
        return ui.run_questions(state_text, held["questions"], model)

    def waiting(touched, rev, seen, text, n, *flat):
        if not ui.newest(seen, "answers", rev):
            return gr.update()
        held = current(touched, text, n, flat)
        if held["problems"]:
            return ui.problems_html(held["problems"])
        return ui.pending_html(len(held["questions"]))

    run_in = [questions, shown, *fields]

    def live(trigger):
        """Skeleton first so the column never jumps, then the model."""
        return trigger.then(
            None, None, [touched, rev], js=READ_TOUCH, queue=False,
            show_progress="hidden"
        ).then(
            waiting, [touched, rev, seen, *run_in], out, queue=False,
            show_progress="hidden"
        ).then(decide, [state, model_in, touched, rev, seen, *run_in], out,
               show_progress="minimal")

    def sync_json(text, keep_shown=0):
        """One place where JSON becomes the request the rows show and Decide runs.

        While the JSON is wrong the rows are left alone: the request is what is blocked,
        not the work already in the form."""
        held = ui.request_from_json(text)
        if held["problems"]:
            return ([held, keep_shown] + [gr.update()] * (SLOTS * 5)
                    + [ui.problems_html(held["problems"])])
        rows = ui.questions_to_rows(held["questions"])
        locked = bool(held["locked"])
        return ([held, len(rows)] + _fill(rows, len(rows), locked)
                + [ui.locked_html(held["locked"])])

    json_out = [req, shown, *slots, trouble]
    STALE_JSON = [gr.update()] * len(json_out)

    def from_json(rev, seen, text, keep_shown=0):
        """The editor's own sync, dropped when a newer revision already landed."""
        if SLOW_SYNC:
            time.sleep(SLOW_SYNC)
        if not ui.newest(seen, "request", rev):
            return list(STALE_JSON)
        return sync_json(text, keep_shown)

    def load_questions(rev, seen, text, state_text):
        """A preset or a blank starter: fill the rows, the JSON and the error strip."""
        if not ui.newest(seen, "request", rev):
            return [gr.update()] * (len(json_out) + 2)
        return [state_text] + sync_json(text) + [text]

    preset_out = [state, *json_out, questions]
    for btn, (_, preset_state, preset_q) in zip(preset_btns, PLAYGROUND_PRESETS):
        live(btn.click(
            lambda r, s, q=preset_q, st=preset_state: load_questions(r, s, q, st),
            [rev, seen], preset_out, js=STAMP_JSON, show_progress="hidden",
        ))
    live(blank_btn.click(
        lambda r, s: load_questions(r, s, ui.questions_json(ui.BLANK_QUESTIONS), ""),
        [rev, seen], preset_out, js=STAMP_JSON, show_progress="hidden",
    ))

    def from_rows(rev, seen, n, *flat):
        """A row edit is the request now; the JSON follows it."""
        if SLOW_SYNC:
            time.sleep(SLOW_SYNC)
        if not ui.newest(seen, "request", rev):
            return gr.update(), gr.update(), gr.update()
        rows = _rows_from(flat)[: max(0, min(int(n or 0), SLOTS))]
        held = ui.request_from_rows(rows)
        return held, ui.questions_json(held["questions"]), ui.problems_html(held["problems"])

    # always_last: typing fires faster than the round trip, and an answer about a
    # half-typed request must never be the one left on screen
    for field in fields:
        field.input(from_rows, [rev, seen, shown, *fields], [req, questions, trouble],
                    js=STAMP_ROWS, queue=False, show_progress="hidden",
                    trigger_mode="always_last")

    questions.input(from_json, [rev, seen, questions, shown], json_out,
                    js=STAMP_JSON, queue=False, show_progress="hidden",
                    trigger_mode="always_last")

    def add_row(rev, seen, held, n, *flat):
        """Exactly one more row, whatever the hidden slots happen to contain."""
        n = max(0, min(int(n or 0), SLOTS))
        if not ui.newest(seen, "request", rev):
            return [gr.update()] * (SLOTS * 5 + 1)
        if (held or {}).get("locked") or n >= SLOTS:
            return [n] + [gr.update()] * (SLOTS * 5)
        rows = _rows_from(flat)
        rows[n] = dict(ui.BLANK_ROW)
        return [n + 1] + _fill(rows, n + 1)

    add_btn.click(add_row, [rev, seen, req, shown, *fields], [shown, *slots],
                  js=STAMP_ROWS, queue=False, show_progress="hidden")

    def drop_row(index, rev, seen, held, n, *flat):
        n = max(0, min(int(n or 0), SLOTS))
        if not ui.newest(seen, "request", rev):
            return [gr.update()] * (SLOTS * 5 + 4)
        if (held or {}).get("locked"):
            return [held, n] + [gr.update()] * (SLOTS * 5 + 2)
        rows = _rows_from(flat)
        if index < n:
            rows.pop(index)
            rows.append(dict(ui.BLANK_ROW))
            n -= 1
        fresh = ui.request_from_rows(rows[:n])
        return ([fresh, n] + _fill(rows, n)
                + [ui.questions_json(fresh["questions"]), ui.problems_html(fresh["problems"])])

    for i, drop in enumerate(removers):
        drop.click(lambda r, s, held, n, *flat, i=i: drop_row(i, r, s, held, n, *flat),
                   [rev, seen, req, shown, *fields],
                   [req, shown, *slots, questions, trouble],
                   js=STAMP_ROWS, queue=False, show_progress="hidden")

    live(run_btn.click(lambda: None, None, None, queue=False, show_progress="hidden"))
    live(state.submit(lambda: None, None, None, queue=False, show_progress="hidden"))
    if MULTI:
        live(model_in.input(lambda: None, None, None, queue=False, show_progress="hidden"))

    return model_in, {
        "state": state, "slots": slots, "fields": fields, "questions": questions,
        "trouble": trouble, "out": out, "live": live, "req": req, "shown": shown,
        "rev": rev, "seen": seen,
    }


def _flatten(rows: list[dict]) -> list:
    flat = []
    for i in range(SLOTS):
        row = rows[i] if i < len(rows) else ui.BLANK_ROW
        flat += [row["name"], row["type"], row["ask"], row["options"]]
    return flat


def home_tab() -> dict:
    """Static: the headline figures and where to go next. Nothing here calls a model.

    The in-app cards are buttons wearing the card's markup: a rendered tab button is not
    something to hunt for on a phone, where the strip hides behind a menu."""
    for block in ui.home_blocks(DATA):
        gr.HTML(block)
    cards = {}
    with gr.Row(elem_classes="blk-cards"):
        for slug, title, line in ui.HOME_LINKS:
            with gr.Column(min_width=180, elem_classes="blk-tilewrap"):
                gr.HTML(ui.card_face(title, line))
                cards[slug] = gr.Button(title, elem_classes="blk-tile-hit")
    gr.HTML(ui.home_external())
    return cards


def ask_tab():
    """Free-form asks, drafted into a request and decided. Returns the hand-off button."""
    gr.HTML(ui.ask_note_line())
    model_in = model_switch()
    with gr.Row(elem_classes="blk-askrow"):
        ask_box = gr.Textbox(
            show_label=False, placeholder=ui.ASK["placeholder"],
            lines=1, max_lines=1, scale=8, min_width=120,
            elem_id="ask-input", elem_classes="blk-ask",
        )
        ask_btn = gr.Button(ui.ASK["button"], variant="primary", scale=0,
                            min_width=88, elem_id="ask-run",
                            elem_classes="blk-askgo")
    note = gr.HTML(elem_classes="blk-asknote")
    with gr.Row(equal_height=False, elem_classes="blk-askout"):
        with gr.Column(scale=5, elem_classes="blk-side"):
            drafted = gr.HTML(elem_classes="blk-draftbox")
            hand_off = gr.Button(ui.ASK["open"], size="sm", visible=False,
                                 elem_id="ask-open", elem_classes="blk-handoff")
            held = gr.State(None)
            rev = gr.Textbox("0", visible=False)
            seen = gr.State({})
        with gr.Column(scale=7, elem_classes="blk-main"):
            out = gr.HTML()

    def ask_flow(rev, seen, ask_text, model):
        """Draft the request, show what it says, then decide with it.

        Every step asks whether this is still the ask the panel is waiting for. A model
        chosen while one is in flight owns the panel from that moment, and an answer
        computed for the model it replaced is dropped instead of painted underneath."""
        rev = ui.as_rev(rev)
        ui.newest(seen, "ask", rev)

        def mine() -> bool:
            return ui.applied(seen, "ask") == rev

        if not mine():
            # a generator that yields nothing at all empties its own outputs
            yield {out: gr.skip()}
            return
        # gradio 6.28 drops a later visible=True if an earlier yield repeats visible=False
        yield {note: ui.ask_status_html(ui.ASK["drafting"])}
        try:
            req = author.draft(ask_text)
        except author.AuthorError as exc:
            if mine():
                yield {note: ui.ask_error_html(str(exc))}
            return
        except Exception:  # noqa: BLE001 - the drafter is best effort
            traceback.print_exc()
            if mine():
                yield {note: ui.ask_error_html(ui.ASK["failed"])}
            return
        if not mine():
            return
        # the answer is the model's as much as the draft's; carry both together
        req = dict(req, model=model)
        yield {
            note: "",
            drafted: ui.draft_summary(req["questions"]),
            held: req,
            hand_off: gr.update(visible=True),
            out: ui.pending_html(len(req["questions"]), ui.ASK["deciding"]),
        }
        answers = ui.run_questions(req["state"], req["questions"], model,
                                   author=req["author"])
        if not mine():
            return
        yield {out: answers, held: dict(req, answers=answers)}

    for trigger in (ask_box.submit, ask_btn.click):
        trigger(None, None, rev, js=BUMP_ASK, queue=False,
                show_progress="hidden").then(
            ask_flow, [rev, seen, ask_box, model_in],
            [note, drafted, held, hand_off, out], show_progress="minimal")

    def redecide(rev, seen, req, model):
        """A different model has to answer the same draft, not inherit the old answer."""
        if not req:
            return gr.update(), None
        answers = ui.run_questions(req["state"], req["questions"], model,
                                   author=req.get("author"))
        if ui.applied(seen, "ask") != ui.as_rev(rev):
            return gr.update(), gr.update()  # a newer selection owns the panel now
        return answers, dict(req, answers=answers, model=model)

    def picked_model(rev, seen, req):
        """Choosing a model is the newest thing the panel knows about."""
        ui.newest(seen, "ask", rev)
        if not req:
            return ui.stale_html(ui.ASK["stale"])
        return ui.pending_html(len(req["questions"]), ui.ASK["deciding"])

    if MULTI:
        model_in.input(None, None, rev, js=BUMP_ASK, queue=False,
                       show_progress="hidden").then(
            picked_model, [rev, seen, held], out, queue=False, show_progress="hidden",
        ).then(redecide, [rev, seen, held, model_in], [out, held],
               show_progress="minimal")

    return model_in, {"ask": ask_box, "hand_off": hand_off, "held": held, "out": out,
                      "rev": rev, "seen": seen, "model": model_in}


# --- use cases ----------------------------------------------------------------------


def use_case_tab(case: examples.UseCase):
    """Returns the switcher and the answer column."""

    def run(state: str, model: str):
        return ui.run_use_case(case, state, model)

    gr.HTML(ui.use_case_note(case))
    with gr.Row(equal_height=False):
        with gr.Column(scale=5, elem_classes="blk-side"):
            model_in = model_switch()
            with gr.Row(elem_classes="blk-chips"):
                btns = [gr.Button(ex.label, size="sm") for ex in case.examples]
            if ui.demo_label():
                gr.HTML(ui.demo_label())
            state = gr.Textbox(
                value=case.examples[0].state,
                label=case.state_label,
                lines=case.state_lines,
                max_lines=26,
            )
            run_btn = gr.Button("Decide", variant="primary")
            gr.HTML(ui.use_case_details(case))
        with gr.Column(scale=7, elem_classes="blk-main"):
            out = gr.HTML(ui.run_use_case(case, case.examples[0].state, prefer="saved"))

    def live(trigger):
        return trigger.then(
            lambda: ui.pending_html(len(case.questions)), None, out,
            queue=False, show_progress="hidden",
        ).then(run, [state, model_in], out, show_progress="minimal")

    for btn, ex in zip(btns, case.examples):
        live(btn.click(lambda s=ex.state: s, outputs=state, show_progress="hidden"))
    live(run_btn.click(lambda: None, None, None, queue=False, show_progress="hidden"))
    if MULTI:
        live(model_in.input(lambda: None, None, None, queue=False, show_progress="hidden"))
    return model_in, out


# --- results and how it works -------------------------------------------------------


def results_tab():
    for block in ui.results_blocks(DATA):
        gr.HTML(block)


def how_tab():
    for block in ui.how_blocks():
        gr.HTML(block)


# --- api ----------------------------------------------------------------------------


def systemone(state: str, questions: dict, temperature: float | None = None,
              model: str | None = None) -> dict:
    """TypeSafe /v1/systemone-shaped endpoint: {state, questions} -> {answers, meta}.
    `questions` may also arrive as a JSON string; a request blink can't answer comes back as its reason."""
    if isinstance(questions, str):
        try:
            questions = json.loads(questions)
        except json.JSONDecodeError as exc:
            raise gr.Error(f"questions is not valid JSON: {exc}") from None
    try:
        return blink.decide(as_state(state), questions, temperature=temperature, model=model)
    except blink.BlinkError as exc:
        raise gr.Error(str(exc)) from None


def ask_api(ask: str, model: str | None = None) -> dict:
    """/v1/ask: a free-form ask -> the request that was drafted for it, and the decision.

    The draft is the only step that generates text; `usage` is blink's own pass."""
    try:
        req = author.draft(ask)
    except author.AuthorError as exc:
        raise gr.Error(str(exc)) from None
    try:
        out = blink.decide(req["state"], req["questions"], model=model)
    except blink.BlinkError as exc:
        raise gr.Error(str(exc)) from None
    return {
        "draft": {"state": req["state"], "questions": req["questions"], "author": req["author"]},
        "answers": out["answers"],
        "usage": out["meta"],
    }


# --- assembly -----------------------------------------------------------------------


# --- deep links ---------------------------------------------------------------------

TAB_LABEL = dict(ui.TABS)

# Read the address once on load. Nothing here hunts for a rendered tab button: at
# narrow widths gradio hides the overflow tabs behind a menu and there is nothing to
# find, so every move is made in python with the tab's own id.
READ_URL = r"""() => {
  window.__blinkRev = window.__blinkRev || 0;
  window.__blinkAsk = window.__blinkAsk || 0;
  window.__blinkLast = window.__blinkLast || '';
  return [window.location.search + window.location.hash];
}"""


def _stamp(surface: str) -> str:
    """js for a listener whose first input is the hidden revision box.

    A revision is taken in the browser, at the instant of the keystroke or the click
    that changed something, and travels out with that one round trip. Nothing here
    reads the page: a click that changes no value never reaches this, so opening the
    editor or putting the caret in a row cannot make either surface the newer one."""
    return ("(...a) => { window.__blinkLast = %s;"
            " window.__blinkRev = (window.__blinkRev || 0) + 1;"
            " a[0] = String(window.__blinkRev); return a; }") % json.dumps(surface)


# a row edit, and a load whose request the editor holds verbatim
STAMP_ROWS = _stamp("rows")
STAMP_JSON = _stamp("json")

# read at the head of the Decide chain, so it is ordered against what follows
READ_TOUCH = r"""() => [window.__blinkLast || '', String(window.__blinkRev || 0)]"""

# the ask panel counts its own: a new ask, or a model chosen while one is in flight
BUMP_ASK = r"""() => String(window.__blinkAsk = (window.__blinkAsk || 0) + 1)"""

# same count, taken by a selector on another tab, whose own value comes first
BUMP_ASK_ARG = r"""(...a) => {
  a[1] = String(window.__blinkAsk = (window.__blinkAsk || 0) + 1);
  return a;
}"""

# Rewrite the address so a visitor can copy it, and tell an embedding page too. Replace,
# never push: a tab click should not fill the back button.
SYNC_URL = r"""(tab, sub) => {
  if (!tab) return;
  const url = new URL(window.location.href);
  url.searchParams.set('tab', tab);
  if (sub && tab === %(case_tab)s) {
    url.searchParams.set('case', sub);
  } else {
    url.searchParams.delete('case');
  }
  const next = url.pathname + url.search;
  if (next !== window.location.pathname + window.location.search || window.location.hash) {
    window.history.replaceState(null, '', next);
  }
  try {
    window.parent.postMessage(
      { queryString: url.search.replace(/^\?/, ''), hash: '' },
      'https://huggingface.co'
    );
  } catch (e) { /* not embedded, or a parent that does not listen */ }
}""" % {"case_tab": json.dumps(ui.CASE_TAB)}

TOP_BY_LABEL = {label: slug for slug, label in ui.TABS}
CASE_BY_TITLE = {case.title: case.key for case in examples.USE_CASES}


def picked_tab(evt: gr.SelectData) -> str:
    """The id of the tab just selected, whether it was clicked or taken from the menu."""
    value = getattr(evt, "value", None)
    if isinstance(value, str) and value in TOP_BY_LABEL:
        return TOP_BY_LABEL[value]
    index = getattr(evt, "index", None)
    if isinstance(index, int) and 0 <= index < len(ui.TAB_IDS):
        return ui.TAB_IDS[index]
    return ui.DEFAULT_TAB


def picked_case(evt: gr.SelectData) -> str:
    value = getattr(evt, "value", None)
    if isinstance(value, str) and value in CASE_BY_TITLE:
        return CASE_BY_TITLE[value]
    index = getattr(evt, "index", None)
    if isinstance(index, int) and 0 <= index < len(ui.CASE_IDS):
        return ui.CASE_IDS[index]
    return ui.CASE_IDS[0]


def into_playground(rev, seen, draft, model=None):
    """Put the draft in the playground's own boxes, with the answer it already has.

    The request enters through the same gate as pasted JSON: the rows are a view of it
    and never the thing it is rebuilt from. The answer rides along only when it is the
    one this draft was given by the model now selected."""
    if not draft or not ui.newest(seen, "request", rev):
        return [gr.update()] * (6 + SLOTS * 5)
    ui.newest(seen, "answers", rev)
    held = ui.request_from_questions(draft["questions"])
    rows = ui.questions_to_rows(held["questions"])
    locked = bool(held["locked"])
    answers = draft.get("answers")
    if model is not None and draft.get("model") not in (None, model):
        answers = None
    return [held, len(rows), draft["state"],
            ui.questions_json(held["questions"]),
            ui.problems_html(held["problems"]) + ui.locked_html(held["locked"]),
            answers or ui.stale_html(),
            *_fill(rows, len(rows), locked)]


def open_from_url(raw: str):
    """?tab=results, #results or ?tab=use-cases&case=nextclick -> the tab it names.

    ?ask=… pre-fills the ask box as well; it is never run for the visitor."""
    tab, case = ui.parse_deep_link(raw)
    return (
        gr.update(selected=tab),
        gr.update(selected=case) if case else gr.update(),
        gr.update(value=ui.parse_ask(raw)),
    )


def build() -> gr.Blocks:
    with gr.Blocks(title=TITLE, analytics_enabled=False, **BLOCKS_LOOK) as demo:
        mast = gr.HTML(ui.masthead())
        switches, panels = [], []
        ask = None
        with gr.Tabs() as tabs:
            with gr.Tab(TAB_LABEL[ui.HOME_TAB], id=ui.HOME_TAB):
                cards = home_tab()
            with gr.Tab(TAB_LABEL["playground"], id="playground"):
                first_switch, pg = playground_tab()
                switches.append(first_switch)
                panels.append(pg["out"])
            if author.enabled():
                with gr.Tab(TAB_LABEL[ui.ASK_TAB], id=ui.ASK_TAB):
                    ask_switch, ask = ask_tab()
                    switches.append(ask_switch)
                    panels.append(ask["out"])
            with gr.Tab(TAB_LABEL[ui.CASE_TAB], id=ui.CASE_TAB):
                with gr.Tabs() as case_tabs:
                    for case in examples.USE_CASES:
                        with gr.Tab(case.title, id=case.key):
                            case_switch, case_out = use_case_tab(case)
                            switches.append(case_switch)
                            panels.append(case_out)
            with gr.Tab(TAB_LABEL["results"], id="results"):
                results_tab()
            with gr.Tab(TAB_LABEL["how-it-works"], id="how-it-works"):
                how_tab()
        gr.HTML(ui.footer())
        here = gr.Textbox(visible=False)
        at_tab = gr.Textbox(ui.DEFAULT_TAB, visible=False)
        at_case = gr.Textbox("", visible=False)
        ask_box = ask["ask"] if ask else None
        opened = ([tabs, case_tabs, at_case, at_tab, mast, *pg["slots"]]
                  + ([ask_box] if ask_box is not None else []))

        def go(slug, model):
            """Moving is one call; putting the rows back is the next one.

            A Tabs update remounts the tab's children from the page's own config, so a
            restore sent with it is overwritten by the mount that follows."""
            return [gr.update(selected=slug), slug,
                    ui.masthead(model, drafting=slug == ui.ASK_TAB)]

        def restore_rows(n, held):
            return _restore(n, bool((held or {}).get("locked")))

        move_in = [first_switch]
        move_out = [tabs, at_tab, mast]
        back_in = [pg["shown"], pg["req"]]

        def opened_by_url(raw, model, n, held):
            """A deep link lands on a tab without anyone selecting it.

            Both halves of the address are set here: leaving the case state empty makes
            the sync below drop the very case the link asked for."""
            tab, case = ui.parse_deep_link(raw)
            moved = go(tab, model)
            out = [moved[0], gr.update(selected=case) if case else gr.update(),
                   case or "", *moved[1:], *_restore(n, bool((held or {}).get("locked")))]
            return out + [gr.update(value=ui.parse_ask(raw))] if ask_box is not None else out

        demo.load(opened_by_url, [here, *move_in, *back_in], opened, js=READ_URL,
                  queue=False, show_progress="hidden")

        # the selection is read from the event, never from a rendered tab button, so the
        # overflow menu a narrow window uses works exactly like the strip
        def tab_selected(model, n, held, evt: gr.SelectData):
            # select fires after the mount, so the restore can ride along here
            return [*go(picked_tab(evt), model)[1:],
                    *_restore(n, bool((held or {}).get("locked")))]

        tabs.select(tab_selected, [*move_in, *back_in], [*move_out[1:], *pg["slots"]],
                    queue=False, show_progress="hidden")

        def case_selected(evt: gr.SelectData):
            return picked_case(evt)

        case_tabs.select(case_selected, None, at_case,
                         queue=False, show_progress="hidden")
        for holder in (at_tab, at_case):
            holder.change(None, [at_tab, at_case], None, js=SYNC_URL,
                          queue=False, show_progress="hidden")

        for slug, btn in cards.items():
            btn.click(lambda m, s=slug: go(s, m), move_in, move_out,
                      queue=False, show_progress="hidden").then(
                restore_rows, back_in, pg["slots"],
                queue=False, show_progress="hidden")

        if ask is not None:
            # two steps: the tab first, then the request. A Tabs update in the same call
            # remounts the rows and loses which of them are showing.
            ask["hand_off"].click(
                lambda m: go("playground", m), move_in, move_out,
                queue=False, show_progress="hidden",
            ).then(
                into_playground,
                [pg["rev"], pg["seen"], ask["held"], ask["model"]],
                [pg["req"], pg["shown"], pg["state"], pg["questions"], pg["trouble"],
                 pg["out"], *pg["slots"]],
                js=STAMP_JSON, show_progress="hidden",
            )
        if MULTI:
            # one choice for the whole page. The tab that was clicked re-runs itself;
            # every other answer stops claiming to be this model's.
            ask_at = switches.index(ask_switch) if ask is not None else -1
            for i, sel in enumerate(switches):
                others = [x for j, x in enumerate(switches) if j != i]
                stale = [p for j, p in enumerate(panels) if j != i]
                # each panel says what to do next in its own words
                words = [ui.ASK["stale"] if j == ask_at else None
                         for j in range(len(panels)) if j != i]
                mine = i == ask_at
                # the cached answer has to go with the panel it was painted in, or the
                # hand-off carries the old model's answer into the playground
                held_in = [] if ask is None or mine else [ask["held"]]
                reach = ([ask["rev"], ask["seen"], *held_in] if held_in else [])

                def switched(v, ask_rev=None, ask_seen=None, draft=None,
                             n=len(others), words=tuple(words), d=mine,
                             carry=bool(held_in)):
                    # one update of its own per selector: gradio's postprocessor takes
                    # the value out of the dict it is handed, so a repeated dict arrives
                    # at every selector after the first with nothing left in it
                    out = ([gr.update(value=v) for _ in range(n)]
                           + [ui.masthead(v, drafting=d)]
                           + [ui.stale_html(w) for w in words])
                    if carry:
                        # an ask still in flight was started for the model just replaced
                        ui.newest(ask_seen, "ask", ask_rev)
                        out.append(dict(draft, answers=None) if draft else draft)
                    return out

                sel.input(
                    switched,
                    [sel, *reach],
                    others + [mast] + stale + held_in,
                    js=BUMP_ASK_ARG if reach else None,
                    queue=False,
                    show_progress="hidden",
                )
        try:
            gr.api(systemone, api_name="v1_systemone")
            if author.enabled():
                gr.api(ask_api, api_name="v1_ask")
        except AttributeError:  # gradio without gr.api
            _api_fallback()
    return demo


def _api_fallback() -> None:
    with gr.Row(visible=False):
        s, q, o = gr.Textbox(), gr.JSON(), gr.JSON()
        gr.Button().click(
            lambda a, b: systemone(a, b), [s, q], o, api_name="v1_systemone"
        )
        if author.enabled():
            a, m, ao = gr.Textbox(), gr.Textbox(), gr.JSON()
            gr.Button().click(
                lambda x, y: ask_api(x, y or None), [a, m], ao, api_name="v1_ask"
            )


if __name__ == "__main__":
    # show_api was dropped from launch() in Gradio 6.
    extra = {"show_api": True} if "show_api" in inspect.signature(gr.Blocks.launch).parameters else {}
    # Spaces turns server-side rendering on by default; in that mode the custom CSS is injected without
    # Gradio's selector scoping, so the layout rules lose to component defaults. Render client-side, as tested.
    if "ssr_mode" in inspect.signature(gr.Blocks.launch).parameters:
        extra["ssr_mode"] = False
    build().queue(max_size=32).launch(
        server_name=os.environ.get("BLINK_HOST")
        or os.environ.get("GRADIO_SERVER_NAME")
        or ("0.0.0.0" if os.environ.get("SPACE_ID") else "127.0.0.1"),
        server_port=int(os.environ.get("BLINK_PORT", "7860")),
        **extra,
        **LAUNCH_LOOK,
    )
