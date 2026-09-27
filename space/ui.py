"""Every piece of HTML the interface shows, with no dependency on gradio.

app.py drops these strings into gradio components; build_static.py writes the same
strings into a static page. Both call the same blink.decide, so a bundled request is
rendered by one code path whichever way it is served.
"""

from __future__ import annotations

import hashlib
import json
import urllib.parse

import api_doc
import blink
import examples
import results
import screens

DATA = results.load()
PLAYGROUND_PRESETS = examples.PLAYGROUND_PRESETS

# Read only by the mock engine so the bundled examples look realistic without a GPU.
BIAS_BY_STATE = {
    ex.state: examples.bias_for(case.key, ex.label)
    for case in examples.USE_CASES
    for ex in case.examples
}
BIAS_BY_STATE[examples.PLAYGROUND_STATE] = examples.PLAYGROUND_BIAS


# --- deep links -------------------------------------------------------------------
# (slug, tab label). The slug is what a link carries; the label is what the tab shows.
TABS = (
    ("home", "Home"),
    ("use-cases", "Use cases"),
    ("ask", "Ask"),
    ("playground", "Playground"),
    ("results", "Results"),
    ("how-it-works", "How it works"),
    ("api", "API"),
)
TAB_IDS = tuple(slug for slug, _ in TABS)
DEFAULT_TAB = TAB_IDS[0]  # home
CASE_TAB = "use-cases"
ASK_TAB = "ask"
HOME_TAB = "home"
API_TAB = "api"
# the screenshot case sits last, after the text next click, where a model that reads screens is served
CASE_IDS = tuple(case.key for case in examples.USE_CASES) + ((screens.KEY,) if screens.available() else ())
DEFAULT_CASE = CASE_IDS[0]


def parse_deep_link(raw) -> tuple[str, str | None]:
    """A location's search and hash -> (tab slug, case key or None).

    "?tab=results", "#results" and "?tab=use-cases&case=nextclick" all resolve; anything
    else falls back to the first tab, so a stale link still opens the page.
    """
    if not isinstance(raw, str):
        return DEFAULT_TAB, None
    search, _, fragment = raw.partition("#")
    params = urllib.parse.parse_qs(search.lstrip("?"))
    asked = (params.get("tab") or [""])[0].strip().lower()
    fragment = fragment.strip().lower()
    tab = asked if asked in TAB_IDS else (fragment if fragment in TAB_IDS else "")
    case = (params.get("case") or [""])[0].strip().lower()
    case = case if case in CASE_IDS else None
    if case and not tab:
        tab = CASE_TAB  # a case on its own is still a use-case link
    if not tab and (params.get("ask") or [""])[0].strip():
        tab = ASK_TAB  # an ask link lands where the ask box is
    return (tab or DEFAULT_TAB), (case if tab == CASE_TAB else None)


def parse_shot(raw) -> str | None:
    """A Screen click link's preset: "?tab=use-cases&case=screen&shot=done" -> "done".

    Only a link that opens Screen click carries one; an unknown shot falls back to the first preset."""
    if parse_deep_link(raw)[1] != screens.KEY:
        return None
    search = raw.partition("#")[0]
    asked = (urllib.parse.parse_qs(search.lstrip("?")).get("shot") or [""])[0].strip().lower()
    if not asked:
        return None
    keys = [shot.key for shot in screens.presets()]
    return asked if asked in keys else keys[0]


def esc(s) -> str:
    return results._e(s)


# --- the project outside this Space -------------------------------------------------
# Every link to the code and the docs is built from here: beside the name on every tab,
# beside the API tab's steps and in How it works. Each opens in a new tab.
GITHUB_URL = "https://github.com/thegovind/blink"
DOCS_URL = "https://thegovind.github.io/blink/"
PROJECT_LINKS = (("GitHub", GITHUB_URL), ("Docs", DOCS_URL))
# each model's own card on the Hub: its weights, their licence and its full story
MODEL_CARDS = tuple((name, f"https://huggingface.co/thegovind/{name}")
                    for name in ("blink-4b", "blink-27b", "blink-mimo-9b"))


def links_html(links=PROJECT_LINKS, cls: str = "blk-links") -> str:
    items = "".join(
        f'<a href="{esc(href)}" target="_blank" rel="noopener">{esc(label)}</a>'
        for label, href in links
    )
    return f'<div class="{cls}">{items}</div>'


# --- engine-dependent chrome --------------------------------------------------------


def engine_name() -> str:
    """The engine actually in use, not the one the environment asked for."""
    return getattr(blink.engine(), "name", "")


def short_model(model_id: str) -> str:
    """What the switcher shows: the repo name, without the owner."""
    return str(model_id).split("/")[-1]


def model_choices() -> list[tuple[str, str]]:
    """(label, id) for every model this deployment serves, the default first."""
    return [(short_model(m), m) for m in blink.models()]


def mock_banner() -> str:
    if engine_name() != "mock":
        return ""
    return (
        '<div class="blk-warn blk-mock">Mock mode: these probabilities are fabricated.</div>'
    )


DEMO_LABEL = "Saved examples only \u00b7 try edits where the model is live."
LIVE_LABEL = "First example is saved \u00b7 edits and clicks run the live model."
LIVE_UNAVAILABLE = "The live model couldn't run. Try again."


def demo_label() -> str:
    """Sits above the inputs: saved-only hosting, or saved examples plus a live model."""
    kind = engine_name()
    if kind == "replay":
        return f'<p class="blk-demo">{esc(DEMO_LABEL)}</p>'
    if kind == "hybrid":
        return f'<p class="blk-demo">{esc(LIVE_LABEL)}</p>'
    return ""


def footer() -> str:
    return '<p class="blk-foot">Weights are for non-commercial research and evaluation; app code is Apache-2.0.</p>'


def error_html(msg: str) -> str:
    return f'<div class="blk-warn">{esc(msg)}</div>'


# Only Screen click sends a screenshot: every other tab, and the Space's API, stay text-only
# even where blink-mimo-9b's vision tower is on (BLINK_VISION=1).
TEXT_ONLY_UI = "Screenshots go in Screen click."
TEXT_ONLY_API = "this Space's API reads text only"


def carries_image(state) -> bool:
    """Whether a state holds a data:image/ URI, well formed or not."""
    return blink.contains_image_uri(state)


def notice_html(msg: str) -> str:
    return f'<div class="blk-notice"><span class="blk-dotmark"></span><p>{esc(msg)}</p></div>'


# --- answer rendering ---------------------------------------------------------------


def _rows(ans: dict, q: dict, delay: int) -> str:
    probs = ans["probabilities"]
    legend = ans.get("legend", {})
    best = ans.get("choice") or ("yes" if ans.get("noul", 0) >= 0.5 else "no")
    out = []
    for i, (key, p) in enumerate(probs.items()):
        label = f"Level {key} · {legend[key]}" if legend else key
        win = " win" if key == best else ""
        out.append(
            f'<li class="blk-opt{win}" style="--d:{delay}ms;--i:{i}">'
            f'<span class="k" title="{esc(label)}">{esc(label)}</span>'
            f'<span class="track"><i style="width:{max(p * 100, 0.6):.1f}%"></i></span>'
            f'<span class="v">{p * 100:.1f}%</span></li>'
        )
    return "".join(out)


def _headline(ans: dict) -> tuple[str, str]:
    if ans["type"] == "choice":
        return esc(ans["choice"]), f"confidence {ans['confidence']:.0%}"
    if ans["type"] == "noul":
        p = ans["noul"]
        return ("Yes" if p >= 0.5 else "No"), f"p(yes) = {p:.3f}"
    top = len(ans["probabilities"]) - 1
    return f"{ans['score']:.2f}", f"of {top} · most likely level {ans['choice']}"


def timing_html(meta: dict) -> str:
    """How long the answer took, and whether it is a saved run or a live one."""
    if meta.get("engine") == "replay":
        return f'<span><b>{meta["latency_ms"]:.1f}</b> ms recorded call \u00b7 saved run</span>'
    if meta.get("model_ms") is not None:
        return f'<span><b>{meta["model_ms"]:.1f}</b> ms model time \u00b7 live</span>'
    return f'<span><b>{meta["latency_ms"]:.1f}</b> ms</span>'


def answered_by_html(meta: dict) -> str:
    return (
        f'<span class="blk-by"><b>{esc(short_model(meta["model"]))}</b></span>'
        if meta.get("model")
        else ""
    )


def render_answers(state, questions: dict, out: dict, author: dict | None = None) -> str:
    blocks = []
    for i, (qkey, q) in enumerate(questions.items()):
        ans = out["answers"][qkey]
        head, sub = _headline(ans)
        delay = 60 + i * 90
        instr = esc(q.get("instructions", ""))
        blocks.append(
            f'<section class="blk-q" style="--d:{delay}ms">'
            f'<div class="blk-q-head"><p>{instr}</p>'
            f'<div class="blk-q-meta"><h4>{esc(qkey)}</h4>'
            f'<span class="blk-kind">{esc(type_label(ans["type"]))}</span></div></div>'
            f'<div class="blk-headline"><b>{head}</b>'
            f'<span class="blk-sub">{esc(sub)}</span></div>'
            f'<ul class="blk-opts">{_rows(ans, q, delay)}</ul>'
            "</section>"
        )
    meta = out["meta"]
    timing = timing_html(meta)
    answered_by = answered_by_html(meta)
    stats = (
        '<div class="blk-stats">'
        + answered_by
        + timing
        + f'<span><b>{meta["input_tokens"]:,}</b> input tokens</span>'
        f'<span><b>{meta["generated_tokens"]}</b> generated</span>'
        f'<span><b>{len(questions)}</b> questions</span>'
        f'<span>temperature <b>{meta["temperature"]:g}</b></span>'
        f"</div>"
    )
    raw = json.dumps({"state": state, **out}, indent=2, ensure_ascii=False)
    more = (
        '<details class="blk-more"><summary>Raw response</summary>'
        f'<div class="blk-pre">{esc(raw)}</div></details>'
    )
    foot = f'<div class="blk-answer-foot">{stats}{drafted_line(author)}{more}</div>'
    return f'<div class="blk-answers">{"".join(blocks)}</div>{foot}'


def render_verdict(tone: str, headline: str, detail: str) -> str:
    """The policy runs for real; nothing downstream of it does, so the box says so."""
    return (
        f'<div class="blk-verdict {esc(tone)}"><span class="blk-dotmark"></span>'
        f'<div><h3>{esc(headline)}<em class="blk-tag">simulated</em></h3>'
        f"<p>{esc(detail)}</p></div></div>"
    )


# --- the question form ------------------------------------------------------------
# Every string the form shows, in one place.
FORM = {
    "state": "State",
    "state_hint": "The text to judge.",
    "questions": "Questions",
    "questions_hint": "What to decide about it.",
    "blank": "Blank",
    "blank_ask": "What should this decide?",
    "blank_yes": "The first option",
    "blank_no": "The second option",
    "name": "Name",
    "type": "Type",
    "ask": "Question",
    "options": "Options",
    "add": "Add question",
    "remove": "Remove",
    "more": "Answer options",
    "json": "Edit as JSON",
    "choice_hint": "One per line: name: description",
    "noul_hint": "Optional. true: … and false: …",
    "score_hint": "One level per line, lowest first",
}

PROBLEM = {
    "json": "Invalid JSON.",
    "shape": "Use an object with named questions.",
    "name": "Every question needs a name.",
    "dupe": "Two questions are called {name}.",
    "few": "{name}: add at least two options.",
    "many": "{name}: use up to {cap} options.",
    "levels": "{name}: add 2 to 10 levels.",
    "type": "{name}: choose a valid type.",
    "empty": "Add a question.",
    "object": "{name}: must be an object.",
    "options": "{name}: add a list of options.",
}

# shown when the JSON holds a request the rows cannot edit without changing it
LOCKED = {
    "over": "Showing {shown} of {total} rows. Edit as JSON.",
    "shape": "This request can't be shown as rows. Edit as JSON.",
}

# (what the form shows, what the wire calls it)
Q_TYPES = (("choice", "choice"), ("yes/no", "noul"), ("score", "score"))
WIRE_TYPES = tuple(wire for _, wire in Q_TYPES)


def type_label(wire) -> str:
    """What a question type is called on screen: the same word the form's picker uses."""
    return next((shown for shown, w in Q_TYPES if w == wire), str(wire or ""))


MAX_FORM_QUESTIONS = 8
BLANK_ROW = {"name": "", "type": "choice", "ask": "", "options": ""}
_SLUG = __import__("re").compile(r"[^a-z0-9]+")


def slug(text) -> str:
    """"Pay invoice" -> "pay_invoice"; a name that is already a slug is left alone."""
    return _SLUG.sub("_", str(text or "").strip().lower()).strip("_")


def _split_option(line: str) -> tuple[str, str]:
    key, sep, desc = line.partition(":")
    key, desc = key.strip(), desc.strip()
    if sep and key and desc:
        return slug(key) or slug(line), desc
    return slug(line), line


def parse_options(text: str, wire: str):
    """The options box -> whatever `criteria` that question type takes."""
    lines = [ln.strip() for ln in str(text or "").splitlines() if ln.strip()]
    if wire == "score":
        return lines
    pairs = [_split_option(ln) for ln in lines if ln]
    keys = [k for k, _ in pairs]
    if len(set(keys)) != len(keys) or not all(keys):
        pairs = [(ln, ln) for ln in lines]  # names collided; keep every option instead
    return dict(pairs)


def format_options(q: dict) -> str:
    """The reverse, so a question loaded from JSON comes back out unchanged."""
    crit = q.get("criteria")
    if q.get("type") == "score":
        return "\n".join(str(c) for c in (crit or []))
    if isinstance(crit, list):  # a choice may list bare options; each line is one
        return "\n".join(str(c) for c in crit)
    if not isinstance(crit, dict):
        return ""
    return "\n".join(f"{k}: {v}" if v else str(k) for k, v in crit.items())


def questions_to_rows(questions: dict) -> list[dict]:
    rows = []
    for name, q in (questions or {}).items():
        q = q if isinstance(q, dict) else {}
        rows.append(
            {
                "name": str(name),
                "type": q.get("type") if q.get("type") in WIRE_TYPES else "choice",
                "ask": str(q.get("instructions") or ""),
                "options": format_options(q),
            }
        )
    return rows[:MAX_FORM_QUESTIONS]


def rows_to_questions(rows) -> tuple[dict, list[str]]:
    """The form -> the wire format, with everything wrong about it said out loud."""
    questions: dict = {}
    problems: list[str] = []
    for row in rows or []:
        name = str(row.get("name") or "").strip()
        ask = str(row.get("ask") or "").strip()
        options = str(row.get("options") or "").strip()
        wire = row.get("type") if row.get("type") in WIRE_TYPES else None
        if not (name or ask or options):
            continue
        if not name:
            problems.append(PROBLEM["name"])
            continue
        if wire is None:
            problems.append(PROBLEM["type"].format(name=name))
            continue
        if name in questions:
            problems.append(PROBLEM["dupe"].format(name=name))
            continue
        crit = parse_options(row.get("options"), wire)
        if wire == "choice":
            if len(crit) < 2:
                problems.append(PROBLEM["few"].format(name=name))
                continue
            if len(crit) > blink.MAX_OPTIONS:
                problems.append(PROBLEM["many"].format(name=name, cap=blink.MAX_OPTIONS))
                continue
        if wire == "score" and not 2 <= len(crit) <= 10:
            problems.append(PROBLEM["levels"].format(name=name))
            continue
        q: dict = {"type": wire, "instructions": ask}
        if crit:
            q["criteria"] = crit
        questions[name] = q
    if not questions and not problems:
        problems.append(PROBLEM["empty"])
    return questions, problems


def questions_from_json(text) -> tuple[dict, list[str]]:
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {}, [PROBLEM["json"]]
    if not isinstance(parsed, dict):
        return {}, [PROBLEM["shape"]]
    return parsed, []


def _criteria_size(crit) -> int:
    if isinstance(crit, dict):
        return len(crit)
    if isinstance(crit, (list, tuple)):
        return len(crit)
    return 0


def validate_questions(parsed) -> tuple[dict, list[str]]:
    """Check the wire format as written, without passing it through the form.

    The form cannot hold every request a caller may send, so the rows are a view and
    this is the gate: whatever survives here is exactly what runs."""
    if not isinstance(parsed, dict):
        return {}, [PROBLEM["shape"]]
    questions: dict = {}
    problems: list[str] = []
    seen: dict = {}
    for raw_name, q in parsed.items():
        key = str(raw_name)
        name = key.strip()
        if not name:
            problems.append(PROBLEM["name"])
            continue
        # "q" and " q " read as one name; the second must not quietly replace the first
        if name in seen:
            problems.append(PROBLEM["dupe"].format(name=name))
            continue
        seen[name] = key
        if not isinstance(q, dict):
            problems.append(PROBLEM["object"].format(name=name))
            continue
        wire = q.get("type")
        if wire not in WIRE_TYPES:
            problems.append(PROBLEM["type"].format(name=name))
            continue
        crit, size = q.get("criteria"), _criteria_size(q.get("criteria"))
        if wire == "choice":
            if crit is not None and not isinstance(crit, (dict, list, tuple)):
                problems.append(PROBLEM["options"].format(name=name))
                continue
            if size < 2:
                problems.append(PROBLEM["few"].format(name=name))
                continue
            if size > blink.MAX_OPTIONS:
                problems.append(PROBLEM["many"].format(name=name, cap=blink.MAX_OPTIONS))
                continue
        elif wire == "score":
            if not isinstance(crit, (list, tuple)):
                problems.append(PROBLEM["options"].format(name=name))
                continue
            if not 2 <= size <= 10:
                problems.append(PROBLEM["levels"].format(name=name))
                continue
        elif crit is not None and not isinstance(crit, dict):
            problems.append(PROBLEM["options"].format(name=name))
            continue
        questions[key] = q
    if not questions and not problems:
        problems.append(PROBLEM["empty"])
    return questions, problems


def request_from_questions(parsed) -> dict:
    """The one way a wire-format request enters the app, wherever it came from."""
    questions, problems = validate_questions(parsed)
    return {
        "questions": {} if problems else questions,
        "problems": problems,
        "locked": "" if problems else locked_reason(questions),
    }


def request_from_json(text) -> dict:
    """One validated request: what runs, what the rows may edit, and what went wrong."""
    parsed, problems = questions_from_json(text)
    if problems:
        return {"questions": {}, "problems": problems, "locked": ""}
    return request_from_questions(parsed)


def request_from_rows(rows) -> dict:
    """The same shape, built from the rows the visitor can see."""
    questions, problems = rows_to_questions(rows)
    return {
        "questions": {} if problems else questions,
        "problems": problems,
        "locked": "",
    }


def as_rev(value) -> int:
    """The revision the browser stamped on a round trip, as a number."""
    try:
        return int(str(value).strip() or 0)
    except (TypeError, ValueError):
        return 0


def applied(seen, lane: str) -> int:
    """The newest revision this surface has already taken."""
    return as_rev((seen or {}).get(lane, 0))


def newest(seen, lane: str, rev) -> bool:
    """True when `rev` is not older than what the surface already took, and takes it.

    The browser stamps a revision the moment the visitor changes something or loads a
    request, so a round trip held up on the way out still says which version of the page
    it was sent for. One that a later round trip has already overtaken is dropped here,
    rather than writing its result over the newer one."""
    if seen is None:
        return True
    now = as_rev(rev)
    if now < applied(seen, lane):
        return False
    seen[lane] = now
    return True


def _wire_shape(q: dict) -> dict:
    """Only what a row can carry, so the fit test compares like with like."""
    out = {"type": q.get("type"), "instructions": str(q.get("instructions") or "")}
    if q.get("criteria"):
        out["criteria"] = q["criteria"]
    return out


def locked_reason(questions: dict) -> str:
    """Empty when the rows can edit this request; otherwise why they cannot."""
    total = len(questions)
    if total > MAX_FORM_QUESTIONS:
        return LOCKED["over"].format(shown=MAX_FORM_QUESTIONS, total=total)
    for q in questions.values():
        if not isinstance(q, dict) or set(q) - {"type", "instructions", "criteria"}:
            return LOCKED["shape"]
    back, problems = rows_to_questions(questions_to_rows(questions))
    if problems:
        return LOCKED["shape"]
    want = {str(k): _wire_shape(v) for k, v in questions.items()}
    got = {str(k): _wire_shape(v) for k, v in back.items()}
    return "" if want == got else LOCKED["shape"]


def locked_html(reason: str) -> str:
    return f'<p class="blk-locked">{esc(reason)}</p>' if reason else ""


def questions_json(questions: dict) -> str:
    return json.dumps(questions, indent=2, ensure_ascii=False)


def _heading(title: str, hint: str) -> str:
    return f'<p class="blk-label blk-form-head">{esc(title)}<span>{esc(hint)}</span></p>'


def state_heading() -> str:
    return _heading(FORM["state"], FORM["state_hint"])


def form_heading() -> str:
    return _heading(FORM["questions"], FORM["questions_hint"])


def run_questions(state, questions: dict, model: str | None = None,
                  prefer: str = "live", author: dict | None = None) -> str:
    """The form's answer path: a questions dict rather than a blob of JSON."""
    parsed = blink.as_state(state)
    if carries_image(parsed):
        return error_html(TEXT_ONLY_UI)
    try:
        out = blink.decide(parsed, questions, bias=BIAS_BY_STATE.get(str(state).strip()),
                           prefer=prefer, model=model)
    except blink.ReplayMiss as exc:
        return notice_html(str(exc))
    except blink.BlinkError as exc:
        return error_html(str(exc))
    except Exception as exc:
        return _live_failure(exc)
    return mock_banner() + render_answers(parsed, questions, out, author)


def problems_html(problems) -> str:
    if not problems:
        return ""
    items = "".join(f"<li>{esc(p)}</li>" for p in problems)
    return f'<ul class="blk-problems">{items}</ul>'


# --- home ---------------------------------------------------------------------------
HOME = {
    "lede": "Give blink a state and questions. Get a probability for each option.",
    "index": "Decision Index 0.1",
    "jevbench": "JevBench public questions",
    "go": "View tab",
    "open": "Open",
}

# (slug or url, title, one line). A slug switches tab; a url opens in a new tab.
HOME_LINKS = (
    ("use-cases", "Use cases", "Sort requests or choose next steps."),
    ("ask", "Ask a question", "A model drafts options; blink scores."),
    ("playground", "Try it", "Enter a state and questions."),
    ("results", "All results", "Scores and how they were measured."),
    ("how-it-works", "How it works", "How blink scores each option."),
)

HOME_EXTERNAL = (
    ("https://huggingface.co/thegovind/blink-4b", "blink-4b", "Model files on Hugging Face."),
    ("https://huggingface.co/thegovind/blink-27b", "blink-27b", "Model files on Hugging Face."),
    ("https://huggingface.co/thegovind/blink-mimo-9b", "blink-mimo-9b", "Model files on Hugging Face."),
    ("https://huggingface.co/spaces/multimodalart/jev-decision-index",
     "Decision Index", "The benchmark and its live board."),
    ("https://github.com/fstandhartinger/jevbench/issues/81",
     "JevBench", "The open request to measure blink-4b."),
)


def _card(href: str, title: str, line: str, external: bool) -> str:
    """Outside links stay links. In-app cards are buttons; this is only their face."""
    if not external:
        return card_face(title, line)
    return (
        f'<a class="blk-tile out" href="{esc(href)}" target="_blank" rel="noopener noreferrer">'
        f"<b>{esc(title)}</b><span>{esc(line)}</span>"
        f'<em>{esc(HOME["open"])}</em></a>'
    )


def card_face(title: str, line: str) -> str:
    return (
        f'<span class="blk-face"><b>{esc(title)}</b><span>{esc(line)}</span>'
        f'<em>{esc(HOME["go"])}</em></span>'
    )


def home_external() -> str:
    out = "".join(_card(h, t, line, True) for h, t, line in HOME_EXTERNAL)
    return f'<nav class="blk-cards out">{out}</nav>' 


def home_blocks(data: dict | None = None) -> list[str]:
    """A static landing page: the headline figures, then where to go next."""
    d = data or DATA
    di, jb = d["decision_index"], d["jevbench"]
    held = d.get("heldout") or {}
    # the figure leads; the same numbers as a table take over where it cannot fit
    index_pane = (
        f'<p class="blk-eyebrow">{esc(HOME["index"])}</p>'
        + f'<div class="blk-wide">{results.headline_chart(d)}</div>'
        + f'<div class="blk-narrow">{results.headline_table(d)}</div>'
        + f'<p class="blk-note">{esc(di["caption"])}</p>'
    )
    held_pane = (
        f'<p class="blk-eyebrow">{esc(held["label"])}</p>'
        + f'<div class="blk-wide">{results.heldout_chart(d)}</div>'
        + f'<div class="blk-narrow">{results.heldout_table(d)}</div>'
        + f'<p class="blk-note">{esc(results.heldout_note(d))}</p>'
        if held
        else ""
    )
    headline = (
        f'<div class="blk-pair"><div class="blk-half">{index_pane}</div>'
        f'<div class="blk-half">{held_pane}</div></div>'
        if held_pane
        else index_pane
    )
    return [
        f'<p class="blk-lede blk-home-lede">{esc(HOME["lede"])}</p>',
        headline,
        f'<p class="blk-eyebrow">{esc(HOME["jevbench"])}</p>'
        + results.headline_jevbench(d)
        + f'<p class="blk-note">{esc(jevbench_note(jb))}</p>',
    ]


def draft_summary(questions: dict) -> str:
    """What the drafter wrote, read-only and compact: the question, its type, its options."""
    if not questions:
        return ""
    rows = []
    for name, q in questions.items():
        q = q if isinstance(q, dict) else {}
        options = format_options(q).splitlines()
        shown = "".join(f"<li>{esc(line)}</li>" for line in options)
        rows.append(
            f'<li class="blk-draft-q"><div class="blk-q-head">'
            f'<p>{esc(q.get("instructions", ""))}</p>'
            f'<div class="blk-q-meta"><b>{esc(name)}</b>'
            f'<em class="blk-kind">{esc(type_label(q.get("type", "")))}</em></div></div>'
            + (f"<ul>{shown}</ul>" if shown else "")
            + "</li>"
        )
    return f'<ul class="blk-draft">{"".join(rows)}</ul>'


# --- free-form asks ----------------------------------------------------------------
ASK = {
    "title": "Ask",
    "hint": "Ask a question with a few possible answers.",
    "placeholder": "How many r in strawberry",
    "button": "Ask",
    "drafting": "Drafting the question",
    "deciding": "Answering",
    "drafted": "drafted by {model} \u00b7 {tokens} generated tokens \u00b7 {seconds}",
    "failed": "Drafting failed. Try again or enter the question and options yourself.",
    "open": "Open in Playground",
    "stale": "Ask again.",
    "note": "Ask a question. A small model drafts the options; blink gives each a probability.",
}


def ask_heading() -> str:
    return _heading(ASK["title"], ASK["hint"])


def ask_note_line() -> str:
    return f'<p class="blk-note">{esc(ASK["note"])}</p>' 


def ask_status_html(label: str) -> str:
    return f'<p class="blk-ask-note"><span class="blk-run">{esc(label)}</span></p>'


def ask_error_html(message: str) -> str:
    return f'<p class="blk-ask-note blk-ask-bad">{esc(message)}</p>'


def drafted_line(author: dict | None) -> str:
    """The one step on the page that generates text, said out loud under the answer."""
    if not author:
        return ""
    said = ASK["drafted"].format(
        model=author.get("model", ""),
        tokens=author.get("generated_tokens", 0),
        seconds=f"{(author.get('model_ms') or 0) / 1000:.1f} s",
    )
    return f'<p class="blk-drafted" id="ask-provenance">{esc(said)}</p>'


def parse_ask(raw) -> str:
    """?ask=… pre-fills the box; it never runs on its own."""
    if not isinstance(raw, str):
        return ""
    search, _, _ = raw.partition("#")
    return (urllib.parse.parse_qs(search.lstrip("?")).get("ask") or [""])[0].strip()


PENDING_LABEL = "Running"


STALE_LABEL = "Answers don't match this request. Click Decide."


BLANK_QUESTIONS = {
    "label": {
        "type": "choice",
        "instructions": FORM["blank_ask"],
        "criteria": {"yes": FORM["blank_yes"], "no": FORM["blank_no"]},
    }
}


def stale_html(label: str | None = None) -> str:
    """The answer column when what it showed no longer matches the request."""
    return (
        f'<div class="blk-empty"><span class="blk-dotmark"></span>'
        f'<p>{esc(label or STALE_LABEL)}</p></div>'
    )


def pending_html(n: int = 3, label: str | None = None) -> str:
    """Stands in for the answer while the model runs, at roughly the answer's height."""
    cards = "".join(
        f'<section class="blk-q blk-sk-card" style="--d:{i * 80}ms">'
        '<div class="blk-sk blk-sk-kicker"></div>'
        '<div class="blk-sk blk-sk-head"></div>'
        + "".join(f'<div class="blk-sk blk-sk-row" style="--i:{j}"></div>' for j in range(3))
        + "</section>"
        for i in range(max(1, min(n, 6)))
    )
    return (
        f'<div class="blk-answers" aria-busy="true">{cards}</div>'
        f'<div class="blk-stats"><span class="blk-run">{esc(label or PENDING_LABEL)}</span></div>'
    )


def pending_for(questions_text: str) -> str:
    """Same skeleton, sized from whatever the editor currently holds."""
    try:
        parsed = json.loads(questions_text)
        return pending_html(len(parsed) if isinstance(parsed, dict) else 3)
    except (json.JSONDecodeError, TypeError):
        return pending_html()


def _live_failure(exc: Exception) -> str:
    if type(exc).__module__.split(".")[0] == "gradio":
        raise exc  # e.g. a quota message the visitor should see as-is
    import traceback

    traceback.print_exc()
    return notice_html(LIVE_UNAVAILABLE)


# --- the two things a request can be ------------------------------------------------


def run_playground(state: str, questions_text: str, model: str | None = None,
                   prefer: str = "live") -> str:
    try:
        questions = json.loads(questions_text)
    except json.JSONDecodeError as exc:
        return error_html(f"The questions are not valid JSON — {exc}")
    if not isinstance(questions, dict):
        return error_html("Questions must be a JSON object keyed by question name.")
    parsed = blink.as_state(state)
    if carries_image(parsed):
        return error_html(TEXT_ONLY_UI)
    try:
        out = blink.decide(parsed, questions, bias=BIAS_BY_STATE.get(state.strip()),
                           prefer=prefer, model=model)
    except blink.ReplayMiss as exc:
        return notice_html(str(exc))
    except blink.BlinkError as exc:
        return error_html(str(exc))
    except Exception as exc:  # the live model failed: keep gradio's own errors, soften the rest
        return _live_failure(exc)
    return mock_banner() + render_answers(parsed, questions, out)


def run_use_case(case: examples.UseCase, state: str, model: str | None = None,
                 prefer: str = "live") -> str:
    if carries_image(state):
        return error_html(TEXT_ONLY_UI)
    try:
        out = blink.decide(state, case.questions, bias=BIAS_BY_STATE.get(state),
                           prefer=prefer, model=model)
    except blink.ReplayMiss as exc:
        return notice_html(str(exc))
    except blink.BlinkError as exc:
        return error_html(str(exc))
    except Exception as exc:
        return _live_failure(exc)
    tone, headline, detail = case.verdict(out["answers"])
    return mock_banner() + render_verdict(tone, headline, detail) + render_answers(
        state, case.questions, out
    )


# --- tab furniture ------------------------------------------------------------------


def base_of(model_id: str) -> str:
    """The base model a served blink model was fine-tuned from (results.json model.bases)."""
    return (DATA["model"].get("bases") or {}).get(model_id, DATA["model"]["base_small"])


# The claim about generated text is about blink's own pass, so it changes where a
# drafting model writes the request first.
MAST = {
    "lede": "Send a state and typed questions. Get a probability for each option. "
            "No generated text.",
    "chip": "0 generated tokens",
    "ask_lede": "Ask in your own words. A separate model drafts the question, then blink gives each "
                "option a probability in one pass.",
    "ask_chip": "drafting model writes text \u00b7 blink writes no text",
}


def masthead(model: str | None = None, drafting: bool = False) -> str:
    """drafting=True on the tab where another model writes the request first."""
    return (
        '<div class="blk-top">'
        '<div class="blk-toprow">'
        '<h1><span class="blk-word">blink<em class="blk-mark" aria-hidden="true"></em>'
        "</span></h1>"
        + links_html(PROJECT_LINKS, "blk-toplinks")
        + "</div>"
        f'<p class="blk-lede">{esc(MAST["ask_lede"] if drafting else MAST["lede"])}</p>'
        '<div class="blk-meta">'
        f'<span class="key">{esc(model or blink.MODEL_ID)}</span>'
        f'<span>{esc(base_of(model or blink.MODEL_ID))} · LoRA, merged</span>'
        f'<span>{esc(MAST["ask_chip"] if drafting else MAST["chip"])}</span>'
        "</div></div>"
        '<div class="blk-rule"></div>'
    )


def playground_note() -> str:
    return (
        '<p class="blk-note">Try an example or edit the state and questions, then click Decide.</p>'
    )


def use_case_note(case: examples.UseCase) -> str:
    return f'<p class="blk-note">{esc(case.blurb)}</p>'


def use_case_details(case: examples.UseCase) -> str:
    return (
        '<details class="blk-d"><summary>Questions</summary>'
        f'<div class="blk-pre">{esc(json.dumps(case.questions, indent=2))}</div></details>'
        '<details class="blk-d"><summary>Decision rule</summary>'
        f'<div class="blk-pre">{esc(case.source)}</div>'
        + (f"<p>{esc(case.policy_note)}</p>" if case.policy_note else "")
        + "</details>"
    )


# --- results ------------------------------------------------------------------------


SELECTION_NOTE = (
    "The 3,000-request DI-S sample picked the prompt and checkpoints. The full suite picked T4 for "
    "blink-27b, so its full result is post-selection; blink-4b was fixed beforehand."
)

EDITION_MOVED = (
    "The live board moved to 0.2 on 2026-09-24. The public kit cannot build it yet; no 0.2 result here."
)

JEVBENCH_NOTE = (
    "{proxies} are public-item development proxies computed here, not official scores or predictions. "
    "An official score needs held-out, judge and sealed items. No official score, rank, or parity "
    "with Jev or JevK5 is claimed."
)


def jevbench_note(jb: dict) -> str:
    vals = [f"{s['public_estimate']:.1f}" for s in jb["systems"] if s.get("public_estimate") is not None]
    proxies = " and ".join(vals) if len(vals) <= 2 else ", ".join(vals[:-1]) + " and " + vals[-1]
    return JEVBENCH_NOTE.format(proxies=proxies or "The public-items figures")


def heldout_detail(held: dict) -> str:
    """What the set is, from the file: how many tasks, how many items, and their names."""
    tasks = held.get("tasks") or []
    per = held.get("items_per_task")
    bits = []
    if tasks and per:
        bits.append(f"{len(tasks)} tasks, {per} items each.")
    if tasks:
        bits.append(", ".join(t["label"] for t in tasks) + ".")
    return " ".join(bits)


def in_domain_line(di: dict) -> str:
    """The overlap that matters, in the open, next to the headline it qualifies."""
    ref = next((s for s in di["systems"] if s["id"] == di.get("reference")), None)
    ours = [s for s in di["systems"] if s["kind"] == "ours" and s.get("index_language_equalized")]
    if not ref or not ours:
        return ""
    short = ref["name"].split()[0]
    equalized = " and ".join(
        f"{esc(s['name'])} would be at {s['index_language_equalized']:.2f}" for s in ours
    )
    return (
        '<p class="blk-note">blink-27b used the public train splits of ContractNLI, iSarcasmEval, '
        "VAST (all of Language), Amazon ESCI and Humicroedit. With Language set to "
        f"{esc(short)}'s score: {equalized} vs {esc(short)} {ref['index']:.2f}.</p>"
    )


def results_blocks(data: dict | None = None) -> list[str]:
    d = data or DATA
    di, jb, pa = d["decision_index"], d["jevbench"], d["pareto"]
    v02 = d.get("decision_index_v02_partial")
    lat = d.get("latency")
    banner = (
        f'<div class="blk-warn">{esc(d.get("placeholder_note", "Placeholder numbers."))}</div>'
        if d.get("placeholder")
        else ""
    )
    edition = f' <span class="blk-hint">Edition {esc(di["edition"])}.</span>' if di.get("edition") else ""
    out = [
        banner
        + f'<h2 class="blk-h2">{esc(di["label"])}</h2>'
        + in_domain_line(di)
        + f'<p class="blk-note">{esc(SELECTION_NOTE)}</p>',
        results.decision_index_chart(d),
        f'<p class="blk-note">{esc(EDITION_MOVED)}</p>'
        '<details class="blk-d"><summary>About this result</summary>'
        f'<p>{esc(di["caption"])}{edition}</p></details>',
    ]
    held = d.get("heldout")
    if held:
        out += [
            f'<h2 class="blk-h2">{esc(held["label"])}</h2>',
            f'<div class="blk-narrowfig blk-wide">{results.heldout_chart(d)}</div>',
            results.heldout_table(d),
            results.heldout_tasks(d),
            '<details class="blk-d"><summary>About these tasks</summary>'
            f'<p>{esc(results.heldout_note(d))}</p>'
            f'<p>{esc(heldout_detail(held))}</p></details>',
        ]
    if v02:
        out += [
            f'<h2 class="blk-h2">{esc(v02["label"])}</h2>',
            results.v02_partial_chart(d),
            '<details class="blk-d"><summary>About this comparison</summary>'
            f'<p>{esc(v02["caption"])}</p></details>',
        ]
    out += [
        f'<h2 class="blk-h2">{esc(jb["label"])}</h2>',
        results.jevbench_chart(d),
        '<details class="blk-d"><summary>About these scores</summary>'
        f'<p>{esc(jevbench_note(jb))}</p><p>{esc(jb["caption"])}</p></details>',
        f'<h2 class="blk-h2">{esc(pa["label"])}</h2>',
        results.pareto_chart(d),
        '<details class="blk-d"><summary>About this comparison</summary>'
        f'<p>{esc(pa["caption"])}</p></details>',
    ]
    sp = d.get("speed_pareto")
    if sp:
        out += [
            f'<h2 class="blk-h2">{esc(sp["label"])}</h2>',
            results.speed_pareto_chart(d),
            '<details class="blk-d"><summary>About this chart</summary>'
            f'<p>{esc(sp["caption"])}</p></details>',
        ]
    cua = d.get("cua_probe")
    if cua:
        out += [
            f'<h2 class="blk-h2">{esc(cua["label"])}</h2>',
            results.cua_probe_chart(d),
            '<details class="blk-d"><summary>About this chart</summary>'
            f'<p>{esc(cua["caption"])}</p></details>',
        ]
    if lat:
        out.append(results.latency_strip(lat))
    out.append(
        '<details class="blk-d"><summary>How these are measured</summary>'
        f'<p>blink was run on the complete archived 0.1 suite: {esc(di["suite"])}. The headline index '
        "averages 19 panel benchmarks within five equal-weight areas. A request the model cannot answer "
        "counts as wrong, so refusing to answer is never free. Skill measures performance "
        "above each metric's chance baseline; breadth is a shifted geometric mean of the five "
        "area skill scores. blink's rows are local runs of the official kit, not submissions; "
        f'every other 0.1 row is the leaderboard snapshot of {esc(di["leaderboard_snapshot"][:10])}.</p>'
        + (
            f'<p>The 0.2 comparison re-scores {v02["shared"]} of the {v02["panel"]} panel '
            "benchmarks 0.2 uses — the ones it keeps from 0.1 — identically for every "
            "system. It is not a 0.2 score and cannot be compared with one.</p>"
            if v02
            else ""
        )
        + "<p>JevBench is an equal-weight harmonic mean of Intelligence, Calibration, Speed "
        "and Cost. An official score is computed over held-out, judge and sealed items that "
        "only a submitted run reaches. No official JevBench score for blink has been published. "
        "For both systems, the figures shown here cover only public items, were produced with "
        "the same harness, and are kept on their own scale.</p>"
        + (
            f'<p>Latency is {lat["p50_ms"]} ms at the median and {lat["max_ms"]} ms at the '
            f'slowest of {lat["requests"]} bundled requests, '
            f'with {lat["generated_tokens"]} tokens generated.</p>'
            if lat
            else ""
        )
        + "</details>"
    )
    return out


# --- how it works -------------------------------------------------------------------

DEMO_Q = {
    "type": "choice",
    "instructions": "Which team should own this ticket?",
    "criteria": {"billing": "Charges and refunds", "technical": "Errors and outages"},
}

# the model repos' code revision, the one the API tab pins (api_doc.CODE_REVISION)
API_SNIPPET = """# pip install "torch==2.13.0" "transformers==5.17.0" "flash-linear-attention==0.5.2" "accelerate>=1.1.0" safetensors huggingface_hub
import os, sys
from huggingface_hub import hf_hub_download

os.environ["BLINK_MODEL"] = "thegovind/blink-4b"
os.environ["BLINK_REVISION"] = "@REVISION@"
sys.path.insert(0, os.path.dirname(hf_hub_download("thegovind/blink-4b", "blink.py", revision="@REVISION@")))
import blink

out = blink.decide(
    "My card was charged twice this month.",
    {
        "queue": {
            "type": "choice",
            "instructions": "Which team should own this ticket?",
            "criteria": {"billing": "Charges and refunds", "technical": "Errors and outages"},
        },
        "urgent": {"type": "noul", "instructions": "Does this need an answer today?"},
    },
)
print(out["answers"]["queue"]["probabilities"])""".replace("@REVISION@", api_doc.CODE_REVISION)


def how_blocks() -> list[str]:
    items = blink.question_options(DEMO_Q)
    labels = blink.LABEL_POOL[: len(items)]
    rendered = blink.user_message("My card was charged twice.", DEMO_Q, labels, items)
    return [
        '<h2 class="blk-h2">How it works</h2>',
        results.pipeline_figure(),
        '<p class="blk-note">LoRA r16 updates every layer’s attention, DeltaNet and MLP projections; '
        "embeddings, norms and the output head stay frozen. blink-4b and blink-27b ship text-only weights. "
        "blink-mimo-9b keeps MiMo's unchanged vision tower; its checkpoint has no MTP tensors. "
        + (SCREEN["how"].format(title=SCREEN["title"]) if screens.available() else
           "The app uses its text side only, reading option-letter scores without generating text.")
        + "</p>",
        '<p class="blk-note">Jev’s RLCD recipe isn’t public, so it wasn’t copied; this is supervised fine-tuning '
        "on decision data. Full details: "
        + " · ".join(f'<a href="{esc(href)}" target="_blank" rel="noopener">{esc(name)} card</a>'
                     for name, href in MODEL_CARDS)
        + ".</p>",
        '<details class="blk-d"><summary>The three question types</summary>'
        "<p><b>choice</b> takes a map of option keys to descriptions and returns a probability "
        "for each, the top choice and a concentration score.</p>"
        "<p><b>noul</b> is yes/no and returns the probability of yes.</p>"
        "<p><b>score</b> takes two to ten ordered levels and returns the expected level; "
        "between levels 2 and 3, it could report 2.4.</p></details>"
        '<details class="blk-d"><summary>What the model actually reads</summary>'
        f'<div class="blk-pre">system: {esc(blink.SYSTEM)}\n\nuser: {esc(rendered)}</div>'
        "<p>Options are lettered rather than named so that every answer is one token. The "
        "letters run A to Z and then AA, AB and on, and each one is checked against the "
        "tokenizer at the exact position the answer would appear — a letter that is not a "
        "single token there is dropped. That leaves room for 255 options in one question.</p>"
        "<p>Runtime cap 131,072 tokens per question; the longest evaluated prompt was 37,906 "
        "tokens. Longer inputs are refused, not truncated.</p>"
        "</details>"
        '<details class="blk-d"><summary>What the probabilities are</summary>'
        "<p>For each question the model's scores over its offered options go through "
        "softmax(logits / T) with T = 1.0. These are option-conditional model probabilities, "
        "not certified probabilities of being correct; calibration can change across tasks, "
        "domains and option sets.</p>"
        "<p>confidence = (p_max − 1/K)/(1 − 1/K) is a concentration score, not a probability "
        "of being correct.</p></details>"
        '<details class="blk-d"><summary>What it will not do</summary>'
        "<p>It cannot write, explain an answer or call a tool. Text inside the state can still "
        "sway the answer. Set any decision threshold for your own risk.</p></details>"
        '<details class="blk-d"><summary>Running it yourself</summary>'
        "<p>blink.py ships with the weights. Use the pinned install line below; the "
        "prompts and readout match the reported evaluations.</p>"
        f'<div class="blk-pre">{esc(API_SNIPPET)}</div>'
        "<p>BLINK_MODEL picks the model, BLINK_TEMPERATURE the readout temperature. The "
        "weights are for non-commercial research use; see the model cards.</p>"
        + links_html()
        + "</details>",
    ]


# --- api ----------------------------------------------------------------------------
# The copy, the table rows and the examples all live in api_doc; this only arranges them.


def _api_mark(kind: str) -> str:
    label = dict(api_doc.COPY["legend"])[kind]
    return f'<i class="blk-mk {esc(kind)}" role="img" aria-label="{esc(label)}" title="{esc(label)}"></i>'


def _api_code(text: str, label: str = "") -> str:
    head = f'<p class="blk-apilabel">{esc(label)}</p>' if label else ""
    return f'{head}<div class="blk-pre">{esc(text)}</div>'


def _api_notes(notes) -> str:
    return "".join(f"<p>{esc(n)}</p>" for n in notes)


def _api_text_table(cols, rows) -> str:
    head = "".join(f'<th scope="col">{esc(h)}</th>' for h in cols)
    body = "".join("<tr>" + "".join(f"<td>{esc(v)}</td>" for v in row) + "</tr>" for row in rows)
    return (f'<div class="blk-tablewrap blk-apimini"><table class="blk-table blk-apitext"><thead><tr>{head}'
            f"</tr></thead><tbody>{body}</tbody></table></div>")


def api_table() -> str:
    """TypeSafe's API, a blink server and this Space, row by row, each cell marked."""
    c = api_doc.COPY
    head = "".join(f'<th scope="col">{esc(h)}</th>' for h in api_doc.TABLE_COLS)
    _, c1, c2, c3 = (esc(h) for h in api_doc.TABLE_COLS)
    # each cell names its column, so a phone can stack a row with its labels (style.css)
    body = "".join(
        f'<tr><th scope="row">{esc(row)}</th><td data-col="{c1}"><code>{esc(theirs)}</code></td>'
        f'<td data-col="{c2}"><span>{_api_mark(sk)}{esc(server)}</span></td>'
        f'<td data-col="{c3}"><span>{_api_mark(pk)}{esc(space)}</span></td></tr>'
        for row, theirs, (sk, server), (pk, space) in api_doc.TABLE_ROWS
    )
    legend = "".join(f"<span>{_api_mark(k)}{esc(label)}</span>" for k, label in c["legend"])
    return (
        f'<figure class="blk-figure blk-apifig"><figcaption>{esc(c["table_label"])}</figcaption>'
        f'<div class="blk-tablewrap"><table class="blk-table blk-apitable"><thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table></div>"
        f'<p class="blk-apilegend">{legend}</p></figure>'
    )


def api_steps() -> str:
    c = api_doc.COPY
    items = "".join(f"<li><b>{esc(title)}</b><code>{esc(code)}</code></li>" for title, code in c["steps"])
    return (f'<div class="blk-stepshead"><p class="blk-eyebrow">{esc(c["steps_label"])}</p>{links_html()}</div>'
            f'<ol class="blk-steps">{items}</ol>')


def _api_fold(summary: str, inner: str) -> str:
    return f'<details class="blk-d blk-apifold"><summary>{esc(summary)}</summary>{inner}</details>'


def api_blocks() -> list[str]:
    """The API tab: the table and three steps first; the details fold away."""
    c, d = api_doc.COPY, api_doc
    draft = f' <span class="blk-tag blk-copytag">{esc(c["draft"])}</span>' if d.DRAFT else ""
    server = (
        _api_code(d.SERVER_RUN)
        + _api_code(d.DOCKER_RUN, "Docker, from the downloaded folder")
        + _api_notes(c["server_notes"])
        + f'<p class="blk-apilabel">{esc(c["key"])}</p>' + _api_notes(c["key_notes"]) + _api_code(d.KEY_RUN)
        + f'<p class="blk-apilabel">{esc(c["batching"])}</p>' + _api_notes(c["batching_notes"])
        + _api_code(d.BATCH_RUN)
    )
    client = (
        _api_code(d.CLIENT_ENV)
        + _api_notes(c["client_notes"])
        + _api_code(d.PYTHON_CLIENT, c["python"])
        + _api_code(d.JS_CLIENT, c["javascript"])
        + _api_code(d.HTTP_CLIENT, c["http"])
    )
    example = (
        f'<p class="blk-apilabel">{esc(c["fields_label"])}</p>'
        + _api_text_table(c["fields_cols"], c["fields"])
        + _api_code(d.pretty(d.EXAMPLE_REQUEST), "Request")
        + _api_code(d.pretty(d.EXAMPLE_RESPONSE), "Response")
        + f'<p>{esc(c["example_note"])}</p>'
    )
    errors = (
        _api_text_table(c["errors_cols"], c["error_rows"])
        + f'<p>{esc(c["error_note"])}</p>'
        + _api_code(d.pretty(d.REFUSED_REQUEST), "A one-level score")
        + _api_code("HTTP/1.1 422 Unprocessable Entity\n" + d.pretty(d.REFUSED_BODY))
        + _api_code("HTTP/1.1 401 Unauthorized\nWWW-Authenticate: Bearer\n" + d.pretty(d.UNAUTHORIZED_BODY),
                    "Without the key the server checks")
        + f'<p>{esc(c["limits"])}</p>'
        + _api_code(d.MODELS_CALL + "\n" + d.pretty(d.MODELS_BODY), c["models"])
    )
    space = (
        _api_notes(c["space_notes"][:1])
        + _api_code(d.SPACE_PYTHON, c["python"])
        + _api_code(d.SPACE_CURL, c["http"])
        + _api_code(d.SPACE_REPLY)
        + _api_notes(c["space_notes"][1:])
    )
    diff = "<ul>" + "".join(f"<li>{esc(n)}</li>" for n in c["diff_notes"]) + "</ul>"
    return [
        f'<h2 class="blk-h2">{esc(c["heading"])}{draft}</h2><p class="blk-note">{esc(c["lede"])}</p>'
        + links_html(((c["wire_docs"], d.WIRE_FORMAT_URL),)),
        api_table(),
        api_steps(),
        '<div class="blk-api">'
        + _api_fold(c["server"], server)
        + _api_fold(c["client"], client)
        + _api_fold(c["example"], example)
        + _api_fold(c["errors"], errors)
        + _api_fold(c["space"], space)
        + _api_fold(c["diff"], diff)
        + "</div>",
    ]


# --- next click on a screen -----------------------------------------------------------
# Every string the screenshot card shows, in one place. The verdicts are screens.verdict's.
SCREEN = {
    "title": "Screen click",
    "blurb": "Pick the next click on a screenshot.",
    "model": "Screenshots use {model}",
    "upload": "Screenshot",
    "task": "Task",
    "task_hint": "What should the agent do here?",
    "grid": "Grid",
    "boxes": "Boxes",
    "boxes_hint": "Drag to draw boxes, or type x1, y1, x2, y2 per line.",
    "clear": "Clear boxes",
    "grid_pill": "Grid {k}\u00d7{k}",
    "done": "Done?",
    "risky": "Risky?",
    "saved": "saved run",
    "not_run": "Not run yet. Click Decide.",
    "draw": "Drag or tap two corners to draw a box.",
    "miss": "No saved run for this screenshot. Try a preset as is.",
    "off": "Screenshot input is off here.",
    "questions": "Questions",
    "rule": "Decision rule",
    "rule_note": "Marks label page elements. This demo never clicks.",
    "raw": "Raw response",
    "read": "What blink read",
    "alt": "Screenshot with {n} numbered {unit}",
    "image_tokens": "image tokens",
    "how": "{title} uses that tower for screenshots. Every other tab uses the text side. Either way, "
           "blink reads option-letter scores and generates no text.",
    "docs": "Run it yourself",
}
# where the card points to run the same thing on one's own server; opens in a new tab
SCREEN_DOCS_URL = "https://thegovind.github.io/blink/computer-use/"
GRID_CHOICES = tuple((SCREEN["grid_pill"].format(k=k), str(k)) for k in screens.GRIDS)

_EYE = ('<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12Z"/>'
        '<circle cx="12" cy="12" r="3"/></svg>')
_CHECK = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12.5 10 17.5 19 7"/></svg>'


def screen_note() -> str:
    return f'<p class="blk-note">{esc(SCREEN["blurb"])}</p>'


def screen_model_chip(model: str | None = None) -> str:
    """Which model reads the screenshot, whatever the page's switch says."""
    name = short_model(model or screens.vision_model() or screens.VISION_NAME)
    return (f'<p class="blk-eyechip" title="{esc(SCREEN["model"].format(model=name))}">{_EYE}'
            f'<span>{esc(SCREEN["model"].format(model=name))}</span></p>')


def _share(v: float, total: float) -> str:
    return f"{v / total * 100:.3f}%"


def screen_stage(shot: screens.Shot, answers: dict | None = None, *, busy: bool = False,
                 draw: bool = False, act: str | None = None) -> str:
    """The screenshot, and over it a mark for every numbered box laid exactly on the one blink reads.

    Answered, each mark glows by its probability and the chosen one is lit in the flash colour."""
    W, H = shot.size
    sw, sh = shot.source_size or shot.size
    s = screens.style(W)
    pad = 0 if shot.grid else s["stroke"] + s["halo"]
    probs = (answers or {}).get("element", {}).get("probabilities", {})
    win = answers["element"]["choice"] if answers and act in ("click", "ask", "unsure") else None
    marks = []
    for b in shot.boxes:
        n = str(b.n)
        x1, y1 = max(0, b.box[0] - pad), max(0, b.box[1] - pad)
        x2, y2 = min(W, b.box[2] + pad), min(H, b.box[3] + pad)
        p = probs.get(n)
        on = " win" if win == n else ""
        box_style = (f"--sx:{_share(x1, W)};--sy:{_share(y1, H)};--sw:{_share(x2 - x1, W)};--sh:{_share(y2 - y1, H)}"
                     + (f";--sp:{p:.4f}" if p is not None else ""))
        title = n + (f" \u00b7 {b.role}: {b.name}" if b.name else "")
        tx1, ty1, tx2, ty2 = b.tag
        shown = p is not None and (not shot.grid or p >= 0.05 or on)
        pct = f"<i>{p:.0%}</i>" if shown else ""
        # how many characters of its monospaced face the number takes, so the page can keep it inside
        chars = len(n) + (len(f"{p:.0%}") if shown else 0)
        tag_style = (f"--tx:{_share(tx1, W)};--ty:{_share(ty1, H)};--tw:{_share(tx2 - tx1, W)};"
                     f"--th:{_share(ty2 - ty1, H)};--tr:{(ty2 - ty1) / W:.5f};--tc:{chars:g}"
                     + (";--tg:4px" if shown else ""))
        marks.append(
            f'<span class="blk-som{" cell" if shot.grid else ""}{on}" data-n="{n}" style="{box_style}"'
            f' title="{esc(title)}"></span>'
            f'<span class="blk-somtag {esc(b.at)}{" cell" if shot.grid else ""}{on}" data-n="{n}"'
            f' style="{tag_style}"><b>{n}</b>{pct}</span>'
        )
    pill = (f'<span class="blk-shot-pill">{esc(SCREEN["grid_pill"].format(k=shot.grid))}</span>'
            if shot.grid else "")
    alt = SCREEN["alt"].format(n=len(shot.boxes), unit="cells" if shot.grid else "boxes")
    classes = "blk-shot" + (" busy" if busy else "") + (" answered" if answers else "")
    return (
        f'<figure class="{classes}" data-draw="{1 if draw else 0}" data-act="{esc(act or "")}">'
        f'<div class="blk-shot-frame" data-w="{sw}" data-h="{sh}" style="aspect-ratio:{W} / {H}">'
        f'<img src="{shot.view or shot.uri}" alt="{esc(alt)}" draggable="false">'
        f'<div class="blk-marks">{"".join(marks)}</div>{pill}</div></figure>'
    )


def _screen_verdict(shot: screens.Shot, answers: dict, meta: dict) -> tuple[str, str]:
    tone, act, headline, detail = screens.verdict(answers)
    choice = answers["element"]["choice"]
    box = next((b for b in shot.boxes if str(b.n) == choice), None)
    mark = {"click": esc(choice), "ask": esc(choice), "unsure": "?", "done": _CHECK}[act]
    line = f"{box.role} \u00b7 {box.name}" if act in ("click", "ask") and box is not None and box.name else ""
    tags = '<em class="blk-tag">simulated</em>'
    if meta.get("engine") == "replay":
        tags += f'<em class="blk-tag blk-saved">{esc(SCREEN["saved"])}</em>'
    return act, (
        f'<div class="blk-sv {esc(tone)}" data-act="{esc(act)}" title="{esc(detail)}">'
        f'<span class="blk-sv-mark" aria-hidden="true">{mark}</span>'
        f'<div class="blk-sv-text"><h3>{esc(headline)}{tags}</h3>'
        + (f"<p>{esc(line)}</p>" if line else "")
        + "</div></div>"
    )


def _screen_meter(label: str, p: float | None, on: bool, kind: str) -> str:
    """A ring for one yes/no check; empty until answered, so it fills in when the answer lands."""
    shown = "\u2013" if p is None else f"{p:.0%}"
    return (
        f'<div class="blk-meter{" on" if on else ""}{" empty" if p is None else ""}" data-kind="{kind}">'
        f'<span class="blk-ring" style="--sp:{p or 0:.4f}" role="img" aria-label="{esc(label)} {shown}">'
        f'<b>{shown}</b></span><span class="k">{esc(label)}</span></div>'
    )


def _screen_meters(answers: dict | None = None) -> str:
    done = answers["done"]["noul"] if answers else None
    risky = answers["risky"]["noul"] if answers else None
    return (
        '<div class="blk-meters">'
        + _screen_meter(SCREEN["done"], done, done is not None and done >= screens.DONE_AT, "done")
        + _screen_meter(SCREEN["risky"], risky, risky is not None and risky >= screens.RISKY_AT, "risky")
        + "</div>"
    )


def _shot_id(shot: screens.Shot) -> str:
    """Which screenshot the card holds: a new one may be shorter, the same one edited may not jump."""
    if not shot.upload:
        return shot.key
    return "upload-" + hashlib.sha256(shot.view[-4096:].encode("ascii")).hexdigest()[:12]


def _elided(shot: screens.Shot, state: dict) -> dict:
    return {**state, "screenshot": f"data:image/png;base64,\u2026 ({len(shot.png):,} bytes)"}


def _idle_row(text: str) -> str:
    """Where the verdict goes before there is one: the same height, so the screenshot never moves."""
    return (f'<div class="blk-sv blk-sv-idle"><span class="blk-sv-mark" aria-hidden="true"></span>'
            f'<div class="blk-sv-text"><p>{esc(text)}</p></div></div>')


def screen_panel(shot: screens.Shot, out: dict | None = None, *, busy: bool = False, draw: bool = False,
                 note: str = "", notice: str = "", rev=0) -> str:
    """The answer column: the verdict, the marked screenshot, then the two checks and the details.

    Every state keeps the rows above the screenshot, so drawing on it never makes it jump. `rev`
    changes with every request, so a repeat answer still clears the busy state it replaces."""
    act, head, foot = None, "", notice
    if out is not None:
        answers, meta = out["answers"], out["meta"]
        act, head = _screen_verdict(shot, answers, meta)
        state, qs = screens.request(shot)
        stats = (
            '<div class="blk-stats">' + answered_by_html(meta) + timing_html(meta)
            + f'<span><b>{meta["input_tokens"]:,}</b> input tokens</span>'
            + (f'<span><b>{meta["visual_tokens"]:,}</b> {esc(SCREEN["image_tokens"])}</span>'
               if meta.get("visual_tokens") else "")
            + f'<span><b>{meta["generated_tokens"]}</b> generated</span>'
            f'<span><b>{len(qs)}</b> questions</span>'
            f'<span>temperature <b>{meta["temperature"]:g}</b></span></div>'
        )
        raw = json.dumps({"state": _elided(shot, state), "questions": qs, **out}, indent=2, ensure_ascii=False)
        more = (
            f'<details class="blk-more"><summary>{esc(SCREEN["rule"])}</summary>'
            f'<div class="blk-pre">{esc(screens.rule_source())}</div>'
            f'<p>{esc(SCREEN["rule_note"])}</p></details>'
            f'<details class="blk-more"><summary>{esc(SCREEN["questions"])}</summary>'
            f'<div class="blk-pre">{esc(json.dumps(qs, indent=2))}</div></details>'
            f'<details class="blk-more"><summary>{esc(SCREEN["raw"])}</summary>'
            f'<div class="blk-pre">{esc(raw)}</div></details>'
        )
        if shot.upload:
            more += (f'<details class="blk-more"><summary>{esc(SCREEN["read"])}</summary>'
                     f'<img class="blk-read" src="{screens.thumb(shot)}" alt="{esc(SCREEN["read"])}"></details>')
        foot = _screen_meters(answers) + f'<div class="blk-answer-foot">{stats}{more}</div>'
    elif busy:
        head = ('<div class="blk-sv blk-sv-wait" aria-busy="true"><span class="blk-sv-mark"></span>'
                f'<div class="blk-sv-text"><span class="blk-run">{esc(PENDING_LABEL)}</span></div></div>')
    else:
        head = _idle_row(note or SCREEN["draw" if draw else "not_run"])
    if out is None:
        foot = _screen_meters() + foot
    stage = screen_stage(shot, out["answers"] if out is not None else None, busy=busy, draw=draw, act=act)
    banner = mock_banner()
    docs = links_html(((SCREEN["docs"], SCREEN_DOCS_URL),), cls="blk-links blk-runit")
    return (f'<div class="blk-scr{" busy" if busy else ""}" data-rev="{esc(rev)}" data-shot="{_shot_id(shot)}">'
            f"{banner}{head}{stage}{foot}{docs}</div>")


def run_screen(shot: screens.Shot, prefer: str = "live", draw: bool = False, rev=0) -> str:
    """Ask blink-mimo-9b about this screenshot; whatever goes wrong is said in the answer column."""
    try:
        out = screens.decide(shot, prefer=prefer)
    except screens.ScreenMiss:
        live = getattr(blink.engine(screens.vision_model()), "name", "") != "replay"
        return screen_panel(shot, draw=draw, note=SCREEN["not_run" if live else "miss"], rev=rev)
    except blink.ImageError as exc:
        if "reads text only" in str(exc):
            return screen_panel(shot, draw=draw, note=SCREEN["off"], rev=rev)
        return screen_panel(shot, draw=draw, notice=error_html(str(exc)), rev=rev)
    except blink.BlinkError as exc:
        return screen_panel(shot, draw=draw, notice=error_html(str(exc)), rev=rev)
    except Exception as exc:  # the live model failed: keep gradio's own errors, soften the rest
        return screen_panel(shot, draw=draw, notice=_live_failure(exc), rev=rev)
    return screen_panel(shot, out, draw=draw, rev=rev)
