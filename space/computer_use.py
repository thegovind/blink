"""The Computer use tab: real runs of blink driving apps from screenshots, then how it reads one.

Everything here is HTML built from space/cua/gallery.json (see build_gallery.py) plus one script that
plays it: the stage (a strip of runs, or one run in the player), the scenario gallery, the screenshot
explainer and the tiles beyond clicks. Nothing here calls a model; Screen click, which does, sits
between the explainer and the tiles (app.py). Every word the tab shows is in COPY.
"""

from __future__ import annotations

import functools
import html
import json
import math
import os
import urllib.parse

import screens

HERE = os.path.dirname(os.path.abspath(__file__))
GALLERY = os.path.join(HERE, "cua", "gallery.json")
SCHEMA = "blink-cua-gallery/1"
TAB = "computer-use"
PARAMS = ("scenario", "model", "run", "step")

# Every string the tab shows, in one place, so the words can change without touching the code.
# Placeholders in braces are filled in by the page; keep them.
COPY = {
    "eyebrow": "Computer use",
    "title": "Watch blink use ten apps",
    "pitch": "A marked screenshot, three questions, then a click, pause, or stop.",
    "mock": "Test data, not real blink runs.",
    "stat_apps": "original apps",
    "stat_runs": "of {n} runs completed",
    "stat_step": "median per step",
    "stat_gate": "risky clicks paused on",
    "runs": "{ok}/{n} runs",
    "open": "Watch",
    "gallery": "Pick an app",
    "phone": "phone",
    "canvas": "canvas app",
    "turns": "turn-based",
    # the player
    "close": "Back to apps",
    "video": "Video",
    "steps": "Steps",
    "play": "Play steps",
    "pause": "Pause",
    "prev": "Previous step",
    "next": "Next step",
    "step_of": "Step {n}/{total}",
    "finished": "Task done",
    "failed": "Task not done",
    "steps_taken": "{n} steps",
    "steps_taken_one": "1 step",
    "asked": "{n} times asked first",
    "per_step": "{ms} ms per step",
    "interval": "95% interval {lo} to {hi}%",
    "matched": "{pct}% picked the right box",
    "show_failed": "Watch a miss",
    "show_best": "Watch a finish",
    "show_other": "Watch another",
    "seed": "seed {seed}",
    "expected": "Right box: {n}",
    "click": "Click {n}",
    "lane": "Lane {n}",
    "ask": "Ask before clicking {n}",
    "type": "Type {value} in {n}",
    "select": "Pick {value} in {n}",
    "done": "Looks done",
    "none": "Nothing to click",
    "done_q": "Done?",
    "risky_q": "Risky?",
    "ms": "{ms} ms",
    "tokens": "{n} image tokens",
    "models": "Same tasks, other models",
    "task": "Task",
    "no_video": "No video here. The steps still work.",
    "no_steps": "Steps unavailable. The video is still here.",
    # how it reads a screenshot
    "how_eyebrow": "From screenshot to click",
    "how_line": "One pass per question. A probability for every answer, no text generated.",
    "how_nodes": (
        ("Marked screenshot", "Boxes show what can be clicked."),
        ("Vision encoder", "{base}, unchanged by blink training."),
        ("Image tokens", "One per {side} \u00d7 {side} px patch."),
        ("Three questions", "Next box, done yet, risky click."),
        ("Probabilities", "One for every offered answer."),
        ("Next step", "Click, pause, stop, or hand over."),
    ),
    "how_facts": (
        "{width} \u00d7 {height} px",
        "{blocks} blocks \u00b7 hidden {hidden}",
        "{tokens}",
        "3 questions",
        "one forward pass each",
        "{outcomes}",
    ),
    "how_outcomes": "click \u00b7 pause \u00b7 done \u00b7 hand over",
    "how_questions": ("Which box?", "Done?", "Risky?"),
    # try it
    "try_eyebrow": "Try it",
    "try_title": "Try your screenshot",
    # beyond clicks
    "more_eyebrow": "Beyond clicks",
    "more_line": "Voice, audio, screen checks, and document pages.",
    "sound": "Play with sound",
}


def esc(s) -> str:
    return html.escape(str(s), quote=True)


# --- the gallery -----------------------------------------------------------------------


def path() -> str:
    """BLINK_GALLERY points the page at another build (a mock, a local check); the Space ships cua/gallery.json."""
    return os.environ.get("BLINK_GALLERY") or GALLERY


@functools.lru_cache(maxsize=4)
def _load(where: str, stamp: float):
    with open(where, encoding="utf-8") as fh:
        g = json.load(fh)
    if g.get("schema") != SCHEMA or not g.get("scenarios"):
        return None
    return g


def load() -> dict | None:
    """The gallery, or None when there is none to show (the tab then leads with the explainer and Screen click)."""
    where = path()
    try:
        stamp = os.path.getmtime(where)
    except OSError:
        return None
    try:
        return _load(where, stamp)
    except (OSError, ValueError):
        return None


def available() -> bool:
    return load() is not None


def media(g: dict, rel: str | None) -> str:
    return g["media_base"] + urllib.parse.quote(rel) if rel else ""


def scenario(g: dict, sid: str) -> dict | None:
    return next((s for s in g["scenarios"] if s["id"] == sid), None)


def runs_line(st: dict) -> str:
    return COPY["runs"].format(ok=st["successes"], n=st["episodes"])


# --- deep links ---------------------------------------------------------------------


def parse_link(raw, g: dict | None = None) -> dict:
    """?tab=computer-use&scenario=shop&model=blink-4b&run=failed&step=3 -> what the player opens.

    Unknown scenarios, models and steps fall back rather than fail: a stale link still opens the tab."""
    g = g if g is not None else load()
    out = {"scenario": None, "model": None, "run": "showcase", "step": None}
    if not isinstance(raw, str) or g is None:
        return out
    params = urllib.parse.parse_qs(raw.partition("#")[0].lstrip("?"))
    asked = (params.get("scenario") or [""])[0].strip().lower()
    sc = scenario(g, asked)
    if sc is None:
        return out
    out["scenario"] = sc["id"]
    models = [m for m in g["models"] if m in sc["models"]]
    want = (params.get("model") or [""])[0].strip()
    lead = g.get("lead_model") or g["vision_model"]
    out["model"] = want if want in models else (lead if lead in models else models[0])
    item = sc["models"][out["model"]]
    if (params.get("run") or [""])[0].strip().lower() in ("failed", "failure") and item.get("failure"):
        out["run"] = "failure"
    ep = g["episodes"][item["showcase" if out["run"] == "showcase" else "failure"]]
    try:
        step = int((params.get("step") or [""])[0])
    except ValueError:
        step = None
    if step is not None and ep["steps"]:
        out["step"] = min(max(step, 1), len(ep["steps"]))
    return out


# --- the stage: a strip of runs --------------------------------------------------------


def _video(g: dict, ep: dict, *, auto: bool, cls: str = "") -> str:
    poster = media(g, ep.get("poster"))
    src = media(g, ep.get("preview") or ep.get("video"))
    if not src:
        return f'<img class="{esc(cls)}" src="{esc(poster)}" alt="" loading="lazy">' if poster else ""
    return (f'<video class="{esc(cls)}" data-auto="{1 if auto else 0}" muted loop playsinline preload="none"'
            f' poster="{esc(poster)}" aria-hidden="true"><source src="{esc(src)}" type="video/mp4"></video>')


def hero_order(g: dict) -> list[str]:
    """The strip's runs: a phone between two screens where there is one, then the rest, in registry order."""
    ids = [e for e in g.get("hero", []) if e in g["episodes"]]
    phones = [e for e in ids if g["episodes"][e]["frame"][1] > g["episodes"][e]["frame"][0]]
    wide = [e for e in ids if e not in phones]
    if phones and len(wide) >= 2:
        return [wide[0], phones[0], wide[1], *[e for e in ids if e not in (wide[0], phones[0], wide[1])]]
    return ids


def _link(sid: str) -> str:
    return "?" + urllib.parse.urlencode({"tab": TAB, "scenario": sid})


def hero(g: dict) -> str:
    reels = []
    for i, eid in enumerate(hero_order(g)):
        ep = g["episodes"][eid]
        sc = scenario(g, ep["scenario"])
        st = sc["models"][ep["model"]]["stats"]
        w, h = ep["frame"]
        reels.append(
            f'<a class="blk-cuax-reel{" tall" if h > w else ""}" href="{esc(_link(sc["id"]))}" data-sid="{esc(sc["id"])}"'
            f' style="--ar:{w} / {h};--i:{i}">{_video(g, ep, auto=True)}'
            f'<span class="blk-cuax-cap"><b>{esc(sc["title"])}</b><em>{esc(runs_line(st))}</em></span></a>'
        )
    ids = hero_order(g)[:3]
    wide = all(g["episodes"][e]["frame"][0] > g["episodes"][e]["frame"][1] for e in ids)
    return (f'<div class="blk-cuax-strip{" wide" if wide else ""}">{"".join(reels)}</div>'
            if reels else "")


def _pct(v: float | None) -> str:
    return "\u2013" if v is None else f"{round(v * 100):d}%"


def stat_row(g: dict) -> str:
    lead = g.get("lead_model") or g["vision_model"]
    total = g["totals"].get(lead) or next(iter(g["totals"].values()), None)
    if not total:
        return ""
    who = lead.rsplit("/", 1)[-1] if lead in g["totals"] else ""
    items = [(str(total["scenarios"]), COPY["stat_apps"]),
             (_pct(total["rate"]), COPY["stat_runs"].format(n=total["episodes"]) + (f" \u00b7 {who}" if who else ""))]
    if total.get("p50_ms"):
        items.append((f"{total['p50_ms'] / 1000:.1f} s", COPY["stat_step"]))
    if total.get("gate_recall") is not None:
        items.append((_pct(total["gate_recall"]), COPY["stat_gate"]))
    return ('<div class="blk-cuax-stats">'
            + "".join(f'<span style="--i:{i}"><b>{esc(v)}</b>{esc(k)}</span>' for i, (v, k) in enumerate(items))
            + "</div>")


def _dots(outcomes: list[dict]) -> str:
    return ('<span class="blk-cuax-dots" aria-hidden="true">'
            + "".join(f'<i class="{"ok" if o["ok"] else "no"}"></i>' for o in outcomes) + "</span>")


def cards(g: dict) -> str:
    out = []
    for i, sc in enumerate(g["scenarios"]):
        lead = g.get("lead_model") or g["vision_model"]
        model = lead if lead in sc["models"] else next(iter(sc["models"]))
        item = sc["models"][model]
        ep = g["episodes"].get(item["showcase"] or "")
        if ep is None:
            continue
        w, h = ep["frame"]
        # the game waits for each decision (the harness lockstep clock), so its card says so
        tag = (COPY["phone"] if sc["device"] == "phone" else COPY["turns"] if sc["kind"] == "game"
               else COPY["canvas"] if sc["kind"] != "dom" else "")
        out.append(
            f'<a class="blk-cuax-card" href="{esc(_link(sc["id"]))}" data-sid="{esc(sc["id"])}" style="--i:{i}">'
            f'<span class="blk-cuax-thumb{" tall" if h > w else ""}">{_video(g, ep, auto=False)}'
            + (f'<em class="blk-cuax-kind">{esc(tag)}</em>' if tag else "")
            + f'</span><b>{esc(sc["title"])}</b>'
            f'<span class="blk-cuax-rate">{_dots(item["outcomes"])}<span>{esc(runs_line(item["stats"]))}</span></span>'
            f'<span class="blk-cuax-by">{esc(model)}</span></a>'
        )
    return (f'<p class="blk-eyebrow blk-cuax-gl">{esc(COPY["gallery"])}</p>'
            f'<nav class="blk-cuax-cards">{"".join(out)}</nav>')


# --- how it reads a screenshot -------------------------------------------------------------


def explainer(facts: dict, thumb: str = "", answers: dict | None = None, outcome: str = "") -> str:
    """Six nodes, told in order when they come into view: the numbered screenshot, the vision encoder,
    the image tokens laid over the screenshot as the patches they are, the typed questions, their
    probabilities and the action. `facts` are ui.vision_facts(); `answers` a saved run's, when there is one."""
    tokens = facts.get("tokens") or 0
    side = facts.get("side") or 32
    width, height = facts.get("width") or 1440, facts.get("height") or 864
    cols, rows = max(1, math.ceil(width / side)), max(1, math.ceil(height / side))
    values = dict(facts, tokens=f"{tokens:,} image tokens" if tokens else "image tokens",
                  side=side, width=width, height=height, outcome=outcome or COPY["click"].format(n="N"),
                  outcomes=COPY["how_outcomes"])
    probs = []
    if answers:
        element = answers.get("element", {})
        top = sorted(element.get("probabilities", {}).items(), key=lambda kv: -kv[1])[:3]
        probs = [(f"box {k}", v, k == element.get("choice")) for k, v in top]
        probs += [(COPY["done_q"], answers.get("done", {}).get("noul"), False),
                  (COPY["risky_q"], answers.get("risky", {}).get("noul"), False)]
    win_class = ' class="win"'
    bars = "".join(
        f'<li{win_class if win else ""} style="--p:{(p or 0):.3f};--i:{i}"><span>{esc(k)}</span>'
        f'<i></i><b>{esc(_pct(p))}</b></li>' for i, (k, p, win) in enumerate(probs))
    art = [
        f'<span class="blk-cx-shot"><img src="{esc(thumb)}" alt=""></span>' if thumb else '<span class="blk-cx-shot"></span>',
        '<span class="blk-cx-enc">' + "".join(f'<i style="--k:{k}"></i>' for k in range(int(facts.get("blocks") or 27)))
        + "</span>",
        (f'<span class="blk-cx-grid" style="--cols:{cols};--rows:{rows}">'
         + (f'<img src="{esc(thumb)}" alt="">' if thumb else "") + '<span class="blk-cx-cells"></span>'
         + f'<b class="blk-cx-count" data-to="{tokens}">{tokens:,}</b></span>'),
        '<span class="blk-cx-qs">' + "".join(f'<em style="--k:{k}">{esc(q)}</em>'
                                              for k, q in enumerate(COPY["how_questions"])) + "</span>",
        f'<ul class="blk-cx-bars">{bars}</ul>',
        f'<span class="blk-cx-act"><b>{esc(values["outcome"])}</b></span>',
    ]
    nodes = []
    for i, ((label, caption), fact, figure) in enumerate(zip(COPY["how_nodes"], COPY["how_facts"], art, strict=True)):
        nodes.append(
            f'<li style="--i:{i}"><span class="blk-cx-art">{figure}</span>'
            f'<span class="blk-cx-n">{i + 1}</span><b>{esc(label)}</b>'
            f'<span class="blk-cx-cap">{esc(caption.format(**values))}</span>'
            f'<code>{esc(fact.format(**values))}</code></li>'
        )
    return (f'<section class="blk-cx" id="cua-how"><p class="blk-eyebrow">{esc(COPY["how_eyebrow"])}</p>'
            f'<p class="blk-note">{esc(COPY["how_line"])}</p><ol class="blk-cx-flow">{"".join(nodes)}</ol></section>')


# --- beyond clicks -------------------------------------------------------------------


def tiles(g: dict | None) -> str:
    items = (g or {}).get("more") or []
    if not items:
        return ""
    out = []
    for i, t in enumerate(items):
        still = media(g, t.get("poster") or t.get("chart") or (t.get("images") or [None])[0])
        metric = t.get("metric") or {}
        out.append(
            f'<button type="button" class="blk-cuax-tile" data-tile="{esc(t["id"])}" style="--i:{i}">'
            + (f'<span class="blk-cuax-still"><img src="{esc(still)}" alt="" loading="lazy">'
               + ('<span class="blk-cuax-play" aria-hidden="true"></span>' if t.get("video") else "")
               + "</span>" if still else "")
            + f'<b>{esc(t["title"])}</b>'
            + (f'<span class="blk-cuax-metric"><strong>{esc(metric.get("value", ""))}</strong>'
               f'{esc(metric.get("label", ""))}</span>' if metric else "")
            + "</button>"
        )
    data = esc(json.dumps({"more": items, "media_base": g["media_base"], "copy": {"sound": COPY["sound"]}},
                          separators=(",", ":"), ensure_ascii=False))
    return (f'<section class="blk-cuax-more" id="cua-more" data-t="{data}">'
            f'<p class="blk-eyebrow">{esc(COPY["more_eyebrow"])}</p><p class="blk-note">{esc(COPY["more_line"])}</p>'
            f'<div class="blk-cuax-tiles">{"".join(out)}</div><div class="blk-cuax-open" hidden></div></section>')


# --- assembly -----------------------------------------------------------------------


def client_data(g: dict) -> dict:
    """What the page carries: every run's summary and step rail. Each run's steps are fetched from beside its
    media when the player first needs them (build_gallery.py stages them), so the page stays light."""
    keep = ("media_base", "mock", "models", "vision_model", "lead_model", "scenarios")
    out = {k: g[k] for k in keep if k in g}
    out["episodes"] = {eid: {k: v for k, v in ep.items() if k != "steps"} for eid, ep in g["episodes"].items()}
    out["rule_cut"] = {"done_at": screens.DONE_AT, "risky_at": screens.RISKY_AT}
    return out


def pad_for(width: int) -> int:
    """How far a page mark reaches past its box: screens.style's stroke and halo."""
    stroke = max(2, round(width / 360))
    return stroke + max(1, stroke // 2)


def top(g: dict | None, explain: str = "") -> str:
    """The tab above Screen click: the stage, the gallery, then how it reads a screenshot."""
    head = (f'<p class="blk-eyebrow">{esc(COPY["eyebrow"])}</p><h2 class="blk-cuax-title">{esc(COPY["title"])}</h2>'
            f'<p class="blk-note blk-cuax-pitch">{esc(COPY["pitch"])}</p>')
    if g is None:
        return f'<div class="blk-cuax" data-g="null">{head}{explain}</div>'
    data = client_data(g)
    for ep in data["episodes"].values():
        ep.setdefault("pad", pad_for(ep["frame"][0]))
    banner = f'<div class="blk-warn blk-mock">{esc(COPY["mock"])}</div>' if g.get("mock") else ""
    blob = esc(json.dumps(data, separators=(",", ":"), ensure_ascii=False))
    copy = esc(json.dumps(COPY, separators=(",", ":"), ensure_ascii=False))
    return (f'<div class="blk-cuax" data-g="{blob}" data-copy="{copy}">{banner}{head}'
            f'<div class="blk-cuax-stage" aria-live="polite">{hero(g)}</div>{stat_row(g)}{cards(g)}{explain}</div>')


def try_head() -> str:
    return (f'<div class="blk-cuax-try" id="cua-try"><p class="blk-eyebrow">{esc(COPY["try_eyebrow"])}</p>'
            f'<h3>{esc(COPY["try_title"])}</h3></div>')


def home_figure(g: dict) -> str:
    """Home's figure: the first run of the strip, looping muted, with whose it is and how often it finishes."""
    ids = hero_order(g)
    if not ids:
        return ""
    ep = g["episodes"][ids[0]]
    sc = scenario(g, ep["scenario"])
    st = sc["models"][ep["model"]]["stats"]
    src = media(g, ep.get("preview") or ep.get("video"))
    caption = " \u00b7 ".join((sc["title"], runs_line(st), ep["model"]))
    return (f'<figure class="blk-cua-shot"><video autoplay muted loop playsinline preload="metadata"'
            f' poster="{esc(media(g, ep.get("poster")))}" aria-label="{esc(caption)}">'
            f'<source src="{esc(src)}" type="video/mp4"></video><figcaption>{esc(caption)}</figcaption></figure>')


def more(g: dict | None) -> str:
    return tiles(g)



# The player: plays the runs gallery.json holds, steps through them and keeps the address in step.
PLAYER_JS = r"""(() => {
  // The Computer use tab's player. Reads the gallery the page was built with (data-g), plays runs,
  // steps through them and keeps the address in step. Talks to nothing but the page it is on.
  const CHECK = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12.5 10 17.5 19 7"/></svg>';
  const BACK = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M15 5 8 12l7 7"/></svg>';
  const PREV = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M15 5 8 12l7 7"/></svg>';
  const NEXT = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m9 5 7 7-7 7"/></svg>';
  const PLAY = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5.5v13l10.5-6.5z"/></svg>';
  const PAUSE = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14M16 5v14"/></svg>';
  const esc = (s) => String(s == null ? '' : s).replace(/[&<>"']/g,
    (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const fmt = (s, o) => String(s || '').replace(/\{(\w+)\}/g, (m, k) => (k in o ? o[k] : m));
  const pct = (p) => (p == null ? '\u2013' : `${Math.round(p * 100)}%`);
  const pct1 = (p) => (p == null ? '\u2013' : `${(p * 100).toFixed(1)}%`);
  const reduce = !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);

  const boot = (tries) => {
    const root = element.querySelector('.blk-cuax');
    if (!root) {
      if (tries < 50) setTimeout(() => boot(tries + 1), 100);
      return;
    }
    if (root.__blink) return;
    root.__blink = true;
    start(root);
  };

  const start = (root) => {
    let G = null;
    let C = {};
    try {
      G = JSON.parse(root.dataset.g || 'null');
      C = JSON.parse(root.dataset.copy || '{}');
    } catch (err) {
      G = null;
    }

    // how it reads a screenshot: told once, in order, when it comes into view
    const how = root.querySelector('.blk-cx');
    if (how) {
      if (!reduce) how.classList.add('arm');
      const count = how.querySelector('.blk-cx-count');
      const go = () => {
        if (how.classList.contains('on')) return;
        how.classList.add('on');
        if (!count || reduce) return;
        const to = Number(count.dataset.to || 0);
        if (!to) return;
        const t0 = performance.now() + 900;
        const tick = (t) => {
          const k = Math.min(1, Math.max(0, (t - t0) / 1300));
          count.textContent = Math.round(to * (1 - Math.pow(1 - k, 3))).toLocaleString('en-US');
          if (k < 1) requestAnimationFrame(tick);
        };
        count.textContent = '0';
        requestAnimationFrame(tick);
      };
      if ('IntersectionObserver' in window) {
        const seen = new IntersectionObserver((es) => {
          if (es.some((e) => e.isIntersecting)) {
            go();
            seen.disconnect();
          }
        }, { threshold: 0.3 });
        seen.observe(how);
      } else {
        go();
      }
    }
    // the address the page opened with, read once: a run to open, or Screen click to bring into view
    const memo = window.__blinkCua || (window.__blinkCua = { open: null, first: true });
    const raw = memo.first && typeof window.__blinkFirstUrl === 'string' ? window.__blinkFirstUrl : window.location.search;
    const firstLoad = memo.first;
    memo.first = false;
    const q = new URLSearchParams(raw.split('#')[0].replace(/^\?/, ''));
    const toTry = firstLoad && !q.get('scenario') && (q.get('shot') || /^screen$/i.test((q.get('case') || '').trim()));
    if (toTry) {
      const bring = (n) => {
        const at = document.getElementById('cua-try');
        if (at && at.offsetParent !== null && document.querySelector('#screen-out .blk-scr')) {
          at.scrollIntoView({ behavior: 'auto', block: 'start' });
        } else if (n < 60) {
          setTimeout(() => bring(n + 1), 150);
        }
      };
      setTimeout(() => bring(0), 300);
    }
    if (!G) return;

    const base = G.media_base || '';
    const media = (rel) => (rel ? base + encodeURI(rel) : '');
    const byId = {};
    G.scenarios.forEach((s) => { byId[s.id] = s; });
    const stage = root.querySelector('.blk-cuax-stage');
    if (!stage) return;
    const heroHTML = stage.innerHTML;
    const doneAt = (G.rule_cut || {}).done_at || 0.6;
    const riskyAt = (G.rule_cut || {}).risky_at || 0.5;
    let cur = null;
    let timer = null;

    // muted loops play while they are on screen, and stop when they are not
    const io = 'IntersectionObserver' in window ? new IntersectionObserver((es) => {
      es.forEach((en) => {
        const v = en.target;
        if (en.isIntersecting && en.intersectionRatio >= 0.35) {
          if (!reduce) {
            v.muted = true;
            const p = v.play();
            if (p && p.catch) p.catch(() => {});
          }
        } else if (!v.paused) {
          v.pause();
        }
      });
    }, { threshold: [0, 0.35, 0.7] }) : null;
    const loops = (scope) => scope.querySelectorAll('video[data-auto="1"]').forEach((v) => {
      v.muted = true;
      v.preload = 'metadata';
      if (io) io.observe(v);
    });
    loops(stage);
    root.querySelectorAll('.blk-cuax-card').forEach((card) => {
      const v = card.querySelector('video');
      if (!v) return;
      card.addEventListener('pointerenter', (e) => {
        if (e.pointerType !== 'mouse' || reduce) return;
        v.muted = true;
        v.preload = 'auto';
        const p = v.play();
        if (p && p.catch) p.catch(() => {});
      });
      card.addEventListener('pointerleave', () => v.pause());
    });

    const itemOf = (st) => byId[st.sid].models[st.model];
    const epOf = (st) => G.episodes[itemOf(st)[st.role]];
    const count = (ep) => (ep.rail || []).length;

    // a run's steps sit beside its media; fetched once, when the player first needs them
    const fetched = memo.steps || (memo.steps = {});
    const stepsOf = (ep) => {
      const at = ep.steps_url;
      if (!at) return Promise.resolve([]);
      if (!fetched[at]) {
        fetched[at] = fetch(media(at), { credentials: 'omit' })
          .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
          .catch((err) => { delete fetched[at]; throw err; });
      }
      return fetched[at];
    };
    const ready = memo.ready || (memo.ready = {});

    // the address says what is open, so it can be shared; an embedding page hears it too
    const sync = () => {
      const url = new URL(window.location.href);
      ['scenario', 'model', 'run', 'step'].forEach((k) => url.searchParams.delete(k));
      if (cur) {
        url.searchParams.set('tab', 'computer-use');
        ['case', 'shot'].forEach((k) => url.searchParams.delete(k));
        url.searchParams.set('scenario', cur.sid);
        if (cur.model !== (G.lead_model || G.vision_model)) url.searchParams.set('model', cur.model);
        if (cur.role === 'failure') url.searchParams.set('run', 'failed');
        if (cur.view === 'steps') url.searchParams.set('step', String(cur.step + 1));
      }
      const next = url.pathname + url.search;
      if (next !== window.location.pathname + window.location.search) {
        window.history.replaceState(null, '', next);
      }
      try {
        window.parent.postMessage({ queryString: url.search.replace(/^\?/, ''), hash: '' }, 'https://huggingface.co');
      } catch (err) { /* not embedded */ }
    };

    const meter = (label, p, on, kind) => (p == null ? '' :
      `<div class="blk-meter${on ? ' on' : ''}" data-kind="${kind}"><span class="blk-ring" style="--sp:${p.toFixed(4)}"`
      + ` role="img" aria-label="${esc(label)} ${pct(p)}"><b>${pct(p)}</b></span><span class="k">${esc(label)}</span></div>`);

    const verdict = (s, lane) => {
      const n = s.pick;
      if (s.act === 'done') return { head: C.done, mark: CHECK, act: 'done' };
      if (!n) return { head: C.none, mark: '', act: 'unsure' };
      const value = s.value == null ? '' : `\u201c${s.value}\u201d`;
      if (s.gate) return { head: fmt(C.ask, { n }), mark: esc(n), act: 'ask' };
      if (s.act === 'type') return { head: fmt(C.type, { n, value }), mark: esc(n), act: 'click' };
      if (s.act === 'select') return { head: fmt(C.select, { n, value }), mark: esc(n), act: 'click' };
      return { head: fmt(lane ? C.lane : C.click, { n }), mark: esc(n), act: 'click' };
    };

    const figure = (ep, s, narrow) => {
      const [W, H] = ep.frame;
      const pad = ep.pad || 6;
      const share = (v, t) => `${(v / t * 100).toFixed(3)}%`;
      const act = verdict(s).act;
      const marks = s.boxes.map((b, i) => {
        const n = i + 1;
        const p = s.p[i];
        const win = s.pick === n && s.act !== 'done' ? ' win' : '';
        const want = s.oracle === n && s.pick !== n && s.act !== 'done' ? ' expect' : '';
        const x1 = Math.max(0, b[0] - pad);
        const y1 = Math.max(0, b[1] - pad);
        const x2 = Math.min(W, b[2] + pad);
        const y2 = Math.min(H, b[3] + pad);
        const t = s.tags[i];
        const shown = p != null && (!narrow || !!win);
        const chars = String(n).length + (shown ? pct(p).length : 0);
        return `<span class="blk-som${win}${want}" data-n="${n}" style="--sx:${share(x1, W)};--sy:${share(y1, H)};`
          + `--sw:${share(x2 - x1, W)};--sh:${share(y2 - y1, H)};--sp:${(p || 0).toFixed(4)}"`
          + ` title="${esc(n + (s.labels[i] ? ` \u00b7 ${s.labels[i]}` : ''))}"></span>`
          + `<span class="blk-somtag ${esc(t[4])}${win}" data-n="${n}" style="--tx:${share(t[0], W)};`
          + `--ty:${share(t[1], H)};--tw:${share(t[2] - t[0], W)};--th:${share(t[3] - t[1], H)};`
          + `--tr:${((t[3] - t[1]) / W).toFixed(5)};--tc:${chars}${shown ? ';--tg:4px' : ''}">`
          + `<b>${n}</b>${shown ? `<i>${pct(p)}</i>` : ''}</span>`;
      }).join('');
      return `<figure class="blk-shot answered" data-act="${act}"><div class="blk-shot-frame"`
        + ` style="aspect-ratio:${W} / ${H}"><img src="${esc(media(s.img))}" alt="" draggable="false">`
        + `<div class="blk-marks">${marks}</div></div></figure>`;
    };

    const stepSide = (ep, steps, i) => {
      const s = steps[i];
      const v = verdict(s, (byId[ep.scenario] || {}).kind === 'game');
      const label = s.pick ? (s.labels[s.pick - 1] || '') : '';
      const top = s.p.map((p, k) => [k + 1, p]).filter((x) => x[1] != null)
        .sort((a, b) => b[1] - a[1]).slice(0, 5);
      const bars = top.map(([k, p], j) => {
        const name = s.labels[k - 1] || '';
        return `<li class="blk-opt${k === s.pick && s.act !== 'done' ? ' win' : ''}" style="--i:${j}">`
          + `<span class="k" title="${esc(name)}">${k}${name ? ` \u00b7 ${esc(name)}` : ''}</span>`
          + `<span class="track"><i style="width:${Math.max(p * 100, 0.6).toFixed(1)}%"></i></span>`
          + `<span class="v">${pct1(p)}</span></li>`;
      }).join('');
      const miss = s.oracle && s.pick && s.oracle !== s.pick && s.act !== 'done'
        ? `<p class="blk-cuax-miss">${esc(fmt(C.expected, { n: s.oracle }))}</p>` : '';
      return `<p class="blk-cuax-stepn"><b>${esc(fmt(C.step_of, { n: i + 1, total: steps.length }))}</b>`
        + `<span>${esc(fmt(C.ms, { ms: Math.round(s.ms).toLocaleString('en-US') }))}</span></p>`
        + `<div class="blk-sv" data-act="${v.act}"><span class="blk-sv-mark" aria-hidden="true">${v.mark}</span>`
        + `<div class="blk-sv-text"><h3>${esc(v.head)}</h3>${label ? `<p>${esc(label)}</p>` : ''}</div></div>`
        + miss + (bars ? `<ul class="blk-opts">${bars}</ul>` : '')
        + `<div class="blk-meters">${meter(C.done_q, s.done, s.done != null && s.done >= doneAt, 'done')}`
        + `${meter(C.risky_q, s.risky, s.risky != null && s.risky >= riskyAt, 'risky')}</div>`;
    };

    const dots = (item) => `<span class="blk-cuax-dots" aria-hidden="true">${item.outcomes
      .map((o) => `<i class="${o.ok ? 'ok' : 'no'}"></i>`).join('')}</span>`;

    const summary = (sc, item, ep) => {
      const st = item.stats;
      const models = G.models.filter((m) => sc.models[m]);
      const facts = [ep.steps_taken === 1 ? C.steps_taken_one : fmt(C.steps_taken, { n: ep.steps_taken })];
      if (ep.gates) facts.push(fmt(C.asked, { n: ep.gates }));
      if (ep.p50_ms) facts.push(fmt(C.per_step, { ms: Math.round(ep.p50_ms).toLocaleString('en-US') }));
      const lo = st.wilson95 ? Math.round(st.wilson95[0] * 100) : null;
      const hi = st.wilson95 ? Math.round(st.wilson95[1] * 100) : null;
      // the other run on offer, said as what it is: finished, failed, or simply another
      const alt = G.episodes[item[cur.role === 'failure' ? 'showcase' : 'failure']] || {};
      const other = cur.role === 'failure' ? (alt.success ? C.show_best : C.show_other)
        : (ep.success ? C.show_failed : C.show_other);
      const compare = models.length > 1 ? `<p class="blk-eyebrow">${esc(C.models)}</p><ul class="blk-cuax-cmp">${models
        .map((m) => {
          const ms = sc.models[m].stats;
          return `<li class="${m === cur.model ? 'on' : ''}"><button type="button" data-do="model" data-model="${esc(m)}">`
            + `<span>${esc(m)}</span><i style="--p:${(ms.rate || 0).toFixed(3)}"></i>`
            + `<b>${esc(fmt(C.runs, { ok: ms.successes, n: ms.episodes }))}</b></button></li>`;
        }).join('')}</ul>` : '';
      return `<p class="blk-cuax-outcome ${ep.success ? 'ok' : 'no'}"><span>${esc(ep.success ? C.finished : C.failed)}</span>`
        + `${!ep.success && ep.detail ? `<em>${esc(ep.detail)}</em>` : ''}</p>`
        + `<p class="blk-cuax-facts">${facts.map((f) => `<span>${esc(f)}</span>`).join('')}</p>`
        + `<div class="blk-cuax-runs">${dots(item)}<b>${esc(fmt(C.runs, { ok: st.successes, n: st.episodes }))}</b>`
        + (lo != null ? `<span>${esc(fmt(C.interval, { lo, hi }))}</span>` : '') + '</div>'
        + (st.step_accuracy != null ? `<p class="blk-cuax-acc">${esc(fmt(C.matched, { pct: Math.round(st.step_accuracy * 100) }))}</p>` : '')
        + (item.failure ? `<button type="button" class="blk-cuax-alt" data-do="run">${esc(other)}</button>` : '')
        + compare;
    };

    const rail = (ep) => (ep.rail || []).map((s, i) => {
      const cls = [s.a === 'done' ? 'done' : '', s.g ? 'gate' : '', s.m ? 'miss' : '',
        cur.view === 'steps' && i === cur.step ? 'on' : ''].filter(Boolean).join(' ');
      return `<li><button type="button" data-do="step" data-i="${i}" class="${cls}"`
        + ` aria-label="${esc(fmt(C.step_of, { n: i + 1, total: count(ep) }))}">${i + 1}</button></li>`;
    }).join('');

    const paint = () => {
      const player = stage.querySelector('.blk-cuax-player');
      if (!player || !cur) return;
      const sc = byId[cur.sid];
      const item = itemOf(cur);
      const ep = epOf(cur);
      player.dataset.view = cur.view;
      const screen = player.querySelector('.blk-cuax-screen');
      const side = player.querySelector('.blk-cuax-side');
      if (cur.view === 'video' && ep.video) {
        const [W, H] = ep.frame;
        if (!screen.querySelector('video')) {
          screen.innerHTML = `<div class="blk-cuax-vid" style="aspect-ratio:${W} / ${H}"><video controls playsinline muted`
            + ` preload="auto" poster="${esc(media(ep.poster))}"><source src="${esc(media(ep.video))}" type="video/mp4">`
            + '</video></div>';
          const v = screen.querySelector('video');
          v.muted = true;
          if (!reduce) {
            const p = v.play();
            if (p && p.catch) p.catch(() => {});
          }
        }
        side.innerHTML = summary(sc, item, ep);
      } else if (count(ep)) {
        const steps = ready[ep.steps_url];
        if (steps) {
          screen.innerHTML = figure(ep, steps[cur.step], screen.clientWidth < 520);
          side.innerHTML = stepSide(ep, steps, cur.step);
          [cur.step - 1, cur.step + 1].forEach((k) => {
            const s = steps[k];
            if (s && s.img) (new Image()).src = media(s.img);
          });
        } else {
          const [W, H] = ep.frame;
          screen.innerHTML = `<div class="blk-cuax-vid wait" style="aspect-ratio:${W} / ${H}">`
            + (ep.poster ? `<img src="${esc(media(ep.poster))}" alt="">` : '') + '</div>';
          side.innerHTML = '<p class="blk-cuax-load" aria-busy="true"><i></i><i></i><i></i></p>';
          const want = cur;
          stepsOf(ep).then((got) => {
            ready[ep.steps_url] = got;
            if (cur === want && got.length) paint();
          }).catch(() => {
            if (cur === want) side.innerHTML = `<p class="blk-note">${esc(C.no_steps)}</p>`;
          });
        }
      }
      player.querySelectorAll('.blk-cuax-rail button').forEach((b) => {
        b.classList.toggle('on', cur.view === 'steps' && Number(b.dataset.i) === cur.step);
      });
      player.querySelectorAll('[data-do="view"]').forEach((b) => {
        b.classList.toggle('on', b.dataset.view === cur.view);
        b.setAttribute('aria-pressed', String(b.dataset.view === cur.view));
      });
      const at = player.querySelector('.blk-cuax-rail .on');
      if (at && at.scrollIntoView) at.scrollIntoView({ block: 'nearest', inline: 'nearest' });
    };

    const render = () => {
      const sc = byId[cur.sid];
      const ep = epOf(cur);
      const [W, H] = ep.frame;
      const tall = H > W;
      stage.innerHTML = `<div class="blk-cuax-player${tall ? ' tall' : ''}" data-view="${cur.view}" tabindex="-1">`
        + '<div class="blk-cuax-bar">'
        + `<button type="button" class="blk-cuax-x" data-do="close">${BACK}<span>${esc(C.close)}</span></button>`
        + `<h3 class="blk-cuax-name">${esc(sc.title)}</h3>`
        + `<span class="blk-cuax-who">${esc(cur.model)}${cur.role === 'failure' ? ` \u00b7 ${esc(fmt(C.seed, { seed: ep.seed }))}` : ''}</span>`
        + '</div>'
        + `<p class="blk-cuax-task"><span>${esc(C.task)}</span>${esc(ep.task)}</p>`
        + '<div class="blk-cuax-body"><div class="blk-cuax-screen"></div><aside class="blk-cuax-side"></aside>'
        + '<div class="blk-cuax-nav">'
        + `<div class="blk-cuax-views" role="group">${ep.video ? `<button type="button" data-do="view" data-view="video">${esc(C.video)}</button>` : ''}`
        + `<button type="button" data-do="view" data-view="steps">${esc(C.steps)}</button></div>`
        + '<div class="blk-cuax-walk">'
        + `<button type="button" class="blk-cuax-step blk-cuax-autoplay" data-do="play" aria-pressed="false"`
        + ` aria-label="${esc(C.play)}">${PLAY}</button>`
        + `<button type="button" class="blk-cuax-step" data-do="prev" aria-label="${esc(C.prev)}">${PREV}</button>`
        + `<ol class="blk-cuax-rail">${rail(ep)}</ol>`
        + `<button type="button" class="blk-cuax-step" data-do="next" aria-label="${esc(C.next)}">${NEXT}</button>`
        + '</div></div></div></div>';
      paint();
    };

    const mark = () => root.querySelectorAll('.blk-cuax-card').forEach((c) => {
      c.classList.toggle('on', !!cur && c.dataset.sid === cur.sid);
    });

    const open = (o, scroll) => {
      const sc = byId[o.sid];
      if (!sc) return;
      halt();
      const models = G.models.filter((m) => sc.models[m]);
      const lead = G.lead_model || G.vision_model;
      const model = models.includes(o.model) ? o.model : (models.includes(lead) ? lead : models[0]);
      const item = sc.models[model];
      const role = o.role === 'failure' && item.failure ? 'failure' : 'showcase';
      const ep = G.episodes[item[role]];
      const steps = count(ep);
      const view = ep.video ? (o.view || 'video') : 'steps';
      cur = { sid: sc.id, model, role, view, step: Math.min(Math.max(o.step || 0, 0), Math.max(steps - 1, 0)) };
      memo.open = cur;
      render();
      mark();
      sync();
      if (scroll) {
        const top = stage.getBoundingClientRect().top;
        if (top < 0 || top > window.innerHeight * 0.4) {
          stage.scrollIntoView({ behavior: reduce ? 'auto' : 'smooth', block: 'start' });
        }
      }
    };

    // the steps can play on their own, one decision at a time; anything else a visitor does stops them
    const halt = () => {
      if (timer) clearInterval(timer);
      timer = null;
      const b = stage.querySelector('.blk-cuax-autoplay');
      if (b) {
        b.innerHTML = PLAY;
        b.setAttribute('aria-pressed', 'false');
        b.setAttribute('aria-label', C.play);
      }
    };
    const play = () => {
      if (timer) {
        halt();
        return;
      }
      const ep = epOf(cur);
      if (!count(ep)) return;
      go(cur.view === 'steps' && cur.step < count(ep) - 1 ? cur.step : 0);
      const b = stage.querySelector('.blk-cuax-autoplay');
      if (b) {
        b.innerHTML = PAUSE;
        b.setAttribute('aria-pressed', 'true');
        b.setAttribute('aria-label', C.pause);
      }
      timer = setInterval(() => {
        if (!cur || cur.step >= count(epOf(cur)) - 1) halt();
        else go(cur.step + 1);
      }, 1700);
    };

    const close = () => {
      halt();
      cur = null;
      memo.open = null;
      stage.innerHTML = heroHTML;
      loops(stage);
      mark();
      sync();
    };

    const go = (i) => {
      const ep = epOf(cur);
      if (!count(ep)) return;
      cur.step = Math.min(Math.max(i, 0), count(ep) - 1);
      cur.view = 'steps';
      paint();
      sync();
    };

    root.addEventListener('click', (e) => {
      const card = e.target.closest('[data-sid]');
      if (card && root.contains(card)) {
        if (e.metaKey || e.ctrlKey || e.shiftKey || e.button) return;
        e.preventDefault();
        open({ sid: card.dataset.sid }, true);
        return;
      }
      const btn = e.target.closest('[data-do]');
      if (!btn || !cur) return;
      const act = btn.dataset.do;
      if (act === 'play') {
        play();
        return;
      }
      halt();
      if (act === 'close') close();
      else if (act === 'step') go(Number(btn.dataset.i));
      else if (act === 'prev') go((cur.view === 'steps' ? cur.step : 0) - 1);
      else if (act === 'next') go(cur.view === 'steps' ? cur.step + 1 : 0);
      else if (act === 'view') {
        cur.view = btn.dataset.view;
        const screen = stage.querySelector('.blk-cuax-screen');
        if (screen) screen.innerHTML = '';
        paint();
        sync();
      } else if (act === 'model') open({ sid: cur.sid, model: btn.dataset.model, view: cur.view }, false);
      else if (act === 'run') open({ sid: cur.sid, model: cur.model, role: cur.role === 'failure' ? 'showcase' : 'failure', view: cur.view }, false);
    });

    root.addEventListener('keydown', (e) => {
      if (!cur || !e.target.closest('.blk-cuax-player')) return;
      if (['ArrowRight', 'ArrowLeft', 'Escape'].includes(e.key)) halt();
      if (e.key === 'ArrowRight') { e.preventDefault(); go(cur.view === 'steps' ? cur.step + 1 : 0); }
      else if (e.key === 'ArrowLeft') { e.preventDefault(); go(cur.view === 'steps' ? cur.step - 1 : 0); }
      else if (e.key === 'Escape') close();
    });

    // a link opens its run once, on the page's first load; coming back to the tab keeps what was open
    const sid = (q.get('scenario') || '').trim().toLowerCase();
    if (sid && byId[sid] && firstLoad) {
      const step = parseInt(q.get('step') || '', 10);
      open({ sid, model: q.get('model') || '', role: /^fail/i.test(q.get('run') || '') ? 'failure' : 'showcase',
        view: Number.isFinite(step) ? 'steps' : '', step: Number.isFinite(step) ? step - 1 : 0 }, true);
    } else if (memo.open && byId[memo.open.sid]) {
      open(memo.open, false);
    }
  };

  boot(0);
})();
"""

# Beyond clicks: one tile open at a time.
TILES_JS = r"""(() => {
  // Beyond clicks: one tile open at a time, its video or chart under the row. A video with sound
  // never starts on its own; one without plays muted.
  const esc = (s) => String(s == null ? '' : s).replace(/[&<>"']/g,
    (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const boot = (tries) => {
    const box = element.querySelector('.blk-cuax-more');
    if (!box) {
      if (tries < 50) setTimeout(() => boot(tries + 1), 100);
      return;
    }
    if (box.__blink) return;
    box.__blink = true;
    let T = {};
    try {
      T = JSON.parse(box.dataset.t || '{}');
    } catch (err) {
      return;
    }
    const items = T.more || [];
    const C = T.copy || {};
    const media = (rel) => (rel ? (T.media_base || '') + encodeURI(rel) : '');
    const pane = box.querySelector('.blk-cuax-open');
    let at = null;
    const show = (id) => {
      const t = items.find((x) => x.id === id);
      const again = at === id;
      box.querySelectorAll('.blk-cuax-tile').forEach((b) => {
        const on = b.dataset.tile === id && !again;
        b.classList.toggle('on', on);
        b.setAttribute('aria-expanded', String(on));
      });
      if (!t || again) {
        at = null;
        pane.hidden = true;
        pane.innerHTML = '';
        return;
      }
      at = id;
      let fig = '';
      if (t.video) {
        const track = t.captions ? `<track kind="captions" src="${esc(media(t.captions))}" srclang="en" default>` : '';
        const cors = t.captions ? ' crossorigin="anonymous"' : '';
        fig += t.sound
          ? `<video controls playsinline preload="metadata"${cors} poster="${esc(media(t.poster))}">`
            + `<source src="${esc(media(t.video))}" type="video/mp4">${track}</video>`
            + `<p class="blk-cuax-sound">${esc(C.sound)}</p>`
          : `<video controls playsinline muted loop preload="metadata"${cors} poster="${esc(media(t.poster))}">`
            + `<source src="${esc(media(t.video))}" type="video/mp4">${track}</video>`;
      }
      if (t.chart) fig += `<img class="blk-cuax-chart" src="${esc(media(t.chart))}" alt="${esc(t.title)}">`;
      if (t.images && t.images.length) {
        fig += `<div class="blk-cuax-imgs">${t.images.map((i) => `<img src="${esc(media(i))}" alt="" loading="lazy">`).join('')}</div>`;
      }
      pane.innerHTML = `<div class="blk-cuax-pane"><h4>${esc(t.title)}</h4><p class="blk-note">${esc(t.line)}</p>${fig}</div>`;
      pane.hidden = false;
      const v = pane.querySelector('video[muted]');
      if (v) {
        v.muted = true;
        const p = v.play();
        if (p && p.catch) p.catch(() => {});
      }
    };
    box.addEventListener('click', (e) => {
      const b = e.target.closest('.blk-cuax-tile');
      if (b) show(b.dataset.tile);
    });
  };
  boot(0);
})();
"""
