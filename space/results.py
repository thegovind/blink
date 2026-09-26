"""Leaderboard figures, drawn as inline SVG from results.json. No network, no chart library."""

from __future__ import annotations

import html
import json
import math
import os
from functools import lru_cache

HERE = os.path.dirname(os.path.abspath(__file__))
# BLINK_RESULTS points the figures at another results file, for previewing a block that
# has not been measured yet. Nothing in the deployment sets it.
RESULTS_PATH = os.environ.get("BLINK_RESULTS") or os.path.join(HERE, "results.json")

INK = "#0a0a0a"
SOFT = "#6b6b6b"
MUTED = "#737373"
LINE = "#e5e5e5"
GRID = "#ededed"
# blink's own marks are cobalt; every system it is compared with stays gray, so the
# colour means "this is blink" wherever it appears (style.css carries the same values)
OURS = "#2447f5"
OURS_DEEP = "#15237a"
OURS_LIGHT = "#6a82f8"
REFERENCE = "#6b6b6b"
OPEN = "#8c8c8c"
OPEN_CYCLE = [OPEN]
AXIS_COLORS = [MUTED]

# how many leaderboard entries the focused view keeps below ours and the reference
TOP_OPEN = 8
# the size class we always keep a representative of, so the small end stays visible
CLASS_B = (3.5, 5.5)


@lru_cache(maxsize=1)
def load() -> dict:
    with open(RESULTS_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def _e(s) -> str:
    return html.escape(str(s), quote=True)


def _f(v: float, n: int = 2) -> str:
    return f"{v:.{n}f}"


def _tone(kind: str) -> str:
    return {"ours": OURS, "reference": REFERENCE, "open": OPEN}.get(kind, MUTED)


def _clip(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1].rstrip() + "\u2026"


def _svg(body: str, w: int, h: int, label: str) -> str:
    return (
        f'<svg class="blk-fig" viewBox="0 0 {w} {h}" width="100%" '
        f'preserveAspectRatio="xMidYMid meet" role="img" aria-label="{_e(label)}">{body}</svg>'
    )


# --- decision index -----------------------------------------------------------------


def focus(systems: list[dict]) -> list[dict]:
    """Ours and the reference always; then the best open entries and the 4B class best."""
    ranked = sorted(systems, key=lambda s: -s["index"])
    keep = [s for s in ranked if s["kind"] != "open"]
    opens = [s for s in ranked if s["kind"] == "open"]
    keep += opens[:TOP_OPEN]
    in_class = next(
        (s for s in opens if s.get("served_params") and CLASS_B[0] <= s["served_params"] / 1e9 <= CLASS_B[1]),
        None,
    )
    if in_class is not None and in_class not in keep:
        keep.append(in_class)
    return sorted(keep, key=lambda s: -s["index"])


def _index_bars(shown: list[dict], ref: dict | None) -> str:
    gutter, bw, w = 206, 640, 1060
    x0 = gutter
    top, rowh = 40, 32
    h = top + rowh * len(shown) + 18
    lo, hi = 0, int(math.ceil(max(s["index"] for s in shown) / 10.0) * 10)
    p = []

    def sx(v: float) -> float:
        return x0 + bw * (v - lo) / (hi - lo)

    for t in range(lo, hi + 1, 10):
        p.append(f'<line class="blk-grid" x1="{sx(t):.1f}" y1="{top - 12}" x2="{sx(t):.1f}" y2="{h - 16}"/>')
        p.append(f'<text class="blk-ax" x="{sx(t):.1f}" y="{top - 18}" text-anchor="middle">{t}</text>')
    p.append(f'<text class="blk-ax" x="{w - 4}" y="{top - 18}" text-anchor="end">skill \u00b7 breadth</text>')
    if ref:  # drawn before the bars so the value labels keep their halo
        rx = sx(ref["index"])
        p.append(f'<line class="blk-ref" x1="{rx:.1f}" y1="{top - 12}" x2="{rx:.1f}" y2="{h - 16}"/>')

    for i, s in enumerate(shown):
        y = top + i * rowh
        ours = s["kind"] == "ours"
        p.append(
            f'<text class="blk-name{" on" if ours else ""}" x="{gutter - 14}" y="{y + 15}" '
            f'text-anchor="end">{_e(_clip(s["name"], 26))}<title>{_e(s["name"])}</title></text>'
        )
        p.append(f'<text class="blk-ax" x="{gutter - 14}" y="{y + 27}" text-anchor="end">{_e(s["params"])}</text>')
        x1 = sx(s["index"])
        p.append(
            f'<rect class="blk-grow" style="animation-delay:{i * 70}ms" x="{x0}" y="{y + 3}" '
            f'width="{x1 - x0:.1f}" height="15" rx="4" fill="{_tone(s["kind"])}"/>'
        )
        p.append(
            f'<text class="blk-val{" on" if ours else ""}" x="{x1 + 9:.1f}" y="{y + 15}">{_f(s["index"])}</text>'
        )
        p.append(
            f'<text class="blk-ax" x="{w - 4}" y="{y + 15}" text-anchor="end">'
            f'{_f(s["skill"], 1)} \u00b7 {_f(s["breadth"], 1)}</text>'
        )

    return _svg("".join(p), w, h, "Decision Index by system")


def _area_bars(shown: list[dict], areas: list[dict]) -> str:
    w, gutter, rowh = 1060, 206, 32
    gap = 12
    colw = (w - gutter - gap * (len(areas) - 1)) / len(areas)
    barw = colw - 34
    h = 36 + rowh * len(shown) + 6
    q = []
    for j, area in enumerate(areas):
        cx = gutter + j * (colw + gap)
        q.append(
            f'<text class="blk-cap" x="{cx:.1f}" y="18">{_e(area["label"])}</text>'
            f'<line class="blk-grid" x1="{cx:.1f}" y1="24" x2="{cx + colw:.1f}" y2="24"/>'
        )
        vals = [s["areas"][area["id"]] for s in shown if s.get("areas")]
        amax = max(vals) if vals else 1.0
        for i, s in enumerate(shown):
            y = 36 + i * rowh
            if not s.get("areas"):
                q.append(f'<text class="blk-ax" x="{cx:.1f}" y="{y + 15}">\u2014</text>')
                continue
            v = s["areas"][area["id"]]
            bwid = barw * v / amax
            ours = s["kind"] == "ours"
            q.append(
                f'<rect class="blk-grow" style="animation-delay:{(j * 6 + i) * 30}ms" x="{cx:.1f}" y="{y + 4}" '
                f'width="{bwid:.1f}" height="13" rx="3" fill="{OURS if ours else OPEN}"/>'
            )
            q.append(f'<text class="blk-ax" x="{cx + bwid + 6:.1f}" y="{y + 15}">{_f(v, 1)}</text>')

    for i, s in enumerate(shown):
        y = 36 + i * rowh
        ours = s["kind"] == "ours"
        q.append(
            f'<text class="blk-name{" on" if ours else ""}" x="{gutter - 14}" y="{y + 15}" '
            f'text-anchor="end">{_e(_clip(s["name"], 26))}<title>{_e(s["name"])}</title></text>'
        )
    return _svg("".join(q), w, h, "Decision Index by area")


def _index_table(systems: list[dict]) -> str:
    rows = []
    for i, s in enumerate(sorted(systems, key=lambda s: -s["index"]), start=1):
        cls = ' class="on"' if s["kind"] == "ours" else ""
        rows.append(
            f"<tr{cls}><td>{i}</td><td>{_e(s['name'])}</td><td>{_e(s['params'])}</td>"
            f"<td>{_f(s['index'])}</td><td>{_f(s['skill'])}</td><td>{_f(s['breadth'])}</td></tr>"
        )
    return (
        '<details class="blk-d"><summary>Full board: index, skill and breadth</summary>'
        '<div class="blk-tablewrap"><table class="blk-table ranked">'
        "<thead><tr><th>#</th><th>System</th><th>Size</th><th>Index</th><th>Skill</th>"
        "<th>Breadth</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
        "<p>Index averages native benchmark metrics within five equal-weight areas. Skill "
        "measures performance above each metric's chance baseline. Breadth is a shifted "
        "geometric mean of the five area skill scores.</p>"
        "</details>"
    )


def _overlap_note(d: dict) -> str:
    """Where the training data touches the suite. Collapsed, but always one click away."""
    ref = next((s for s in d["systems"] if s["id"] == d.get("reference")), None)
    ours = [s for s in d["systems"] if s["kind"] == "ours" and s.get("index_language_equalized")]
    equalized = ""
    if ref and ours:
        equalized = " ".join(
            f"With Language set to {ref['name'].split()[0]}'s score, {_e(s['name'])} is at "
            f"{_f(s['index_language_equalized'])} against {_e(ref['name'].split()[0])} "
            f"{_f(ref['index'])}."
            for s in ours
        )
    return (
        '<details class="blk-d"><summary>Training overlap with the suite</summary>'
        "<p>All three used the public train splits of ContractNLI, iSarcasmEval and VAST "
        "(the whole Language area), plus Amazon ESCI and Humicroedit. blink-4b and "
        "blink-mimo-9b also used GSM8K-train solution-checking rows. blink-27b and "
        "blink-mimo-9b used searchless_chess training positions for ChessBench; none of "
        "its test positions.</p>"
        "<p>Every final training row was audited for exact normalised strings of 30 or more "
        "characters in any field and shared 13-word passages in question text. Strings or "
        "passages seen in 20 or more suite requests were treated as templates and ignored. "
        "Under these checks blink-4b had no suite content match; 16 rows in each of blink-27b "
        "and blink-mimo-9b shared a passage with 31 suite requests from VAST's and BANKING77's "
        "own train/test splits. Removing those requests leaves both indexes unchanged. "
        "The passage check did not search long-document bodies or option text.</p>"
        f"<p>Most of blink-27b's lead over Jev is in Language, which those splits "
        f"cover. {equalized}</p>"
        "<p>Scores are local runs of the official kit, not leaderboard submissions.</p>"
        "</details>"
    )


def decision_index_chart(data: dict) -> str:
    d = data["decision_index"]
    systems = d["systems"]
    shown = focus(systems)
    ref = next((s for s in shown if s["id"] == d.get("reference")), None)
    n_more = len(systems) - len(shown)
    tail = f" \u00b7 {n_more} more below" if n_more > 0 else ""
    return (
        f'<figure class="blk-figure"><figcaption>Index \u00b7 higher is better{_e(tail)}</figcaption>'
        f"{_index_bars(shown, ref)}</figure>"
        f'<figure class="blk-figure"><figcaption>Five areas</figcaption>'
        f'{_area_bars(shown, d["areas"])}</figure>'
        f"{_overlap_note(d)}"
        f"{_index_table(systems)}"
    )


JB_OFFICIAL_TITLE = "Official JevBench v1.4 \u00b7 full evaluation incl. sealed items"
JB_PUBLIC_TITLE = "Public items \u00b7 development proxy"


# --- decision index 0.2, on the benchmarks both editions share ---------------------

V02_TOP = 8


def _top_rows(systems: list[dict], key: str, n: int) -> list[dict]:
    """The best n, with ours and the reference kept whatever they score."""
    ranked = sorted(systems, key=lambda s: -s[key])
    keep = ranked[:n] + [s for s in ranked if s["kind"] != "open"]
    seen, out = set(), []
    for s in ranked:
        if s["id"] in seen or s not in keep:
            continue
        seen.add(s["id"])
        out.append(s)
    return out


def v02_partial_chart(data: dict) -> str:
    """Null until the shared-benchmark run exists; the section simply is not there."""
    d = data.get("decision_index_v02_partial")
    if not d:
        return ""
    shown = _top_rows(d["systems"], "partial", V02_TOP)
    ref = next((s for s in shown if s["kind"] == "reference"), None)
    gutter, bw, w = 250, 560, 1060
    top, rowh = 40, 30
    h = top + rowh * len(shown) + 18
    lo, hi = 0, int(math.ceil(max(s["partial"] for s in shown) / 10.0) * 10)
    p = []

    def sx(v: float) -> float:
        return gutter + bw * (v - lo) / (hi - lo)

    for t in range(lo, hi + 1, 10):
        p.append(f'<line class="blk-grid" x1="{sx(t):.1f}" y1="{top - 12}" x2="{sx(t):.1f}" y2="{h - 16}"/>')
        p.append(f'<text class="blk-ax" x="{sx(t):.1f}" y="{top - 18}" text-anchor="middle">{t}</text>')
    p.append(f'<text class="blk-ax" x="{w - 4}" y="{top - 18}" text-anchor="end">0.2 official</text>')
    if ref:
        rx = sx(ref["partial"])
        p.append(f'<line class="blk-ref" x1="{rx:.1f}" y1="{top - 12}" x2="{rx:.1f}" y2="{h - 16}"/>')

    for i, s in enumerate(shown):
        y = top + i * rowh
        ours = s["kind"] == "ours"
        p.append(
            f'<text class="blk-name{" on" if ours else ""}" x="{gutter - 14}" y="{y + 15}" '
            f'text-anchor="end">{_e(_clip(s["name"], 32))}<title>{_e(s["name"])}</title></text>'
        )
        x1 = sx(s["partial"])
        p.append(
            f'<rect class="blk-grow" style="animation-delay:{i * 70}ms" x="{gutter}" y="{y + 3}" '
            f'width="{x1 - gutter:.1f}" height="14" rx="4" fill="{_tone(s["kind"])}"/>'
        )
        p.append(
            f'<text class="blk-val{" on" if ours else ""}" x="{x1 + 9:.1f}" y="{y + 15}">'
            f'{_f(s["partial"])}</text>'
        )
        official = s.get("official_02")
        p.append(
            f'<text class="blk-ax" x="{w - 4}" y="{y + 15}" text-anchor="end">'
            f'{_f(official) if official is not None else "not on the board"}</text>'
        )

    svg = _svg("".join(p), w, h, "Mean over the benchmarks 0.1 and 0.2 share")
    missing = ", ".join(_e(m) for m in d.get("missing", []))
    gap = (
        f'<details class="blk-d"><summary>The {len(d.get("missing", []))} benchmarks this '
        f"comparison cannot see</summary><p>{missing}.</p>"
        "<p>ToolRet and BRIGHT are scored on the full 0.1 sets rather than the subsets 0.2 "
        "uses, so even the shared part is not identical.</p></details>"
        if d.get("missing")
        else ""
    )
    return (
        f'<figure class="blk-figure"><figcaption>{d["shared"]} of {d["panel"]} panel '
        f"benchmarks \u00b7 the same subset for every system \u00b7 not a 0.2 score</figcaption>"
        f"{svg}</figure>{gap}"
    )


# --- jevbench: two groups that are never put on one scale ---------------------------


def _radar(rows: list[tuple[str, str, dict]], axes: list[dict], label: str) -> str:
    """rows: (name, colour, axis values). Drawn on a fixed 0-100 grid."""
    w, h = 460, 384
    cx, cy, R = 230, 186, 120
    n = len(axes)
    ang = [(-math.pi / 2) + 2 * math.pi * i / n for i in range(n)]

    def pt(i: int, v: float) -> tuple[float, float]:
        r = R * max(0.0, min(1.0, v / 100.0))
        return cx + r * math.cos(ang[i]), cy + r * math.sin(ang[i])

    p, rings = [], []
    for ring in (25, 50, 75, 100):
        poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in (pt(i, ring) for i in range(n)))
        p.append(f'<polygon class="blk-ring" points="{poly}"/>')
        rx, ry = pt(1, ring)
        rings.append(f'<text class="blk-ring-lab" x="{rx - 3:.1f}" y="{ry - 7:.1f}" text-anchor="end">{ring}</text>')
    for i in range(n):
        x, y = pt(i, 100)
        p.append(f'<line class="blk-spoke" x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}"/>')
        lx, ly = cx + (R + 30) * math.cos(ang[i]), cy + (R + 30) * math.sin(ang[i])
        anchor = "middle" if abs(math.cos(ang[i])) < 0.3 else ("start" if math.cos(ang[i]) > 0 else "end")
        p.append(
            f'<text class="blk-cap" x="{lx:.1f}" y="{ly + 4:.1f}" text-anchor="{anchor}" '
            f'fill="{AXIS_COLORS[i % len(AXIS_COLORS)]}">{_e(axes[i]["label"])}</text>'
        )
    for k, (_, col, vals) in enumerate(reversed(rows)):
        lead = k == len(rows) - 1
        pts = [pt(i, vals[a["id"]]) for i, a in enumerate(axes)]
        poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        p.append(
            f'<polygon class="blk-radar" style="animation-delay:{k * 140}ms" points="{poly}" '
            f'fill="{col if lead else "none"}" fill-opacity="{0.10 if lead else 0}" '
            f'stroke="{col}" stroke-width="{2.6 if lead else 1.5}"/>'
        )
        if lead:
            for x, y in pts:
                p.append(f'<circle class="blk-dot" cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{col}"/>')
    return _svg("".join(p) + "".join(rings), w, h, label)


def _jb_row(name: str, col: str, score, axes: list[dict], vals: dict | None, tag: str, ours: bool) -> str:
    strong = _f(score) if score is not None else "\u2014"
    body = (
        f'<div class="blk-axvals">'
        + " ".join(f'<span><i>{_e(a["label"][:3])}</i>{_f(vals[a["id"]], 1)}</span>' for a in axes)
        + "</div>"
        if vals
        else ""
    )
    tag_html = f'<em class="blk-tag">{_e(tag)}</em>' if tag else ""
    return (
        f'<li class="blk-row{" on" if ours else ""}{"" if score is not None else " quiet"}">'
        f'<b style="--c:{col}">{_e(name)}{tag_html}</b><strong>{strong}</strong>{body}</li>'
    )


def _jb_group(title: str, note: str, rows: list[str], radar: str) -> str:
    return (
        f'<section class="blk-group"><h3 class="blk-group-t">{_e(title)}</h3>'
        f'<p class="blk-group-s">{_e(note)}</p>{radar}'
        f'<ul class="blk-rows">{"".join(rows)}</ul></section>'
    )


def _public_notes(notes: dict) -> str:
    lines = []
    hc = notes.get("hard_correct") or {}
    if hc.get("n"):
        lines.append(
            f"In our runtime blink-4b answers {hc['blink-4b']}/{hc['n']} public "
            f"hard items and JevK5 {hc['jevk5-0.2.0']}/{hc['n']}; JevK5's own runtime reports "
            f"{hc['jevk5_native_published']}/{hc['n']}. No hard-accuracy advantage is claimed."
        )
    for key in ("selection_note", "speed_note", "cost_note"):
        if notes.get(key):
            lines.append(notes[key])
    return (
        '<details class="blk-d"><summary>Public-item details</summary>'
        + "".join(f'<p>{_e(t)}</p>' for t in lines)
        + "</details>"
    ) if lines else ""


def _public_table(systems: list[dict], notes: dict | None = None) -> str:
    measured = [s for s in systems if s.get("public")]
    if not measured:
        return ""
    rows = []
    for s in measured:
        pub = s["public"]
        cls = ' class="on"' if s["kind"] == "ours" else ""
        rows.append(
            f"<tr{cls}><td>{_e(s['name'])}</td><td>{pub['standard'] * 100:.1f}%</td>"
            f"<td>{pub['hard'] * 100:.1f}%</td><td>{pub['hard_ece']:.3f}</td><td>{pub['tvd']:.3f}</td></tr>"
        )
    return (
        '<div class="blk-tablewrap blk-mini"><table class="blk-table">'
        "<thead><tr><th>Public items</th><th>Standard</th><th>Hard</th><th>Hard ECE</th>"
        "<th>TVD</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
        '<p class="blk-note blk-undertable">Hard accuracy: higher is better. '
        "ECE and TVD: lower is better.</p>"
        + _public_notes(notes or {})
    )


def jevbench_chart(data: dict) -> str:
    d = data["jevbench"]
    axes = d["axes"]
    systems = d["systems"]

    colour, spare = {}, list(OPEN_CYCLE)
    for s in systems:
        if s["kind"] == "ours":
            colour[s["id"]] = OURS
        elif s["kind"] == "reference":
            colour[s["id"]] = REFERENCE
        else:
            colour[s["id"]] = spare.pop(0) if spare else MUTED

    official = sorted(
        [s for s in systems if s.get("official")], key=lambda s: -s["official"]["score"]
    )
    missing = [s for s in systems if not s.get("official")]
    public = sorted(
        [s for s in systems if s.get("public_estimate") is not None], key=lambda s: -s["public_estimate"]
    )

    off_rows = [
        _jb_row(s["name"], colour[s["id"]], s["official"]["score"], axes, s["official"]["axes"], "", s["kind"] == "ours")
        for s in official
    ]
    off_rows += [
        _jb_row(s["name"], colour[s["id"]], None, axes, None, "not submitted", s["kind"] == "ours")
        for s in missing
    ]
    pub_rows = [
        _jb_row(s["name"], colour[s["id"]], s["public_estimate"], axes, s["public_axes"], "proxy", s["kind"] == "ours")
        for s in public
    ]

    off_radar = _radar(
        [(s["name"], colour[s["id"]], s["official"]["axes"]) for s in official], axes, "Official axes"
    ) if official else ""
    pub_radar = _radar(
        [(s["name"], colour[s["id"]], s["public_axes"]) for s in public], axes, "Public-item axes"
    ) if public else ""

    return (
        '<div class="blk-duo">'
        + _jb_group(
            JB_OFFICIAL_TITLE,
            "Full evaluation, including private items. blink has not been submitted.",
            off_rows,
            off_radar,
        )
        + _jb_group(
            JB_PUBLIC_TITLE,
            "Same public items and harness for both rows. Not an official score.",
            pub_rows,
            pub_radar,
        )
        + "</div>"
        + _public_table(systems, d.get("notes"))
    )


# --- size / quality pareto ----------------------------------------------------------

TICK_MULTIPLES = {"params_b": (1, 3), "latency_ms": (1, 3), "cost_per_1000": (1, 2, 5)}

# One line per chart, kept together so they are easy to rewrite.
PARETO_CAPTIONS = {
    "pareto": "Smaller is left \u00b7 higher Decision Index is up",
    "speed_pareto": "Faster is left \u00b7 higher Decision Index is up",
}


def _fmt_x(key: str, v: float) -> str:
    if key == "params_b":
        if v >= 1:
            return f"{round(v, 1):g}B"
        return f"{v * 1000:.0f}M"
    if key.endswith("_ms"):
        return f"{round(v / 1000, 1):g} s" if v >= 1000 else f"{round(v):g} ms"
    if key.startswith("cost"):
        return f"${v:g}".replace("$0.", "$.")
    return f"{v:g}"


def _log_ticks(key: str, lx0: float, lx1: float) -> list[float]:
    mults = TICK_MULTIPLES.get(key, (1, 3))
    out = []
    for e in range(math.floor(lx0) - 1, math.ceil(lx1) + 2):
        for m in mults:
            v = m * 10.0**e
            if lx0 <= math.log10(v) <= lx1:
                out.append(v)
    return out


# label geometry, shared with the tests that check nothing overlaps
LAB_GAP = 14  # clear space between a dot and its label
LAB_TOP = 11  # text baseline to box top
LAB_BOT = 4  # text baseline to box bottom
DOT_PAD = 3


def label_width(name: str, tail: str) -> float:
    """Rough advance width of "name  tail" at the sizes .blk-pt-lab and .blk-ax use."""
    return len(name) * 7.1 + len(tail) * 6.4 + 22


def _overlap(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _seg_hits(seg, r) -> bool:
    """Axis-aligned segment against a rect; the staircase is only ever horizontal or vertical."""
    x0, y0, x1, y1 = seg
    if abs(y0 - y1) < 0.01:
        return r[1] <= y0 <= r[3] and min(x0, x1) <= r[2] and r[0] <= max(x0, x1)
    return r[0] <= x0 <= r[2] and min(y0, y1) <= r[3] and r[1] <= max(y0, y1)


def _candidates(x: float, y: float, width: float):
    """Anchors around a point, nearest first: right, left, then centred above and below."""
    for dy in (0, -18, 20, -36, 38, -54, 56, -74, 76):
        yield x + LAB_GAP, y + 4 + dy
        yield x - LAB_GAP - width, y + 4 + dy
    for dy in (-24, 28, -44, 48):
        yield x - width / 2, y + 4 + dy


def _place(x, y, width, frame, taken, dots, own, segs):
    """First anchor that clears every other label, every other dot, and the frontier."""
    for x1, ly in _candidates(x, y, width):
        box = (x1, ly - LAB_TOP, x1 + width, ly + LAB_BOT)
        if not (frame[0] <= box[0] and box[2] <= frame[2] and frame[1] <= box[1] and box[3] <= frame[3]):
            continue
        if any(_overlap(box, o) for o in taken):
            continue
        if any(i != own and _overlap(box, rect) for i, rect in dots.items()):
            continue
        if any(_seg_hits(seg, box) for seg in segs):
            continue
        return box, ly
    return None, None


def speed_pareto_chart(data: dict) -> str:
    """Null until the latency run lands, and then the same chart on a time axis."""
    return pareto_chart(data, "speed_pareto") if data.get("speed_pareto") else ""


def pareto_chart(data: dict, key: str = "pareto") -> str:
    d = data[key]
    xk = d["x_key"]
    pts = d["points"]
    frontier_ids = set(d["frontier"])
    ref = d.get("reference")
    w, h = 1060, 500
    l, r, t, b = 74, 200, 34, 64
    xs = [p_[xk] for p_ in pts]
    ys = [p_["score"] for p_ in pts] + ([ref["score"]] if ref else [])
    lx0, lx1 = math.log10(min(xs) * 0.62), math.log10(max(xs) * 1.9)
    y0 = max(0, math.floor((min(ys) - 3) / 5) * 5)
    y1 = math.ceil((max(ys) + 3) / 5) * 5

    def X(v: float) -> float:
        return l + (w - l - r) * (math.log10(v) - lx0) / (lx1 - lx0)

    def Y(v: float) -> float:
        return (h - b) - (h - b - t) * (v - y0) / (y1 - y0)

    p = []
    for tick in _log_ticks(xk, lx0, lx1):
        x = X(tick)
        p.append(f'<line class="blk-grid" x1="{x:.1f}" y1="{t}" x2="{x:.1f}" y2="{h - b}"/>')
        p.append(
            f'<text class="blk-ax" x="{x:.1f}" y="{h - b + 18}" text-anchor="middle">'
            f"{_e(_fmt_x(xk, tick))}</text>"
        )
    for tick in range(int(math.ceil(y0 / 10.0) * 10), int(y1) + 1, 10):
        y = Y(tick)
        p.append(f'<line class="blk-grid" x1="{l}" y1="{y:.1f}" x2="{w - r}" y2="{y:.1f}"/>')
        p.append(f'<text class="blk-ax" x="{l - 12}" y="{y + 4:.1f}" text-anchor="end">{tick}</text>')
    p.append(f'<text class="blk-cap" x="{l}" y="{h - b + 42}">{_e(d["x_label"])} \u00b7 log scale</text>')
    p.append(
        f'<text class="blk-cap" x="{-(t + (h - b - t) / 2):.1f}" y="18" transform="rotate(-90)" '
        f'text-anchor="middle">{_e(d["y_label"])}</text>'
    )

    taken: list[tuple[float, float, float, float]] = []
    labels, leads, dots = [], [], []

    if ref:
        ry = Y(ref["score"])
        p.append(f'<line class="blk-ref" x1="{l}" y1="{ry:.1f}" x2="{w - r}" y2="{ry:.1f}"/>')
        rw = len(ref["name"]) * 6.2 + 12
        labels.append(
            f'<text class="blk-ax" x="{l + 8:.1f}" y="{ry - 8:.1f}">{_e(ref["name"])}</text>'
        )
        taken.append((l + 4, ry - 19, l + 4 + rw, ry - 3))

    by_id = {q["id"]: q for q in pts}
    chain = sorted((by_id[i] for i in d["frontier"] if i in by_id), key=lambda s: s[xk])
    steps, segs = [], []
    for i, s in enumerate(chain):
        x, y = X(s[xk]), Y(s["score"])
        if i:
            py = Y(chain[i - 1]["score"])
            px = X(chain[i - 1][xk])
            segs.append((px, py, x, py))
            segs.append((x, py, x, y))
            steps.append(f"L{x:.1f},{py:.1f}")
        steps.append(("M" if i == 0 else "L") + f"{x:.1f},{y:.1f}")
    p.append(f'<path class="blk-front" d="{" ".join(steps)}"/>')

    ranked = sorted(pts, key=lambda s: -s["score"])
    named = {s["id"] for s in ranked if s["kind"] != "open"} | frontier_ids
    named |= {s["id"] for s in ranked[:3]}

    place = {s["id"]: (X(s[xk]), Y(s["score"])) for s in pts}
    radius = {
        s["id"]: (7 if s["kind"] == "ours" else (5.5 if s["id"] in named else 3.6)) for s in pts
    }
    dot_rects = {
        i: (place[i][0] - radius[i] - DOT_PAD, place[i][1] - radius[i] - DOT_PAD,
            place[i][0] + radius[i] + DOT_PAD, place[i][1] + radius[i] + DOT_PAD)
        for i in place
    }

    # ours first, then the reference, then the frontier: the last ones drop out if space runs out
    def priority(s: dict) -> tuple:
        rank = 0 if s["kind"] == "ours" else 1 if s["kind"] == "reference" else (
            2 if s["id"] in frontier_ids else 3
        )
        return (rank, -s["score"])

    frame = (2.0, float(t + 4), float(w - 4), float(h - b - 2))
    for s in sorted((q for q in pts if q["id"] in named), key=priority):
        x, y = place[s["id"]]
        name = _clip(s["name"], 20)
        xtxt = _fmt_x(xk, s[xk])
        width = label_width(name, xtxt)
        box, ly = _place(x, y, width, frame, taken, dot_rects, s["id"], segs)
        if box is None:
            continue  # better no label than one pointing at the wrong dot
        taken.append(box)
        hook = box[0] if box[0] > x else (box[2] if box[2] < x else (box[0] + box[2]) / 2)
        leads.append(
            f'<path class="blk-lead" data-for="{_e(s["id"])}" '
            f'd="M{hook:.1f},{ly - 4:.1f} L{x:.1f},{y:.1f}"/>'
        )
        labels.append(
            f'<text class="blk-pt-lab{" on" if s["kind"] == "ours" else ""}" '
            f'data-for="{_e(s["id"])}" x="{box[0]:.1f}" y="{ly:.1f}">'
            f'{_e(name)}<tspan class="blk-ax" dx="8">{_e(xtxt)}</tspan></text>'
        )

    for i, s in enumerate(ranked):
        x, y = place[s["id"]]
        ours = s["kind"] == "ours"
        col = _tone(s["kind"])
        on_front = s["id"] in frontier_ids
        xtxt = _fmt_x(xk, s[xk])
        spread = f" \u00b7 p95 {_fmt_x(xk, s['p95_ms'])}" if s.get("p95_ms") else ""
        tip = f'<title>{_e(s["name"])} \u00b7 {_e(xtxt + spread)} \u00b7 {_f(s["score"])}</title>'
        if on_front:
            dots.append(
                f'<circle class="blk-halo" style="animation-delay:{i * 70}ms" cx="{x:.1f}" '
                f'cy="{y:.1f}" r="13"/>'
            )
        dots.append(
            f'<circle class="blk-pt" data-id="{_e(s["id"])}" style="animation-delay:{i * 70}ms" '
            f'cx="{x:.1f}" cy="{y:.1f}" r="{radius[s["id"]]}" fill="{col if on_front else "#fff"}" '
            f'stroke="{col}" stroke-width="{2.2 if ours else 1.6}">{tip}</circle>'
        )

    label = f'{d["x_label"]} against {d["y_label"]}'
    note = (
        f'<details class="blk-d"><summary>How this was measured</summary><p>{_e(d["note"])}</p></details>'
        if d.get("note") else ""
    )
    body = "".join(p) + "".join(leads) + "".join(dots) + "".join(labels)
    return (
        f'<figure class="blk-figure"><figcaption>{_e(PARETO_CAPTIONS.get(key, ""))}</figcaption>'
        f'{_svg(body, w, h, label)}</figure>{note}'
    )


# --- picking the next click -----------------------------------------------------

CUA_CAPTION = "Accuracy with a 95% interval \u00b7 text and screenshot"

CUA_MARKS = {"text": "text", "screenshot": "screenshot"}


def _cua_groups(rows: list[dict]) -> list[tuple[str, list[dict]]]:
    """One row per model; a variant that only adds an input keeps its model's row."""
    out: list[tuple[str, list[dict]]] = []
    for row in rows:
        base = row["name"].split(" + ")[0]
        for name, items in out:
            if name == base:
                items.append(row)
                break
        else:
            out.append((base, [row]))
    ours = [g for g in out if any(r["kind"] == "ours" for r in g[1])]
    return ours + [g for g in out if g not in ours]


def cua_probe_chart(data: dict) -> str:
    """Null until the probe is run, and then one row per model with both inputs on it."""
    d = data.get("cua_probe")
    if not d:
        return ""
    rows = d["rows"]
    groups = _cua_groups(rows)
    chance = d.get("chance")
    w, gutter = 1060, 268
    bw = 640
    top, rowh, sub = 46, 58, 20
    h = top + rowh * len(groups) + 22
    lo_vals = [r["ci95"][0] for r in rows] + ([chance] if chance else [])
    hi_vals = [r["ci95"][1] for r in rows]
    lo = math.floor((min(lo_vals) - 8) / 10) * 10
    hi = math.ceil((max(hi_vals) + 3) / 5) * 5
    p = []

    def sx(v: float) -> float:
        return gutter + bw * (v - lo) / (hi - lo)

    for tick in range(int(math.ceil(lo / 10.0) * 10), int(hi) + 1, 10):
        x = sx(tick)
        p.append(f'<line class="blk-grid" x1="{x:.1f}" y1="{top - 14}" x2="{x:.1f}" y2="{h - 14}"/>')
        p.append(f'<text class="blk-ax" x="{x:.1f}" y="{top - 20}" text-anchor="middle">{tick}%</text>')
    if chance:
        cx = sx(chance)
        p.append(f'<line class="blk-ref" x1="{cx:.1f}" y1="{top - 14}" x2="{cx:.1f}" y2="{h - 14}"/>')
        p.append(f'<text class="blk-ax" x="{cx + 7:.1f}" y="{h - 2}">chance</text>')

    lx = 6
    for i, (mark, name) in enumerate(CUA_MARKS.items()):
        mx = lx + i * 96
        filled = mark == "text"
        p.append(
            f'<circle cx="{mx:.1f}" cy="{top - 24}" r="5" fill="{SOFT if filled else "#fff"}" '
            f'stroke="{SOFT}" stroke-width="1.6"/>'
        )
        p.append(f'<text class="blk-ax" x="{mx + 10:.1f}" y="{top - 20}">{_e(name)}</text>')

    for gi, (name, items) in enumerate(groups):
        y = top + gi * rowh
        ours = any(r["kind"] == "ours" for r in items)
        mid = y + (sub * (len(items) - 1)) / 2 + 4
        p.append(
            f'<text class="blk-name{" on" if ours else ""}" x="{gutter - 16}" y="{mid:.1f}" '
            f'text-anchor="end">{_e(_clip(name, 28))}<title>{_e(name)}</title></text>'
        )
        for ri, row in enumerate(items):
            cy = y + ri * sub + 4
            col = OURS if row["kind"] == "ours" else OPEN
            lo_x, hi_x = sx(row["ci95"][0]), sx(row["ci95"][1])
            p.append(
                f'<rect class="blk-grow" style="animation-delay:{(gi * 2 + ri) * 70}ms" '
                f'x="{lo_x:.1f}" y="{cy - 1.5:.1f}" width="{hi_x - lo_x:.1f}" height="3" rx="1.5" '
                f'fill="{OURS_LIGHT if row["kind"] == "ours" else OPEN}"/>'
            )
            filled = row["input"] == "text"
            tip = (
                f'{row["name"]} \u00b7 {row["input"]} \u00b7 {row["accuracy"]:.1f}% '
                f'[{row["ci95"][0]:.1f}, {row["ci95"][1]:.1f}] \u00b7 n={row["n"]} '
                f'\u00b7 ECE {row["ece"]:.3f}'
            )
            p.append(
                f'<circle class="blk-pt" style="animation-delay:{(gi * 2 + ri) * 70}ms" '
                f'cx="{sx(row["accuracy"]):.1f}" cy="{cy:.1f}" r="{6 if row["kind"] == "ours" else 5}" '
                f'fill="{col if filled else "#fff"}" stroke="{col}" '
                f'stroke-width="{2.2 if row["kind"] == "ours" else 1.6}"><title>{_e(tip)}</title></circle>'
            )
            p.append(
                f'<text class="blk-val{" on" if row["kind"] == "ours" else ""}" '
                f'x="{hi_x + 10:.1f}" y="{cy + 4:.1f}">{row["accuracy"]:.1f}</text>'
            )

    svg = _svg("".join(p), w, h, f'{d["label"]} by model and input')
    note = (
        f'<details class="blk-d"><summary>About this test</summary><p>{_e(d["note"])}</p></details>'
        if d.get("note") else ""
    )
    return (
        f'<figure class="blk-figure"><figcaption>{_e(CUA_CAPTION)}</figcaption>{svg}</figure>{note}'
    )


# --- home headlines ----------------------------------------------------------------


def headline_rows(data: dict) -> list[dict]:
    """Ours and the one system they are measured against, best first."""
    systems = data["decision_index"]["systems"]
    ref = data["decision_index"].get("reference")
    keep = [s for s in systems if s["kind"] == "ours" or s["id"] == ref]
    return sorted(keep, key=lambda s: -s["index"])


def headline_chart(data: dict) -> str:
    """A short bar chart: just us and the reference, on the same scale as the full board."""
    shown = headline_rows(data)
    if not shown:
        return ""
    w, gutter, bw = 560, 116, 392
    top, rowh = 28, 32
    h = top + rowh * len(shown) + 12
    lo, hi = 0, int(math.ceil(max(s["index"] for s in shown) / 10.0) * 10)
    p = []

    def sx(v: float) -> float:
        return gutter + bw * (v - lo) / (hi - lo)

    for t in range(lo, hi + 1, 10):
        p.append(f'<line class="blk-grid" x1="{sx(t):.1f}" y1="{top - 12}" x2="{sx(t):.1f}" y2="{h - 14}"/>')
        p.append(f'<text class="blk-ax" x="{sx(t):.1f}" y="{top - 18}" text-anchor="middle">{t}</text>')
    for i, row in enumerate(shown):
        y = top + i * rowh
        ours = row["kind"] == "ours"
        p.append(
            f'<text class="blk-name{" on" if ours else ""}" x="{gutter - 16}" y="{y + 19}" '
            f'text-anchor="end">{_e(_clip(row["name"], 18))}</text>'
        )
        x1 = sx(row["index"])
        p.append(
            f'<rect class="blk-grow" style="animation-delay:{i * 80}ms" x="{gutter}" y="{y + 6}" '
            f'width="{x1 - gutter:.1f}" height="16" rx="4" fill="{_tone(row["kind"])}"/>'
        )
        p.append(
            f'<text class="blk-val{" on" if ours else ""}" x="{x1 + 9:.1f}" y="{y + 19}">'
            f'{_f(row["index"])}</text>'
        )
    return (
        f'<figure class="blk-figure">{_svg("".join(p), w, h, "Decision Index by system")}</figure>'
    )


def headline_jevbench(data: dict) -> str:
    """The public-item proxies, on their own, with the row that was never submitted."""
    d = data.get("jevbench") or {}
    rows = [s for s in d.get("systems", []) if s.get("public_estimate") is not None]
    if not rows:
        return ""
    rows.sort(key=lambda s: -s["public_estimate"])
    items = "".join(
        f'<li class="blk-row{" on" if s["kind"] == "ours" else ""}">'
        f'<b style="--c:{OURS if s["kind"] == "ours" else OPEN}">{_e(s["name"])}</b>'
        f'<strong>{_f(s["public_estimate"])}</strong></li>'
        for s in rows
    )
    return f'<ul class="blk-rows blk-headline-rows">{items}</ul>'


# --- held-out tasks -----------------------------------------------------------------

HELDOUT_TONES = [OURS, OURS_DEEP, OURS_LIGHT]


def heldout_systems(data: dict) -> list[dict]:
    """Best first, so the figure and the per-task view agree on the order."""
    d = data.get("heldout") or {}
    return sorted(d.get("systems", []), key=lambda s: -s["accuracy"])


def heldout_chart(data: dict) -> str:
    """One bar per model with its 95% interval: the numbers nothing was tuned on."""
    d = data.get("heldout") or {}
    shown = heldout_systems(data)
    if not shown:
        return ""
    w, gutter, bw = 560, 132, 350
    top, rowh = 28, 32
    h = top + rowh * len(shown) + 12

    def sx(v: float) -> float:
        return gutter + bw * v

    p = []
    for tick in (0, 25, 50, 75, 100):
        x = sx(tick / 100.0)
        p.append(f'<line class="blk-grid" x1="{x:.1f}" y1="{top - 12}" x2="{x:.1f}" y2="{h - 14}"/>')
        p.append(f'<text class="blk-ax" x="{x:.1f}" y="{top - 18}" text-anchor="middle">{tick}</text>')
    for i, s in enumerate(shown):
        y = top + i * rowh
        cy = y + 11
        lo_x, hi_x = sx(s["ci95"][0]), sx(s["ci95"][1])
        p.append(
            f'<text class="blk-name on" x="{gutter - 12}" y="{cy + 4:.1f}" '
            f'text-anchor="end">{_e(_clip(s["name"], 18))}</text>'
        )
        p.append(
            f'<rect class="blk-grow" style="animation-delay:{i * 90}ms" x="{gutter}" '
            f'y="{y + 3}" width="{sx(s["accuracy"]) - gutter:.1f}" height="16" rx="4" '
            f'fill="{HELDOUT_TONES[min(i, len(HELDOUT_TONES) - 1)]}"/>'
        )
        # drawn twice: a dark outline so the ends read on the page, a light core so
        # they read on the bar
        for cls in ("blk-whisk-edge", "blk-whisk"):
            p.append(
                f'<line class="{cls}" x1="{lo_x:.1f}" y1="{cy:.1f}" '
                f'x2="{hi_x:.1f}" y2="{cy:.1f}"/>'
            )
            for x in (lo_x, hi_x):
                p.append(
                    f'<line class="{cls}" x1="{x:.1f}" y1="{cy - 6:.1f}" '
                    f'x2="{x:.1f}" y2="{cy + 6:.1f}"/>'
                )
        tip = (
            f'{s["name"]} \u00b7 {s["accuracy"] * 100:.1f}% '
            f'[{s["ci95"][0] * 100:.1f}, {s["ci95"][1] * 100:.1f}] \u00b7 ECE {s["ece"]:.3f}'
        )
        p.append(
            f'<text class="blk-val on" x="{hi_x + 9:.1f}" y="{cy + 4:.1f}">'
            f'{s["accuracy"] * 100:.1f}<title>{_e(tip)}</title></text>'
        )
    return f'<figure class="blk-figure">{_svg("".join(p), w, h, _e(d["label"]))}</figure>'


def heldout_tasks(data: dict) -> str:
    """The same eight tasks one at a time, so an average cannot hide a weak one."""
    d = data.get("heldout") or {}
    shown = heldout_systems(data)
    tasks = d.get("tasks") or []
    if not shown or not tasks:
        return ""
    w, gutter, bw = 1060, 210, 690
    top, rowh, sub = 62, 56, 17
    h = top + rowh * len(tasks) + 16

    def sx(v: float) -> float:
        return gutter + bw * v

    p = []
    for tick in range(0, 101, 20):
        x = sx(tick / 100.0)
        p.append(f'<line class="blk-grid" x1="{x:.1f}" y1="{top - 16}" x2="{x:.1f}" y2="{h - 14}"/>')
        p.append(f'<text class="blk-ax" x="{x:.1f}" y="{top - 22}" text-anchor="middle">{tick}%</text>')
    for i, s in enumerate(shown):
        mx = 6 + i * 230
        tone = HELDOUT_TONES[min(i, len(HELDOUT_TONES) - 1)]
        p.append(f'<circle cx="{mx}" cy="{top - 42}" r="5" fill="{tone}"/>')
        p.append(
            f'<text class="blk-ax" x="{mx + 11}" y="{top - 38}">'
            f'{_e(s["name"])} \u00b7 ECE {s["ece"]:.3f}</text>'
        )
    for ti, task in enumerate(tasks):
        y = top + ti * rowh
        mid = y + (sub * (len(shown) - 1)) / 2 + 4
        p.append(
            f'<text class="blk-name on" x="{gutter - 16}" y="{mid:.1f}" '
            f'text-anchor="end">{_e(_clip(task["label"], 24))}<title>{_e(task["id"])}</title></text>'
        )
        for si, s in enumerate(shown):
            cell = (s.get("tasks") or {}).get(task["id"])
            if not cell:
                continue
            cy = y + si * sub + 4
            tone = HELDOUT_TONES[min(si, len(HELDOUT_TONES) - 1)]
            tip = (
                f'{s["name"]} \u00b7 {task["label"]} \u00b7 {cell["accuracy"] * 100:.1f}% '
                f'\u00b7 ECE {cell["ece"]:.3f}'
            )
            p.append(
                f'<rect class="blk-grow" style="animation-delay:{(ti * 3 + si) * 40}ms" '
                f'x="{gutter}" y="{cy - 5:.1f}" width="{sx(cell["accuracy"]) - gutter:.1f}" '
                f'height="10" rx="1.5" fill="{tone}"><title>{_e(tip)}</title></rect>'
            )
            p.append(
                f'<text class="blk-val{" on" if si == 0 else ""}" '
                f'x="{sx(cell["accuracy"]) + 9:.1f}" y="{cy + 4:.1f}">'
                f'{cell["accuracy"] * 100:.0f}</text>'
            )
    label = f'{d["label"]} by task'
    return f'<figure class="blk-figure"><figcaption>{_e(label)}</figcaption>{_svg("".join(p), w, h, label)}</figure>'


HELDOUT_TABLE_CAPTION = ("Average accuracy (each task counts equally), 95% intervals, and calibration error "
                         "from the same run.")
HELDOUT_COLS = ("Model", "Accuracy", "95% interval", "ECE")


def heldout_table(data: dict) -> str:
    """The same figures as text, so nothing needs a pointer to read."""
    shown = heldout_systems(data)
    if not shown:
        return ""
    head = "".join(f"<th>{_e(c)}</th>" for c in HELDOUT_COLS)
    body = "".join(
        f'<tr><th scope="row">{_e(s["name"])}</th>'
        f'<td>{s["accuracy"] * 100:.1f}%</td>'
        f'<td>{s["ci95"][0] * 100:.1f}\u2013{s["ci95"][1] * 100:.1f}</td>'
        f'<td>{s["ece"]:.3f}</td></tr>'
        for s in shown
    )
    return (
        f'<table class="blk-nums"><caption>{_e(HELDOUT_TABLE_CAPTION)}</caption>'
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
    )


def headline_table(data: dict) -> str:
    """The headline bars as text, for the widths where a wide figure cannot fit."""
    shown = headline_rows(data)
    if not shown:
        return ""
    label = data["decision_index"]["label"]
    body = "".join(
        f'<tr><th scope="row">{_e(s["name"])}</th><td>{_f(s["index"])}</td></tr>'
        for s in shown
    )
    return (
        f'<table class="blk-nums"><caption>{_e(label)}</caption>'
        f'<thead><tr><th>Model</th><th>Index</th></tr></thead>'
        f"<tbody>{body}</tbody></table>"
    )


def heldout_note(data: dict) -> str:
    """The caption plus what the file says the set is made of."""
    d = data.get("heldout") or {}
    if not d:
        return ""
    bits = [d["caption"]]
    frozen = str(d.get("frozen_utc") or "")[:10]
    if frozen:
        bits.append(f"Frozen {frozen}.")
    return " ".join(bits)


# --- latency ------------------------------------------------------------------------


def latency_strip(lat: dict) -> str:
    """Rendered only when replay.json has been recorded; results.json carries null until then."""
    return (
        '<figure class="blk-figure"><figcaption>Latency</figcaption>'
        '<div class="blk-stats">'
        f'<span><b>{lat["p50_ms"]:g}</b> ms median</span>'
        f'<span><b>{lat["max_ms"]:g}</b> ms slowest</span>'
        f'<span><b>{lat["requests"]}</b> requests</span>'
        f'<span><b>{lat["generated_tokens"]}</b> generated</span>'
        "</div>"
        f'<p class="blk-note">{_e(lat["caption"])}</p></figure>'
    )


# --- how-it-works figure ------------------------------------------------------------


def pipeline_figure() -> str:
    w, h = 1060, 246
    stages = [
        (4, "State", "A ticket, an email,\na document, a row of JSON"),
        (272, "Questions", "choice · noul · score\nwith your own criteria"),
        (540, "Read options", "one forward pass\nper batch"),
        (808, "Probabilities", "normalised over the\noptions you offered"),
    ]
    box, mid = 232, 118
    p = []
    for i, (x, title, sub) in enumerate(stages):
        p.append(
            f'<g class="blk-stage" style="animation-delay:{i * 180}ms">'
            f'<rect x="{x}" y="{mid - 52}" width="{box}" height="104" rx="10" class="blk-card"/>'
            f'<text class="blk-stage-n" x="{x + 18}" y="{mid - 66}">{i + 1}</text>'
            f'<text class="blk-stage-t" x="{x + 18}" y="{mid - 16}">{_e(title)}</text>'
        )
        for j, line in enumerate(sub.split("\n")):
            p.append(f'<text class="blk-stage-s" x="{x + 18}" y="{mid + 8 + j * 17}">{_e(line)}</text>')
        p.append("</g>")
        if i < len(stages) - 1:
            nx = stages[i + 1][0]
            p.append(
                f'<path class="blk-flow" style="animation-delay:{i * 180 + 120}ms" '
                f'd="M{x + box + 6},{mid} L{nx - 12},{mid}"/>'
                f'<path class="blk-flow-head" style="animation-delay:{i * 180 + 120}ms" '
                f'd="M{nx - 20},{mid - 6} l8,6 -8,6"/>'
            )
    p.append(
        '<text class="blk-stage-s" x="4" y="214">No text is generated. Every offered option gets a probability.</text>'
    )
    p.append('<text class="blk-ax" x="4" y="236">Large requests may need more than one batch.</text>')
    return f'<figure class="blk-figure blk-bare">{_svg("".join(p), w, h, "How a decision is made")}</figure>'
