"""Minimal deterministic SVG builder with real font measurement.

Text is measured against every font a viewer is likely to fall back to
(Inter, Liberation Sans = Arial metrics, DejaVu Sans) and the widest result is
used, so a box that fits here fits everywhere. Nothing is auto-truncated: text
that cannot fit is reported by ``Canvas.overflows`` and must be fixed in the
figure code.
"""
from __future__ import annotations

import functools
import html
from dataclasses import dataclass, field

# ---------------------------------------------------------------- palette ---
INK = "#1C1917"
INK_SOFT = "#44403C"
MUTED = "#78716C"
FAINT = "#A8A29E"
RULE = "#E7E5E4"
BORDER = "#D6D3D1"
PANEL = "#FFFFFF"
BG = "#FAFAF9"
SUBTLE = "#F5F5F4"

ACC = "#0F766E"          # the one accent: trained by us
ACC_DEEP = "#0B5F59"
ACC_LINE = "#7FBDB6"
ACC_FILL = "#E9F4F2"     # Gated DeltaNet layers
ACC_FILL2 = "#C3E3DE"    # full-attention layers

# The figure sits on an opaque light card inside an opaque mat, so it reads as a
# deliberate panel on a dark page as well as a light one. Never transparent.
MAT = "#ECE9E5"
CARD_PAD = 14
CARD_RX = 16
CARD_LINE = "#D9D5D1"

FROZEN_FILL = "#F0EFED"
FROZEN_LINE = "#CFCBC7"

SANS = ("Inter, 'Inter Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', "
        "Roboto, 'Helvetica Neue', Arial, 'Liberation Sans', sans-serif")
# No font-variant-numeric here. Inter's `tnum` feature swaps the hyphen-minus for a
# digit-width tabular form, which renders "blink-4b" as "blink - 4b" and also makes
# rendered widths disagree with the advances measured below.
TEXT_CSS = f"font-family:{SANS};white-space:pre"
MONO = ("ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, "
        "'Liberation Mono', 'DejaVu Sans Mono', monospace")

_SANS_FILES = {
    400: ["/usr/share/fonts/opentype/inter/Inter-Regular.otf",
          "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
          "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
    500: ["/usr/share/fonts/opentype/inter/Inter-Medium.otf",
          "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
          "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
    600: ["/usr/share/fonts/opentype/inter/Inter-SemiBold.otf",
          "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
          "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
    700: ["/usr/share/fonts/opentype/inter/Inter-Bold.otf",
          "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
          "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
}
_MONO_FILES = {
    400: ["/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
          "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"],
    500: ["/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
          "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"],
    600: ["/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
          "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"],
}


@functools.lru_cache(maxsize=None)
def _metrics(path):
    from fontTools.ttLib import TTFont
    f = TTFont(path, lazy=True, fontNumber=0)
    cmap = f.getBestCmap()
    hmtx = f["hmtx"]
    upem = f["head"].unitsPerEm
    adv = {}
    for cp, gname in cmap.items():
        try:
            adv[cp] = hmtx[gname][0] / upem
        except KeyError:
            pass
    fallback = adv.get(ord("n"), 0.55)
    f.close()
    return adv, fallback


def text_width(s: str, size: float, weight: int = 400, mono: bool = False) -> float:
    """Widest advance width of ``s`` across the candidate fallback fonts."""
    files = (_MONO_FILES if mono else _SANS_FILES)[weight]
    best = 0.0
    for path in files:
        try:
            adv, fb = _metrics(path)
        except Exception:
            continue
        w = sum(adv.get(ord(c), fb) for c in s)
        best = max(best, w * size)
    return best * 1.015  # hinting / letter-spacing safety


# ------------------------------------------------------------ text styles ---
@dataclass(frozen=True)
class Style:
    size: float
    weight: int = 400
    fill: str = INK
    mono: bool = False
    tracking: float = 0.0
    upper: bool = False

    def width(self, s: str) -> float:
        t = s.upper() if self.upper else s
        return text_width(t, self.size, self.weight, self.mono) + self.tracking * max(len(t) - 1, 0)


STYLES = {
    "h1":    Style(23, 600, INK),
    "sub":   Style(13.5, 400, MUTED),
    "sec":   Style(11, 600, MUTED, tracking=0.95, upper=True),
    "sec_l": Style(11, 600, "#CFCAC4", tracking=0.95, upper=True),
    "lbl":   Style(13, 500, INK),
    "lblb":  Style(13, 600, INK),
    "body":  Style(12.5, 400, INK_SOFT),
    "note":  Style(11.5, 400, MUTED),
    "tiny":  Style(10.5, 400, FAINT),
    "num":   Style(13, 600, INK),
    "numa":  Style(13, 600, ACC),
    "code":  Style(11.5, 400, INK_SOFT, mono=True),
    "codea": Style(11.5, 500, ACC_DEEP, mono=True),
    "wlbl":  Style(13, 500, "#FFFFFF"),
    "wnote": Style(11.5, 400, "#C8C4BF"),
    "wsec":  Style(11, 600, "#A8A29E", tracking=0.95, upper=True),
    "wnum":  Style(13, 600, "#FFFFFF"),
}


@dataclass
class Canvas:
    w: float
    h: float
    parts: list = field(default_factory=list)
    overflows: list = field(default_factory=list)

    # ------------------------------------------------------------- shapes --
    def raw(self, s: str):
        self.parts.append(s)

    def defer(self) -> int:
        """Reserve a slot so a panel background can be drawn under later content."""
        self.parts.append("")
        return len(self.parts) - 1

    def fill_deferred(self, idx: int, x, y, w, h, fill=PANEL, stroke=BORDER, rx=9, sw=1):
        st = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ' stroke="none"'
        self.parts[idx] = (f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
                           f'rx="{rx}" fill="{fill}"{st}/>')

    def rect(self, x, y, w, h, fill=PANEL, stroke=BORDER, rx=7, sw=1, dash=None, extra=""):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        st = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ' stroke="none"'
        self.raw(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
                 f'rx="{rx}" fill="{fill}"{st}{d}{extra}/>')

    def line(self, x1, y1, x2, y2, stroke=RULE, sw=1, dash=None, cap="round"):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.raw(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
                 f'stroke="{stroke}" stroke-width="{sw}" stroke-linecap="{cap}"{d}/>')

    def path(self, d, stroke=BORDER, fill="none", sw=1, marker=None, dash=None):
        m = f' marker-end="url(#{marker})"' if marker else ""
        da = f' stroke-dasharray="{dash}"' if dash else ""
        self.raw(f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" '
                 f'stroke-linecap="round" stroke-linejoin="round"{m}{da}/>')

    def arrow(self, x1, y1, x2, y2, stroke=FAINT, sw=1.4, marker="arw"):
        self.path(f"M {x1:.2f} {y1:.2f} L {x2:.2f} {y2:.2f}", stroke=stroke, sw=sw, marker=marker)

    # --------------------------------------------------------------- text --
    def text(self, x, y, s, style="body", anchor="start", maxw=None, fill=None, tag=""):
        st = STYLES[style]
        shown = s.upper() if st.upper else s
        wid = st.width(s)
        if maxw is not None and wid > maxw:
            self.overflows.append((tag or shown[:48], round(wid, 1), round(maxw, 1)))
        col = fill or st.fill
        extra = ""
        if st.mono:
            extra += f' font-family="{MONO}"'
        if st.tracking:
            extra += f' letter-spacing="{st.tracking}"'
        self.raw(f'<text x="{x:.2f}" y="{y:.2f}" font-size="{st.size}" font-weight="{st.weight}" '
                 f'fill="{col}" text-anchor="{anchor}"{extra}>{html.escape(shown)}</text>')
        return wid

    def wrap(self, x, y, s, style="body", maxw=200, lh=16, anchor="start", max_lines=99):
        st = STYLES[style]
        words, lines, cur = s.split(" "), [], ""
        for wd in words:
            trial = f"{cur} {wd}".strip()
            if st.width(trial) <= maxw or not cur:
                cur = trial
            else:
                lines.append(cur)
                cur = wd
        if cur:
            lines.append(cur)
        if len(lines) > max_lines:
            self.overflows.append((s[:48], len(lines), max_lines))
        for i, ln in enumerate(lines):
            self.text(x, y + i * lh, ln, style, anchor, maxw=maxw, tag=ln)
        return y + (len(lines) - 1) * lh, len(lines)

    # ------------------------------------------------------------ compose --
    @property
    def out_w(self) -> float:
        return self.w + 2 * CARD_PAD

    @property
    def out_h(self) -> float:
        return self.h + 2 * CARD_PAD

    def render(self) -> str:
        defs = f"""<defs>
  <marker id="arw" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5"
          orient="auto-start-reverse" markerUnits="strokeWidth">
    <path d="M 0 1.4 L 9 5 L 0 8.6 z" fill="{FAINT}"/>
  </marker>
  <marker id="arwa" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5"
          orient="auto-start-reverse" markerUnits="strokeWidth">
    <path d="M 0 1.4 L 9 5 L 0 8.6 z" fill="{ACC}"/>
  </marker>
  <marker id="arww" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5"
          orient="auto-start-reverse" markerUnits="strokeWidth">
    <path d="M 0 1.4 L 9 5 L 0 8.6 z" fill="#8A847D"/>
  </marker>
  <pattern id="hatch" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
    <rect width="7" height="7" fill="{SUBTLE}"/>
    <line x1="0" y1="0" x2="0" y2="7" stroke="{BORDER}" stroke-width="1.6"/>
  </pattern>
  <clipPath id="card">
    <rect x="{CARD_PAD}" y="{CARD_PAD}" width="{self.w:.0f}" height="{self.h:.0f}" rx="{CARD_RX}"/>
  </clipPath>
</defs>"""
        style = f'<style>text{{{TEXT_CSS};dominant-baseline:auto}}</style>' 
        body = "\n".join(self.parts)
        ow, oh = self.out_w, self.out_h
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{ow:.0f}" height="{oh:.0f}" '
                f'viewBox="0 0 {ow:.0f} {oh:.0f}" role="img">\n'
                f'{style}\n{defs}\n'
                f'<rect width="{ow:.0f}" height="{oh:.0f}" fill="{MAT}"/>\n'
                f'<rect x="{CARD_PAD}" y="{CARD_PAD}" width="{self.w:.0f}" height="{self.h:.0f}" '
                f'rx="{CARD_RX}" fill="{BG}"/>\n'
                f'<g clip-path="url(#card)" transform="translate({CARD_PAD},{CARD_PAD})">\n'
                f'{body}\n</g>\n'
                f'<rect x="{CARD_PAD + 0.5}" y="{CARD_PAD + 0.5}" width="{self.w - 1:.0f}" '
                f'height="{self.h - 1:.0f}" rx="{CARD_RX - 0.5}" fill="none" '
                f'stroke="{CARD_LINE}" stroke-width="1"/>\n</svg>\n')

