#!/usr/bin/env python3
"""Build blink's documentation site from the repository's own Markdown.

    uv run --no-project --with-requirements site/requirements.txt python site/build.py --out _site
    python3 site/check.py _site
    python3 -m http.server --bind 127.0.0.1 --directory _site 8000

The pages are the README, docs/, the model cards, CHANGELOG, CONTRIBUTING, SECURITY and
examples/. Every string the site adds is in site/strings.json; text it reuses verbatim is in
site/strings-reused.json; links and pinned assets are in site/config.json. A Mermaid block
uses the SVG that site/render_diagrams.py saved in site/diagrams/ for it, and is drawn in the
browser when there is none. The home page reuses the Space's own renderers (space/results.py
and space/ui.py) and its saved blink-4b run.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import posixpath
import re
import shutil
import struct
import sys
import unicodedata
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE.parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}


@dataclass(frozen=True)
class Page:
    slug: str
    source: str | None
    group: str | None
    label: str | None = None
    # (link prefix, repository directory it stands for): the cards were written for their model repos
    aliases: tuple[tuple[str, str], ...] = ()


MODEL_PAGES = tuple(
    Page(f"models/{m}", f"docs/models/{m}.md", "models",
         aliases=(("assets/", f"docs/assets/{m}/"), ("", f"docs/models/{m}/")))
    for m in CONFIG["models"]
)
PAGES = (
    Page("overview", "README.md", "start", label="nav.overview"),
    Page("wire-format", "docs/WIRE_FORMAT.md", "start"),
    Page("examples", None, "start", label="nav.examples"),
    *MODEL_PAGES,
    Page("results", "docs/RESULTS.md", "evaluation"),
    Page("evaluation", "docs/EVALUATION.md", "evaluation"),
    Page("history", "docs/HISTORY.md", "evaluation"),
    Page("first-principles", "docs/blink-first-principles.md", "learn", label="nav.first_principles"),
    Page("changelog", "CHANGELOG.md", "project"),
    Page("contributing", "CONTRIBUTING.md", "project"),
    Page("security", "SECURITY.md", "project"),
)
HOME = Page("", None, None)
HOME_README = Page("", "README.md", None)  # README text shown on the home page
GROUPS = ("start", "models", "evaluation", "learn", "project")
HOME_DOC_GROUPS = ("start", "evaluation", "learn")


class BuildError(RuntimeError):
    """The sources cannot produce a site worth publishing."""


# --- strings and small helpers -----------------------------------------------------------


def load_strings() -> dict[str, str]:
    """strings.json: every string the site adds. strings-reused.json: text copied verbatim from its source."""
    own = json.loads((HERE / "strings.json").read_text(encoding="utf-8"))
    reused = json.loads((HERE / "strings-reused.json").read_text(encoding="utf-8"))["strings"]
    out = dict(own)
    for key, item in reused.items():
        if key in out:
            raise BuildError(f"{key} is in both strings.json and strings-reused.json")
        out[key] = item["text"]
    return out


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def github_slug(text: str) -> str:
    """The anchor GitHub gives a heading: lower case, punctuation and symbols dropped, spaces to hyphens."""
    kept = [ch for ch in text.lower()
            if ch in " -_" or ch.isalnum() or unicodedata.category(ch).startswith("M")]
    return "".join(kept).replace(" ", "-")


class Slugger:
    """GitHub's duplicate rule: the second "Usage" is "usage-1"."""

    def __init__(self) -> None:
        self.seen: dict[str, int] = {}

    def slug(self, text: str) -> str:
        base = slug = github_slug(text)
        while slug in self.seen:
            self.seen[base] += 1
            slug = f"{base}-{self.seen[base]}"
        self.seen[slug] = 0
        return slug


def rel_url(from_slug: str, target: str) -> str:
    """A link from the page at <from_slug>/index.html to a path relative to the site root."""
    depth = len([p for p in from_slug.split("/") if p])
    return ("../" * depth + target) or "./"


def image_size(path: Path) -> tuple[int, int] | None:
    data = path.read_bytes()[:4096]
    if data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR":
        return struct.unpack(">II", data[16:24])
    if path.suffix.lower() == ".svg":
        text = data.decode("utf-8", "replace")
        m = re.search(r"<svg\b[^>]*>", text)
        if m:
            w = re.search(r'\bwidth="([\d.]+)"', m.group(0))
            h = re.search(r'\bheight="([\d.]+)"', m.group(0))
            if w and h:
                return round(float(w.group(1))), round(float(h.group(1)))
    return None


def front_matter_value(text: str, key: str) -> str | None:
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if not m:
        return None
    hit = re.search(rf"^{re.escape(key)}:\s*(.+?)\s*$", m.group(1), re.M)
    return hit.group(1).strip("'\"") if hit else None


# --- diagrams ----------------------------------------------------------------------------


def diagram_env_hash(site_dir: Path = HERE) -> str:
    """What a saved SVG depends on besides its source: Mermaid's version and settings, and site.css."""
    mer = CONFIG["mermaid"]
    css = hashlib.sha256((site_dir / "static" / "site.css").read_bytes()).hexdigest()
    blob = json.dumps({"version": mer["version"], "config": mer["config"], "css": css}, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def diagram_key(code: str, env_hash: str) -> str:
    return hashlib.sha256(f"{env_hash}\n{code}".encode("utf-8")).hexdigest()[:16]


class Diagrams:
    def __init__(self, site_dir: Path) -> None:
        self.dir = site_dir / "diagrams"
        self.env_hash = diagram_env_hash(site_dir)
        self.used: list[str] = []
        self.missing: list[str] = []

    def html(self, code: str, env: dict) -> str:
        key = diagram_key(code, self.env_hash)
        path = self.dir / f"{key}.svg"
        self.used.append(key)
        if path.exists():
            return f'<figure class="diagram">{path.read_text(encoding="utf-8").strip()}</figure>\n'
        self.missing.append(key)
        env["mermaid"] = True
        return f'<figure class="diagram"><pre class="mermaid">{esc(code)}</pre></figure>\n'


# --- markdown ----------------------------------------------------------------------------


def markdown():
    from markdown_it import MarkdownIt
    from mdit_py_plugins.dollarmath import dollarmath_plugin
    from mdit_py_plugins.front_matter import front_matter_plugin

    md = MarkdownIt("commonmark", {"html": True, "typographer": False})
    md.enable(["table", "strikethrough"])
    md.use(front_matter_plugin)
    # GitHub's rule: "$5 and $10" is money, not math
    md.use(dollarmath_plugin, allow_labels=False, allow_space=False, allow_digits=False, double_inline=True)
    return md


def inline_text(tok) -> str:
    parts = []
    for c in tok.children or []:
        if c.type in ("text", "code_inline", "math_inline", "math_inline_double"):
            parts.append(c.content)
        elif c.type in ("softbreak", "hardbreak"):
            parts.append(" ")
        elif c.type == "image":
            parts.append(c.content)
    return "".join(parts)


class Site:
    def __init__(self, root: Path, out: Path, site_dir: Path = HERE) -> None:
        self.root = root.resolve()
        self.out = out.resolve()
        self.site_dir = site_dir
        self.s = load_strings()
        self.md = markdown()
        self.diagrams = Diagrams(site_dir)
        self.by_source = {p.source: p for p in PAGES if p.source}
        self.moved = self._moved()
        self.examples = self._example_files()
        self.unlinked: list[tuple[str, str]] = []
        self.assets: set[str] = set()
        self.repo_url = f"https://github.com/{CONFIG['repo']}"
        self._rules()

    # repository paths that the exporter renamed, e.g. release/out/blink-4b/README.md
    def _moved(self) -> dict[str, str]:
        path = self.root / "EXPORT_MANIFEST.json"
        if not path.exists():
            return {}
        items = json.loads(path.read_text(encoding="utf-8"))
        return {i["src"]: i["dest"] for i in items if i.get("src") and not i["src"].startswith("<")}

    def _example_files(self) -> list[str]:
        folder = self.root / "examples"
        files = sorted(p.name for p in folder.iterdir() if p.is_file()) if folder.is_dir() else []
        order = {"choice": 0, "noul": 1, "score": 2, "mixed": 3}
        js = sorted((f for f in files if f.endswith(".json")),
                    key=lambda f: (order.get(f.rsplit(".", 1)[0].rsplit("-", 1)[-1], 9), f))
        return js + [f for f in files if not f.endswith(".json")]

    # --- links ---------------------------------------------------------------------------

    def github(self, path: str, directory: bool = False) -> str:
        kind = "tree" if directory else "blob"
        return f"{self.repo_url}/{kind}/{CONFIG['branch']}/{urllib.parse.quote(path)}"

    def locate(self, page: Page, raw: str) -> tuple[str, str] | None:
        """What a relative link names: ("page", slug), ("example", anchor), ("file", path) or ("dir", path)."""
        base = posixpath.dirname(page.source or "")
        cands = [raw.lstrip("/") if raw.startswith("/") else posixpath.normpath(posixpath.join(base, raw))]
        for prefix, repl in page.aliases:
            if raw.startswith(prefix):
                cands.append(posixpath.normpath(repl + raw[len(prefix):]))
        cands += [self.moved[c] for c in list(cands) if c in self.moved]
        for cand in cands:
            cand = cand.rstrip("/")
            if cand in ("", "..", ".") or cand.startswith("../"):
                continue
            if cand in self.by_source:
                return "page", self.by_source[cand].slug
            if cand == "examples":
                return "example", ""
            if cand.startswith("examples/") and cand[len("examples/"):] in self.examples:
                return "example", github_slug(cand[len("examples/"):])
            full = self.root / cand
            if full.is_file():
                return "file", cand
            if full.is_dir():
                return "dir", cand
        return None

    def resolve(self, page: Page, href: str, image: bool = False) -> str | None:
        """A Markdown link as the site should write it, or None when it points at nothing public."""
        if not href or href.startswith("#"):
            return href
        site = CONFIG["site_url"]
        if href.startswith(site):
            path, _, frag = href[len(site):].partition("#")
            return rel_url(page.slug, path) + (f"#{frag}" if frag else "")
        if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", href) or href.startswith("//"):
            return href
        raw, _, frag = href.partition("#")
        anchor = f"#{frag}" if frag else ""
        found = self.locate(page, urllib.parse.unquote(raw.split("?", 1)[0]))
        if found is None:
            return None
        kind, value = found
        if kind == "page":
            return rel_url(page.slug, value + "/") + anchor
        if kind == "example":
            return rel_url(page.slug, "examples/") + (f"#{value}" if value else anchor)
        if kind == "file" and image and Path(value).suffix.lower() in IMAGE_SUFFIXES:
            self.assets.add(value)
            return rel_url(page.slug, value)
        return self.github(value, directory=kind == "dir") + (anchor if kind == "file" else "")

    # --- rendering -----------------------------------------------------------------------

    def _rules(self) -> None:
        md = self.md

        def fence(renderer, tokens, idx, options, env):
            tok = tokens[idx]
            lang = tok.info.strip().split()[0] if tok.info.strip() else ""
            if lang == "mermaid":
                return self.diagrams.html(tok.content, env)
            return code_html(tok.content, lang)

        def code_block(renderer, tokens, idx, options, env):
            return code_html(tokens[idx].content, "")

        def heading_close(renderer, tokens, idx, options, env):
            tag = tokens[idx].tag
            hid = tokens[idx].meta.get("id")
            mark = (f'<a class="anchor" href="#{esc(hid)}" aria-hidden="true" tabindex="-1">#</a>'
                    if hid and tag != "h1" else "")
            return f"{mark}</{tag}>\n"

        def table_open(renderer, tokens, idx, options, env):
            return '<div class="table-wrap"><table>\n'

        def table_close(renderer, tokens, idx, options, env):
            return "</table></div>\n"

        def link_open(renderer, tokens, idx, options, env):
            if tokens[idx].meta.get("unlink"):
                return ""
            return renderer.renderToken(tokens, idx, options, env)

        def link_close(renderer, tokens, idx, options, env):
            return "" if tokens[idx].meta.get("unlink") else "</a>"

        def math_inline(renderer, tokens, idx, options, env):
            return f'<span class="math inline">{esc(tokens[idx].content)}</span>'

        def math_inline_double(renderer, tokens, idx, options, env):
            return f'<span class="math display">{esc(tokens[idx].content)}</span>'

        def math_block(renderer, tokens, idx, options, env):
            return f'<div class="math display">{esc(tokens[idx].content.strip())}</div>\n'

        md.add_render_rule("fence", fence)
        md.add_render_rule("code_block", code_block)
        md.add_render_rule("heading_close", heading_close)
        md.add_render_rule("table_open", table_open)
        md.add_render_rule("table_close", table_close)
        md.add_render_rule("link_open", link_open)
        md.add_render_rule("link_close", link_close)
        md.add_render_rule("math_inline", math_inline)
        md.add_render_rule("math_inline_double", math_inline_double)
        md.add_render_rule("math_block", math_block)

    def prepare(self, tokens, page: Page, env: dict) -> None:
        for i, tok in enumerate(tokens):
            if tok.type == "heading_open":
                text = inline_text(tokens[i + 1]).strip()
                hid = env["slugger"].slug(text)
                tok.attrSet("id", hid)
                tokens[i + 2].meta["id"] = hid
                level = int(tok.tag[1])
                if level == 1 and not env.get("title"):
                    env["title"] = text
                elif level in (2, 3):
                    env["toc"].append((level, text, hid))
            if tok.type == "math_block":
                env["math"] = True
            if tok.type != "inline" or not tok.children:
                continue
            stack = []
            for c in tok.children:
                if c.type.startswith("math_inline"):
                    env["math"] = True
                elif c.type == "link_open":
                    href = c.attrGet("href") or ""
                    new = self.resolve(page, href)
                    stack.append(c)
                    if new is None:
                        c.meta["unlink"] = True
                        self.unlinked.append((page.slug, href))
                    else:
                        c.attrSet("href", new)
                elif c.type == "link_close" and stack:
                    if stack.pop().meta.get("unlink"):
                        c.meta["unlink"] = True
                elif c.type == "image":
                    src = c.attrGet("src") or ""
                    new = self.resolve(page, src, image=True)
                    if new is None:
                        raise BuildError(f"{page.source}: image {src!r} is not in the repository")
                    c.attrSet("src", new)
                    c.attrSet("loading", "lazy")
                    c.attrSet("decoding", "async")
                    found = None if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", src) else self.locate(page, src.split("#")[0])
                    size = image_size(self.root / found[1]) if found and found[0] == "file" else None
                    if size:
                        c.attrSet("width", str(size[0]))
                        c.attrSet("height", str(size[1]))

    def render(self, text: str, page: Page, env: dict | None = None) -> tuple[str, dict]:
        env = env or new_env()
        tokens = self.md.parse(text, env)
        self.prepare(tokens, page, env)
        return self.md.renderer.render(tokens, self.md.options, env), env

    def copy_assets(self) -> None:
        for rel in sorted(self.assets):
            dest = self.out / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.root / rel, dest)

    # --- chrome --------------------------------------------------------------------------

    def label(self, page: Page, title: str | None = None) -> str:
        if page.label:
            return self.s[page.label]
        return title or page.slug

    def nav(self, current: Page, titles: dict[str, str]) -> str:
        out = []
        for group in GROUPS:
            items = []
            for p in (p for p in PAGES if p.group == group):
                on = ' aria-current="page"' if p.slug == current.slug else ""
                items.append(f'<li><a href="{rel_url(current.slug, p.slug + "/")}"{on}>'
                             f"{esc(self.label(p, titles.get(p.slug)))}</a></li>")
            out.append(f'<p class="nav-group">{esc(self.s["nav.group." + group])}</p>'
                       f'<ul>{"".join(items)}</ul>')
        return "".join(out)

    def toc(self, entries) -> str:
        if not entries:
            return ""
        items = "".join(f'<li class="l{lvl}"><a href="#{esc(hid)}">{esc(text)}</a></li>'
                        for lvl, text, hid in entries)
        return f"<ul>{items}</ul>"

    def head_extra(self, env: dict) -> str:
        bits = []
        if env.get("math"):
            k = CONFIG["katex"]
            bits.append(f'<link rel="stylesheet" href="{esc(k["css"]["url"])}" '
                        f'integrity="{esc(k["css"]["integrity"])}" crossorigin="anonymous">')
            bits.append(f'<script defer src="{esc(k["js"]["url"])}" '
                        f'integrity="{esc(k["js"]["integrity"])}" crossorigin="anonymous"></script>')
        if env.get("mermaid"):
            m = CONFIG["mermaid"]
            bits.append(f'<script defer src="{esc(m["js"]["url"])}" '
                        f'integrity="{esc(m["js"]["integrity"])}" crossorigin="anonymous"></script>')
            bits.append('<script type="application/json" id="mermaid-config">'
                        + json.dumps(m["config"]).replace("</", "<\\/") + "</script>")
        return "\n".join(bits)

    def shell(self, page: Page, title: str, description: str, body: str, env: dict,
              body_class: str, root_prefix: str | None = None, index: bool = True) -> str:
        root = root_prefix if root_prefix is not None else rel_url(page.slug, "")
        if root == "./":
            root = ""
        site = CONFIG["site_url"]
        if index:
            canonical = esc(site + (page.slug + "/" if page.slug else ""))
            meta = (f'<link rel="canonical" href="{canonical}">\n'
                    f'<meta property="og:type" content="website">\n'
                    f'<meta property="og:title" content="{esc(title)}">\n'
                    f'<meta property="og:description" content="{esc(description)}">\n'
                    f'<meta property="og:url" content="{canonical}">\n'
                    f'<meta property="og:image" content="{esc(site + "docs/assets/blink-4b/blink-4b-readout.png")}">\n'
                    '<meta name="twitter:card" content="summary_large_image">')
        else:
            meta = '<meta name="robots" content="noindex">'
        values = {
            "title": esc(title),
            "description": esc(description),
            "meta": meta,
            "root": root,
            "home": root or "./",
            "fonts_css": esc(CONFIG["fonts_css"]),
            "head_extra": self.head_extra(env),
            "body_class": body_class,
            "copy": esc(self.s["code.copy"]),
            "copied": esc(self.s["code.copied"]),
            "skip": esc(self.s["site.skip"]),
            "docs": esc(self.s["site.docs"]),
            "space": esc(self.s["home.cta_space"]),
            "github": esc(self.s["site.github"]),
            "space_url": esc(CONFIG["space_url"]),
            "github_url": esc(self.repo_url),
            "hub_url": esc(CONFIG["hub_url"] + CONFIG["repo"].split("/")[0]),
            "hub": esc(self.s["home.model_hub"]),
            "footer_note": esc(self.s["footer.note"]),
            "body": body,
        }
        return fill(template("base.html"), values)

    # --- pages ---------------------------------------------------------------------------

    def page_text(self, page: Page) -> str:
        if page.source:
            return (self.root / page.source).read_text(encoding="utf-8")
        return self.examples_markdown()

    def examples_markdown(self) -> str:
        fence_lang = {".json": "json", ".sh": "bash", ".py": "python"}
        parts = [f"# {self.s['nav.examples']}", "", self.s["examples.intro"], ""]
        for name in self.examples:
            text = (self.root / "examples" / name).read_text(encoding="utf-8")
            lang = fence_lang.get(Path(name).suffix, "")
            ticks = "````" if "```" in text else "```"
            parts += [f"## `{name}`", "", f"{ticks}{lang}", text.rstrip("\n"), ticks, ""]
        return "\n".join(parts)

    def docs_page(self, page: Page, titles: dict[str, str]) -> str:
        content, env = self.render(self.page_text(page), page)
        title = env.get("title") or self.label(page)
        toc = self.toc(env["toc"])
        if toc:
            mobile = (f'<details class="tocm"><summary>{esc(self.s["toc.title"])}</summary>'
                      f'<nav aria-label="{esc(self.s["toc.title"])}">{toc}</nav></details>')
            content = content.replace("</h1>\n", "</h1>\n" + mobile, 1) if "</h1>" in content else mobile + content
            aside = (f'<aside class="toc"><nav aria-label="{esc(self.s["toc.title"])}">'
                     f'<p class="toc-title">{esc(self.s["toc.title"])}</p>{toc}</nav></aside>')
        else:
            aside = '<aside class="toc" aria-hidden="true"></aside>'
        source_url = self.github(page.source) if page.source else self.github("examples", directory=True)
        nav = self.nav(page, titles)
        body = fill(template("docs.html"), {
            "nav": nav,
            "nav_aria": esc(self.s["nav.aria"]),
            "menu": esc(self.s["nav.menu"]),
            "content": content,
            "toc": aside,
            "source_url": esc(source_url),
            "source": esc(self.s["page.source"]),
        })
        label = self.label(page, title)
        return self.shell(page, f"{label} · blink", self.pitch, body, env, "docs-page")

    def titles(self) -> dict[str, str]:
        out = {}
        for p in PAGES:
            if p.source:
                text = self.page_text(p)
                tokens = self.md.parse(text, new_env())
                for i, t in enumerate(tokens):
                    if t.type == "heading_open" and t.tag == "h1":
                        out[p.slug] = inline_text(tokens[i + 1]).strip()
                        break
        return out

    # --- home ----------------------------------------------------------------------------

    def readme_pitch(self) -> str:
        tokens = self.md.parse((self.root / "README.md").read_text(encoding="utf-8"), new_env())
        seen_h1 = False
        for i, t in enumerate(tokens):
            if t.type == "heading_open" and t.tag == "h1":
                seen_h1 = True
            elif seen_h1 and t.type == "paragraph_open":
                kids = [k for k in tokens[i + 1].children or [] if not (k.type == "text" and not k.content)]
                if kids and kids[0].type == "strong_open" and kids[-1].type == "strong_close":
                    return inline_text(tokens[i + 1]).strip()
                break
        raise BuildError("README.md: the paragraph under the title is no longer one bold line")

    def readme_section(self, heading: str) -> str:
        text = (self.root / "README.md").read_text(encoding="utf-8")
        m = re.search(rf"^## {re.escape(heading)}\s*\n(.*?)(?=^## |\Z)", text, re.S | re.M)
        if not m:
            raise BuildError(f"README.md has no '## {heading}' section")
        return m.group(1).strip() + "\n"

    def space_pieces(self) -> dict[str, str]:
        """The pipeline figure and one saved answer, drawn by the Space's own code."""
        space = str(self.root / "space")
        sys.path.insert(0, space)
        try:
            import blink
            import examples
            import results
            import ui

            if Path(blink.__file__).resolve().parent != Path(space).resolve():
                raise BuildError(f"imported the wrong blink module: {blink.__file__}")

            figure = results.pipeline_figure()
            engine = blink.ReplayEngine(blink.REPLAY_PATH, blink.TEMPERATURE)
            saved, blink._ENGINE = blink._ENGINE, engine
            try:
                ex_cfg = CONFIG["example"]
                case = next(c for c in examples.USE_CASES if c.key == ex_cfg["case"])
                ex = next(e for e in case.examples if e.label == ex_cfg["label"])
                out = blink.decide(ex.state, case.questions)
            finally:
                blink._ENGINE = saved
            if out["meta"].get("engine") != "replay" or out["meta"].get("model") != engine.model_id:
                raise BuildError("the home example did not come from the saved run")
            shown = {k: case.questions[k] for k in ex_cfg["questions"]}
            answers = ui.render_answers(ex.state, shown, out)
            foot = answers.find('<div class="blk-answer-foot">')
            if not answers.startswith('<div class="blk-answers">') or foot < 0:
                raise BuildError("space/ui.py render_answers changed shape")
            return {
                "figure": figure,
                "stages": pipeline_stages(figure),
                "state": ex.state,
                "answers": answers[:foot],
                "model": engine.model_id,
                "case": case.key,
            }
        finally:
            sys.path.remove(space)

    def home_page(self, titles: dict[str, str]) -> str:
        s = self.s
        pieces = self.space_pieces()
        quick, env = self.render(self.readme_section("Quickstart"), HOME_README)
        hub = CONFIG["hub_url"] + CONFIG["repo"].split("/")[0] + "/"
        chips = "".join(f'<a class="chip key" href="{esc(hub + m)}">{esc(CONFIG["repo"].split("/")[0] + "/" + m)}</a>'
                        for m in CONFIG["models"])
        chips += f'<span class="chip flash">{esc(s["hero.chip"])}</span>'
        ctas = (f'<a class="btn primary" href="{esc(CONFIG["space_url"])}">{esc(s["home.cta_space"])}</a>'
                f'<a class="btn" href="overview/">{esc(s["home.cta_docs"])}</a>'
                f'<a class="btn" href="{esc(self.repo_url)}">{esc(s["site.github"])}</a>')
        tiles = []
        for p in MODEL_PAGES:
            name = p.slug.split("/")[-1]
            base = front_matter_value((self.root / p.source).read_text(encoding="utf-8"), "base_model") or ""
            tiles.append(
                '<div class="tile model">'
                f'<b>{esc(titles.get(p.slug, name))}</b><span>{esc(base)}</span>'
                '<div class="tile-links">'
                f'<a class="pill" href="{p.slug}/">{esc(s["home.model_card"])}</a>'
                f'<a class="pill" href="{esc(hub + name)}">{esc(s["home.model_hub"])}</a>'
                "</div></div>"
            )
        docs = []
        for p in PAGES:
            if p.group in HOME_DOC_GROUPS:
                docs.append(f'<a class="tile doc" href="{p.slug}/"><b>{esc(self.label(p, titles.get(p.slug)))}</b>'
                            '<em aria-hidden="true">\u2192</em></a>')
        steps, notes = pieces["stages"]
        stages = "".join(
            f'<li><span class="n">{i}</span><b>{esc(t)}</b><span>{esc(sub)}</span></li>'
            for i, (t, sub) in enumerate(steps, 1)
        )
        stage_notes = "".join(f"<p>{esc(n)}</p>" for n in notes)
        case_url = f'{CONFIG["space_url"]}?tab=use-cases&case={urllib.parse.quote(pieces["case"])}'
        body = fill(template("home.html"), {
            "pitch": esc(self.pitch),
            "chips": chips,
            "ctas": ctas,
            "how": esc(s["home.how"]),
            "figure": pieces["figure"],
            "stages": stages,
            "stage_notes": stage_notes,
            "state_label": esc(s["home.state"]),
            "state": esc(pieces["state"]),
            "answers": pieces["answers"],
            "caption": esc(s["home.example_caption"]),
            "case_url": esc(case_url),
            "example_link": esc(s["home.example_link"]),
            "quick_title": esc(s["home.quickstart"]),
            "quickstart": quick,
            "models_title": esc(s["nav.group.models"]),
            "model_tiles": "".join(tiles),
            "docs_title": esc(s["site.docs"]),
            "doc_tiles": "".join(docs),
        })
        return self.shell(HOME, "blink", self.pitch, body, env, "home-page")

    def not_found(self) -> str:
        base = urllib.parse.urlparse(CONFIG["site_url"]).path or "/"
        body = fill(template("404.html"), {
            "title": esc(self.s["notfound.title"]),
            "home": esc(base),
            "home_label": esc(self.s["notfound.home"]),
        })
        return self.shell(HOME, f'{self.s["notfound.title"]} · blink', self.pitch, body, new_env(),
                          "plain-page", root_prefix=base, index=False)

    # --- output --------------------------------------------------------------------------

    def write(self, rel: str, text: str) -> None:
        path = self.out / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")

    def build(self) -> dict:
        if self.out.exists():
            shutil.rmtree(self.out)
        self.out.mkdir(parents=True)
        self.pitch = self.readme_pitch()
        titles = self.titles()
        for page in PAGES:
            self.write(f"{page.slug}/index.html", self.docs_page(page, titles))
        self.write("index.html", self.home_page(titles))
        self.write("404.html", self.not_found())
        self.copy_assets()
        for name in ("site.css", "site.js", "favicon.svg"):
            dest = self.out / "assets" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.site_dir / "static" / name, dest)
        self.write(".nojekyll", "")
        self.write("sitemap.xml", sitemap([""] + [p.slug + "/" for p in PAGES]))
        return {
            "pages": len(PAGES) + 1,
            "assets": len(self.assets),
            "diagrams": len(self.diagrams.used),
            "diagrams_in_browser": len(self.diagrams.missing),
            "unlinked": self.unlinked,
        }


def new_env() -> dict:
    return {"toc": [], "slugger": Slugger(), "math": False, "mermaid": False}


def code_html(code: str, lang: str) -> str:
    cls = f' class="language-{esc(lang)}"' if lang else ""
    head = f'<div class="code-head"><span class="code-lang">{esc(lang)}</span></div>'
    return f'<div class="code">{head}<pre><code{cls}>{esc(code)}</code></pre></div>\n'


def pipeline_stages(figure: str) -> tuple[list[tuple[str, str]], list[str]]:
    """The Space figure's stages as (title, line) pairs and its closing notes, for a phone."""
    stages = []
    for g in re.findall(r'<g class="blk-stage".*?</g>', figure, re.S):
        title = re.search(r'class="blk-stage-t"[^>]*>(.*?)</text>', g, re.S)
        lines = re.findall(r'class="blk-stage-s"[^>]*>(.*?)</text>', g, re.S)
        if not title:
            raise BuildError("space/results.py pipeline_figure changed shape")
        stages.append((html.unescape(title.group(1)), " ".join(html.unescape(x) for x in lines)))
    notes = [html.unescape(x) for x in re.findall(r"<text[^>]*>(.*?)</text>", figure[figure.rfind("</g>"):], re.S)]
    if len(stages) < 2:
        raise BuildError("space/results.py pipeline_figure changed shape")
    return stages, notes


_TEMPLATES: dict[str, str] = {}


def template(name: str) -> str:
    if name not in _TEMPLATES:
        _TEMPLATES[name] = (HERE / "templates" / name).read_text(encoding="utf-8")
    return _TEMPLATES[name]


def fill(text: str, values: dict[str, str]) -> str:
    def sub(m):
        key = m.group(1)
        if key not in values:
            raise BuildError(f"template placeholder {{{{{key}}}}} has no value")
        return values[key]
    return re.sub(r"\{\{(\w+)\}\}", sub, text)


def sitemap(paths: list[str]) -> str:
    site = CONFIG["site_url"]
    urls = "".join(f"<url><loc>{esc(site + p)}</loc></url>" for p in paths)
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">{urls}</urlset>\n')


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="repository root (default: this checkout)")
    ap.add_argument("--out", type=Path, default=DEFAULT_ROOT / "_site")
    ap.add_argument("--strict", action="store_true", help="fail when a diagram has no saved SVG")
    a = ap.parse_args(argv)
    try:
        report = Site(a.root, a.out).build()
    except BuildError as exc:
        print(f"build: {exc}", file=sys.stderr)
        return 2
    print(f"built {report['pages']} pages, {report['assets']} images, {report['diagrams']} diagrams "
          f"({report['diagrams_in_browser']} drawn in the browser) into {a.out}")
    seen: dict[tuple[str, str], int] = {}
    for item in report["unlinked"]:
        seen[item] = seen.get(item, 0) + 1
    if seen:
        print(f"{len(report['unlinked'])} links name files that are not in this repository; "
              "their text is kept without a link:")
    for (slug, href), n in sorted(seen.items()):
        print(f"  {slug or 'home'}: {href}" + (f" (x{n})" if n > 1 else ""))
    if a.strict and report["diagrams_in_browser"]:
        print("build: diagrams without a saved SVG; run site/render_diagrams.py", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
